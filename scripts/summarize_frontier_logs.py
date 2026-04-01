#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path


KV_PAT = re.compile(r"([A-Za-z0-9_+.-]+):([^\s]+)")


def coerce(value: str):
    text = value.rstrip(",")
    try:
        if any(ch in text for ch in ".eE"):
            return float(text)
        return int(text)
    except ValueError:
        return text


def parse_pairs(line: str) -> dict[str, object]:
    return {key: coerce(value) for key, value in KV_PAT.findall(line)}


def first_number(text: str) -> int | None:
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else None


def number_after_colon(text: str) -> int | None:
    match = re.search(r":\s*(\d+)", text)
    return int(match.group(1)) if match else None


def infer_mode(summary: dict[str, object]) -> str:
    qat_bits = summary.get("qat_bits")
    qat_enabled = summary.get("qat_enabled")
    onset = summary.get("qat_onset_scale")
    late = summary.get("late_qat_threshold")
    if qat_bits and int(qat_bits) > 0:
        prefix = "int4-always" if qat_enabled else "int4-late"
        return f"{prefix}@{onset}"
    if qat_enabled:
        return "legacy-int6-always"
    if late and float(late) > 0:
        return f"legacy-int6-late@{late}"
    return "noqat"


def parse_log(path: Path) -> dict[str, object]:
    summary: dict[str, object] = {
        "path": str(path),
        "run_dir": None,
        "run_id": None,
        "seed": None,
        "iterations": None,
        "warmdown_iters": None,
        "qat_bits": None,
        "qat_enabled": None,
        "qat_onset_scale": None,
        "late_qat_threshold": None,
        "step200_val_bpb": None,
        "post_ema_val_bpb": None,
        "final_float_fixed_bpb": None,
        "final_float_sliding_bpb": None,
        "pre_ema_source_bpb": None,
        "pre_ema_int6_bpb": None,
        "final_quant_fixed_bpb": None,
        "final_quant_sliding_bpb": None,
        "final_int6_bpb": None,
        "submission_size_bytes": None,
        "pre_ema_submission_size_bytes": None,
        "enable_event": None,
        "notes": [],
    }

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line.startswith("Run dir: "):
            run_dir = line.split(": ", 1)[1]
            summary["run_dir"] = run_dir
            summary["run_id"] = Path(run_dir).name
        elif line.startswith("SEED:"):
            summary["seed"] = first_number(line)
        elif line.startswith("seed:"):
            summary["seed"] = first_number(line)
        elif line.startswith("ITERATIONS:"):
            summary["iterations"] = first_number(line)
        elif line.startswith("WARMDOWN_ITERS:"):
            summary["warmdown_iters"] = first_number(line)
        elif line.startswith("QAT_BITS:"):
            summary["qat_bits"] = first_number(line)
        elif line.startswith("QAT_ENABLED:"):
            summary["qat_enabled"] = first_number(line)
        elif line.startswith("QAT_ONSET_SCALE:"):
            summary["qat_onset_scale"] = line.split(":", 1)[1].strip()
        elif line.startswith("LATE_QAT_THRESHOLD:"):
            summary["late_qat_threshold"] = line.split(":", 1)[1].strip()
        elif line.startswith("step:200/200 val_loss:"):
            summary["step200_val_bpb"] = parse_pairs(line).get("val_bpb")
        elif line.startswith("DIAGNOSTIC post_ema "):
            summary["post_ema_val_bpb"] = parse_pairs(line).get("val_bpb")
        elif line.startswith("final_float_fixed_exact "):
            summary["final_float_fixed_bpb"] = parse_pairs(line).get("val_bpb")
        elif line.startswith("final_float_sliding_exact "):
            summary["final_float_sliding_bpb"] = parse_pairs(line).get("val_bpb")
        elif line.startswith("pre_ema_roundtrip_source_exact "):
            summary["pre_ema_source_bpb"] = parse_pairs(line).get("val_bpb")
        elif line.startswith("pre_ema_int6_roundtrip_exact "):
            summary["pre_ema_int6_bpb"] = parse_pairs(line).get("val_bpb")
        elif line.startswith("final_quantized_fixed_exact "):
            summary["final_quant_fixed_bpb"] = parse_pairs(line).get("val_bpb")
        elif line.startswith("final_quantized_sliding_exact "):
            summary["final_quant_sliding_bpb"] = parse_pairs(line).get("val_bpb")
        elif line.startswith("final_int6_roundtrip_exact "):
            summary["final_int6_bpb"] = parse_pairs(line).get("val_bpb")
        elif line.startswith("Total submission size int6+"):
            summary["submission_size_bytes"] = number_after_colon(line)
        elif line.startswith("pre_ema_total_submission_size_int6+"):
            summary["pre_ema_submission_size_bytes"] = number_after_colon(line)
        elif line.startswith("qat:enabled ") or line.startswith("late_qat:enabled "):
            summary["enable_event"] = line
        elif line.startswith("pre_ema_export:skipped "):
            summary["notes"].append(line)

    summary["mode"] = infer_mode(summary)
    return summary


def render_markdown(rows: list[dict[str, object]]) -> str:
    headers = [
        "run_id",
        "mode",
        "seed",
        "iters",
        "warmdown",
        "step200_bpb",
        "float_fixed",
        "float_sliding",
        "quant_fixed",
        "quant_sliding",
        "pre_ema_int6",
        "post_ema",
        "final_int6",
        "size_bytes",
        "note",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            str(row.get("run_id") or ""),
            str(row.get("mode") or ""),
            str(row.get("seed") or ""),
            str(row.get("iterations") or ""),
            str(row.get("warmdown_iters") or ""),
            f"{row['step200_val_bpb']:.6f}" if row.get("step200_val_bpb") is not None else "",
            f"{row['final_float_fixed_bpb']:.6f}" if row.get("final_float_fixed_bpb") is not None else "",
            f"{row['final_float_sliding_bpb']:.6f}" if row.get("final_float_sliding_bpb") is not None else "",
            f"{row['final_quant_fixed_bpb']:.6f}" if row.get("final_quant_fixed_bpb") is not None else "",
            f"{row['final_quant_sliding_bpb']:.6f}" if row.get("final_quant_sliding_bpb") is not None else "",
            f"{row['pre_ema_int6_bpb']:.6f}" if row.get("pre_ema_int6_bpb") is not None else "",
            f"{row['post_ema_val_bpb']:.6f}" if row.get("post_ema_val_bpb") is not None else "",
            f"{row['final_int6_bpb']:.6f}" if row.get("final_int6_bpb") is not None else "",
            str(row.get("submission_size_bytes") or ""),
            str(row.get("notes", [""])[0] if row.get("notes") else row.get("enable_event") or ""),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize frontier smoke/full logs into a compact table.")
    parser.add_argument("paths", nargs="+", help="Log paths or glob patterns to summarize")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of markdown")
    args = parser.parse_args()

    resolved: list[Path] = []
    for pattern in args.paths:
        matches = [Path(p) for p in sorted(glob.glob(pattern))]
        if matches:
            resolved.extend(matches)
        else:
            resolved.append(Path(pattern))

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)

    rows = [parse_log(path) for path in unique_paths]
    rows.sort(key=lambda row: (str(row.get("run_id") or row["path"])))

    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print(render_markdown(rows))


if __name__ == "__main__":
    main()
