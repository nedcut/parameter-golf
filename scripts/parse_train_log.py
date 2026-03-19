#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

KV_PAT = re.compile(r"([A-Za-z0-9_+.-]+):([^\s]+)")


def coerce(value: str):
    v = value.rstrip(',')
    try:
        if any(c in v for c in '.eE'):
            return float(v)
        return int(v)
    except ValueError:
        return v


def parse_line(line: str):
    pairs = KV_PAT.findall(line)
    if not pairs:
        return None
    out = {}
    for k, v in pairs:
        out[k] = coerce(v)
    return out


def main():
    if len(sys.argv) != 2:
        print('usage: parse_train_log.py <logfile>', file=sys.stderr)
        raise SystemExit(2)
    path = Path(sys.argv[1])
    text = path.read_text(encoding='utf-8', errors='replace').splitlines()

    summary = {
        'logfile': str(path),
        'train_loader': None,
        'world': None,
        'attention': None,
        'opt': None,
        'train_config': None,
        'model_params': None,
        'last_val': None,
        'final_roundtrip': None,
        'size': {},
        'peak_memory': None,
    }

    for line in text:
        if line.startswith('train_loader:dataset:'):
            summary['train_loader'] = line.strip()
        elif line.startswith('world_size:'):
            summary['world'] = parse_line(line)
        elif line.startswith('attention_mode:'):
            summary['attention'] = line.strip()
        elif line.startswith('tie_embeddings:'):
            summary['opt'] = line.strip()
        elif line.startswith('train_batch_tokens:'):
            summary['train_config'] = parse_line(line)
        elif line.startswith('model_params:'):
            summary['model_params'] = int(line.split(':', 1)[1].strip())
        elif ' val_loss:' in line and ' val_bpb:' in line and line.startswith('step:'):
            parsed = parse_line(line)
            if parsed and 'val_bpb' in parsed:
                summary['last_val'] = parsed
        elif line.startswith('Serialized model int8+zlib:'):
            summary['size']['int8_line'] = line.strip()
        elif line.startswith('Total submission size int8+zlib:'):
            summary['size']['total_int8_line'] = line.strip()
        elif line.startswith('peak memory allocated:'):
            summary['peak_memory'] = line.strip()
        elif line.startswith('final_int8_zlib_roundtrip_exact '):
            summary['final_roundtrip'] = parse_line(line)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
