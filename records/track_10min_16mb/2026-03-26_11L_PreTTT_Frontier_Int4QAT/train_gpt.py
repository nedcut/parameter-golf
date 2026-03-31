from __future__ import annotations

import importlib.util
import lzma
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import Tensor, nn


BASE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233"
    / "train_gpt.py"
)

spec = importlib.util.spec_from_file_location("_frontier_base", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load frontier base script from {BASE_SCRIPT}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

CURRENT_QAT_BITS = int(os.environ.get("QAT_BITS", "0"))
CURRENT_QAT_BLOCK_SIZE = int(os.environ.get("QAT_BLOCK_SIZE", "128"))
CURRENT_QAT_ONSET_SCALE = float(
    os.environ.get("QAT_ONSET_SCALE", os.environ.get("LATE_QAT_THRESHOLD", "0.15"))
)
_OPTIMAL_GAUSSIAN_SCALES: dict[int, float] = {4: 2.5139}


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


class FrontierHyperparameters(base.Hyperparameters):
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 1536))
    qat_bits = CURRENT_QAT_BITS
    qat_onset_scale = CURRENT_QAT_ONSET_SCALE
    qat_block_size = CURRENT_QAT_BLOCK_SIZE
    late_qat_threshold = float(
        os.environ.get(
            "LATE_QAT_THRESHOLD",
            str(qat_onset_scale if qat_bits > 0 else 0.15),
        )
    )

    def __init__(self):
        _validate_qat_config(self)


def _validate_qat_config(args: FrontierHyperparameters) -> None:
    if args.qat_bits == 0:
        return
    if args.qat_bits not in _OPTIMAL_GAUSSIAN_SCALES:
        supported = ", ".join(str(bits) for bits in sorted(_OPTIMAL_GAUSSIAN_SCALES))
        raise ValueError(f"QAT_BITS={args.qat_bits} is unsupported; supported values: 0, {supported}")
    if not _is_power_of_two(args.qat_block_size):
        raise ValueError(
            f"QAT_BLOCK_SIZE={args.qat_block_size} must be a positive power of two for Hadamard rotation"
        )
    quantized_widths = {
        "MODEL_DIM": args.model_dim,
        "MLP_MULT * MODEL_DIM": int(args.mlp_mult * args.model_dim),
        "BIGRAM_DIM": args.bigram_dim,
        "VE_DIM": args.ve_dim,
    }
    for name, width in quantized_widths.items():
        if width > 0 and width % args.qat_block_size != 0:
            raise ValueError(
                f"{name}={width} must be divisible by QAT_BLOCK_SIZE={args.qat_block_size} when QAT is enabled"
            )


def _build_hadamard_block(size: int) -> Tensor:
    if not _is_power_of_two(size):
        raise ValueError(f"Hadamard block size must be a positive power of two, got {size}")
    H = torch.ones(1, 1)
    while H.shape[0] < size:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(size)


def _hadamard_rotate(x: Tensor, H: Tensor) -> Tensor:
    *batch, dim = x.shape
    bs = H.shape[0]
    return (x.reshape(*batch, dim // bs, bs) @ H).reshape(*batch, dim)


class HadamardTrustQuantizer(nn.Module):
    def __init__(self, bits: int, block_size: int = 128):
        super().__init__()
        if bits not in _OPTIMAL_GAUSSIAN_SCALES:
            supported = ", ".join(str(supported_bits) for supported_bits in sorted(_OPTIMAL_GAUSSIAN_SCALES))
            raise ValueError(f"HadamardTrustQuantizer only supports bits in {{{supported}}}, got {bits}")
        self.qmax = 2 ** (bits - 1) - 1
        self.n_levels = 2 * self.qmax + 1
        self.alpha = _OPTIMAL_GAUSSIAN_SCALES[bits]
        self.trust = self.alpha / self.n_levels
        self.register_buffer("H", _build_hadamard_block(block_size), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        x_h = _hadamard_rotate(x, self.H)
        std = torch.sqrt(torch.mean(x_h ** 2, dim=-1, keepdim=True)).clamp_min(1e-8)
        scale = self.alpha * std
        step = scale / self.qmax
        xq = torch.clamp(torch.round(x_h / step), -self.qmax, self.qmax) * step
        # Always emit the quantized forward value. Only entries inside the trust
        # region receive straight-through gradients; the rest are stop-gradient.
        mask = (torch.abs(xq - x_h) <= std * self.trust).to(x_h.dtype)
        out_h = x_h * mask + (xq - x_h * mask).detach()
        return _hadamard_rotate(out_h, self.H)


class FrontierCastedLinear(base.CastedLinear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        if CURRENT_QAT_BITS > 0:
            if in_features % CURRENT_QAT_BLOCK_SIZE != 0:
                raise ValueError(
                    f"in_features={in_features} must be divisible by QAT_BLOCK_SIZE={CURRENT_QAT_BLOCK_SIZE} "
                    f"when QAT is enabled"
                )
            self.wq: HadamardTrustQuantizer | None = HadamardTrustQuantizer(
                CURRENT_QAT_BITS, CURRENT_QAT_BLOCK_SIZE
            )
        else:
            self.wq = None

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if self.wq is not None and FrontierCastedLinear._qat_enabled and self.training:
            w = self.wq(w)
        elif FrontierCastedLinear._qat_enabled and self.training and w.ndim == 2:
            # Preserve the March 22 late-onset legacy int6 STE path when int4 QAT is off.
            with torch.no_grad():
                w32 = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def _frontier_mlp_forward(self, x: Tensor) -> Tensor:
    x = F.leaky_relu(self.fc(x), negative_slope=0.5)
    return self.proj(x.square())


def _lzma_compress(payload: bytes, level: int = 9) -> bytes:
    del level  # Match the inherited zlib.compress signature without mutating stdlib zlib.
    return lzma.compress(payload, preset=6)


def _resolved_qat_mode(args: FrontierHyperparameters) -> str:
    if args.qat_bits > 0:
        return "hadamard_int4_always_on" if args.qat_enabled else "hadamard_int4_late"
    if args.qat_enabled:
        return "legacy_int6_always_on"
    if args.late_qat_threshold > 0:
        return "legacy_int6_late"
    return "off"


def _log_preflight(args: FrontierHyperparameters) -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return
    print(
        "frontier_scaffold:"
        f" qat_mode={_resolved_qat_mode(args)}"
        f" qat_bits={args.qat_bits}"
        f" qat_onset_scale={args.qat_onset_scale:.4f}"
        f" late_qat_threshold={args.late_qat_threshold:.4f}"
        f" qat_block_size={args.qat_block_size}",
        flush=True,
    )


base.Hyperparameters = FrontierHyperparameters
base.CastedLinear = FrontierCastedLinear
base.MLP.forward = _frontier_mlp_forward
base._COMPRESSOR = "lzma"
base.zlib = SimpleNamespace(compress=_lzma_compress, decompress=lzma.decompress)


def main() -> None:
    args = FrontierHyperparameters()
    _log_preflight(args)
    original_file = base.__file__
    try:
        base.__file__ = __file__
        base.main()
    finally:
        base.__file__ = original_file


if __name__ == "__main__":
    main()
