from __future__ import annotations

import io
import importlib.util
import lzma
import math
import os
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace
import types

import torch
import torch.nn.functional as F
from torch import Tensor, nn


BASE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233"
    / "train_gpt.py"
)


if "flash_attn_interface" not in sys.modules:
    # The March 22 base script expects FlashAttention 3. On cluster environments
    # where that module is unavailable, provide a compatible SDPA fallback.
    flash_attn_interface = types.ModuleType("flash_attn_interface")

    def _flash_attn_fallback(q: Tensor, k: Tensor, v: Tensor, causal: bool = True) -> Tensor:
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=None,
            is_causal=causal,
            enable_gqa=(k.size(-2) != q.size(-2)),
        )
        return y.transpose(1, 2).contiguous()

    flash_attn_interface.flash_attn_func = _flash_attn_fallback
    sys.modules["flash_attn_interface"] = flash_attn_interface

spec = importlib.util.spec_from_file_location("_frontier_base", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load frontier base script from {BASE_SCRIPT}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

CURRENT_QAT_BITS = int(os.environ.get("QAT_BITS", "0"))
CURRENT_QAT_ENABLED = bool(int(os.environ.get("QAT_ENABLED", "0")))
CURRENT_QAT_BLOCK_SIZE = int(os.environ.get("QAT_BLOCK_SIZE", "128"))
CURRENT_QAT_ONSET_SCALE = float(
    os.environ.get("QAT_ONSET_SCALE", os.environ.get("LATE_QAT_THRESHOLD", "0.15"))
)
CURRENT_LEGACY_LATE_QAT_THRESHOLD = float(os.environ.get("LATE_QAT_THRESHOLD", "0"))
CURRENT_WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "20"))
PRE_EMA_EXPORT_DIAGNOSTIC = bool(int(os.environ.get("FRONTIER_PRE_EMA_EXPORT_DIAGNOSTIC", "0")))
PRE_EMA_SNAPSHOT_PATH = Path(os.environ.get("FRONTIER_PRE_EMA_SNAPSHOT_PATH", "final_model_pre_ema.pt"))
_OPTIMAL_GAUSSIAN_SCALES: dict[int, float] = {4: 2.5139}
_PRE_EMA_SNAPSHOT_CAPTURED = False
_PRE_EMA_LOAD_STATE_CALLS = 0


def _frontier_log(msg: str) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    print(msg, flush=True)
    run_id = os.environ.get("RUN_ID")
    if not run_id:
        return
    log_path = Path("logs") / f"{run_id}.txt"
    if not log_path.parent.exists():
        return
    with open(log_path, "a", encoding="utf-8") as f:
        print(msg, file=f)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _resolved_runtime_late_qat_threshold() -> float:
    if CURRENT_QAT_BITS > 0 and not CURRENT_QAT_ENABLED:
        return CURRENT_QAT_ONSET_SCALE
    return CURRENT_LEGACY_LATE_QAT_THRESHOLD


class FrontierHyperparameters(base.Hyperparameters):
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 1536))
    qat_bits = CURRENT_QAT_BITS
    qat_enabled = CURRENT_QAT_ENABLED
    qat_onset_scale = CURRENT_QAT_ONSET_SCALE
    qat_block_size = CURRENT_QAT_BLOCK_SIZE
    late_qat_threshold = _resolved_runtime_late_qat_threshold()

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


class _FrontierCastedLinearMeta(type(base.CastedLinear)):
    def __setattr__(cls, name: str, value: object) -> None:
        if name == "_qat_enabled":
            enabled = bool(value)
            super().__setattr__(name, enabled)
            for module in tuple(getattr(cls, "_qat_instances", ())):
                module._qat_enabled_flag.fill_(enabled)
            return
        super().__setattr__(name, value)


class FrontierCastedLinear(base.CastedLinear, metaclass=_FrontierCastedLinearMeta):
    _qat_enabled: bool = False
    _qat_instances: weakref.WeakSet[FrontierCastedLinear] = weakref.WeakSet()

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer(
            "_qat_enabled_flag",
            torch.tensor(type(self)._qat_enabled, dtype=torch.bool),
            persistent=False,
        )
        type(self)._qat_instances.add(self)
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
        if self.training and self.wq is not None:
            # Use a per-module tensor gate so late-QAT stays runtime-switchable under torch.compile.
            qat_scale = self._qat_enabled_flag.to(device=w.device, dtype=w.dtype)
            w = w + (self.wq(w) - w) * qat_scale
        elif self.training and w.ndim == 2:
            # Preserve the March 22 late-onset legacy int6 STE path when int4 QAT is off.
            with torch.no_grad():
                w32 = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)
            qat_scale = self._qat_enabled_flag.to(device=w.device, dtype=w.dtype)
            w = w + (w_q - w).detach() * qat_scale
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


def _capture_pre_ema_snapshot_if_needed(model: base.GPT) -> None:
    global _PRE_EMA_LOAD_STATE_CALLS, _PRE_EMA_SNAPSHOT_CAPTURED
    if _PRE_EMA_SNAPSHOT_CAPTURED or not PRE_EMA_EXPORT_DIAGNOSTIC:
        return
    _PRE_EMA_LOAD_STATE_CALLS += 1
    target_call_index = 2 if CURRENT_WARMUP_STEPS > 0 else 1
    if _PRE_EMA_LOAD_STATE_CALLS != target_call_index:
        return
    snapshot = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    torch.save(snapshot, PRE_EMA_SNAPSHOT_PATH)
    _PRE_EMA_SNAPSHOT_CAPTURED = True
    _frontier_log(
        f"pre_ema_snapshot:captured path={PRE_EMA_SNAPSHOT_PATH} load_state_dict_call={_PRE_EMA_LOAD_STATE_CALLS}"
    )


_ORIGINAL_GPT_LOAD_STATE_DICT = base.GPT.load_state_dict


def _frontier_gpt_load_state_dict(self, *args, **kwargs):
    _capture_pre_ema_snapshot_if_needed(self)
    return _ORIGINAL_GPT_LOAD_STATE_DICT(self, *args, **kwargs)


def _run_pre_ema_export_diagnostic(args: FrontierHyperparameters) -> None:
    if not PRE_EMA_EXPORT_DIAGNOSTIC:
        return
    if int(os.environ.get("RANK", "0")) != 0:
        return
    if not PRE_EMA_SNAPSHOT_PATH.exists():
        _frontier_log(f"pre_ema_export:skipped snapshot_missing path={PRE_EMA_SNAPSHOT_PATH}")
        return
    if not torch.cuda.is_available():
        _frontier_log("pre_ema_export:skipped cuda_unavailable")
        return

    torch.cuda.empty_cache()
    device = torch.device("cuda", 0)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    grad_accum_steps = 8 // max(world_size, 1)
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = base.load_validation_tokens(args.val_files, val_seq_len)
    sp = base.spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = base.build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    state = torch.load(PRE_EMA_SNAPSHOT_PATH, map_location="cpu")
    export_sd = {k: v for k, v in state.items() if "mtp_heads" not in k}
    if export_sd.keys() != state.keys():
        excluded_mtp = sum(int(t.numel()) for k, t in state.items() if "mtp_heads" in k)
        _frontier_log(f"pre_ema_export:excluding_mtp_params:{excluded_mtp}")

    eval_model = base.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0,
        mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
    ).to(device).bfloat16()
    for module in eval_model.modules():
        if isinstance(module, base.CastedLinear):
            module.float()
    base.restore_low_dim_params_to_fp32(eval_model)
    base.CastedLinear._qat_enabled = False
    eval_model.load_state_dict(export_sd, strict=True)

    torch.cuda.synchronize()
    t_raw = base.time.perf_counter()
    raw_val_loss, raw_val_bpb = base.eval_val(
        args,
        eval_model,
        0,
        1,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    _frontier_log(
        f"pre_ema_roundtrip_source val_loss:{raw_val_loss:.4f} val_bpb:{raw_val_bpb:.4f} "
        f"eval_time:{1000.0 * (base.time.perf_counter() - t_raw):.0f}ms"
    )
    _frontier_log(
        f"pre_ema_roundtrip_source_exact val_loss:{raw_val_loss:.8f} val_bpb:{raw_val_bpb:.8f}"
    )

    raw_model_path = PRE_EMA_SNAPSHOT_PATH.with_suffix(".export.pt")
    torch.save(export_sd, raw_model_path)
    raw_model_bytes = raw_model_path.stat().st_size
    code_bytes = len(Path(__file__).read_text(encoding="utf-8").encode("utf-8"))
    _frontier_log(f"pre_ema_serialized_model: {raw_model_bytes} bytes")
    _frontier_log(f"pre_ema_total_submission_size: {raw_model_bytes + code_bytes} bytes")

    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    quant_result, quant_meta = base.mixed_quantize_int6(sd_cpu, {"mlp", "attn"})
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_blob = base.zlib.compress(quant_buf.getvalue(), 9)
    quant_path = PRE_EMA_SNAPSHOT_PATH.with_suffix(".int6.ptz")
    with open(quant_path, "wb") as f:
        f.write(quant_blob)
    quant_file_bytes = quant_path.stat().st_size
    _frontier_log(f"pre_ema_serialized_model_int6+{base._COMPRESSOR}: {quant_file_bytes} bytes")
    _frontier_log(f"pre_ema_total_submission_size_int6+{base._COMPRESSOR}: {quant_file_bytes + code_bytes} bytes")

    quant_state = torch.load(io.BytesIO(base.zlib.decompress(quant_blob)), map_location="cpu")
    deq_state = base.dequantize_mixed_int6(quant_state["w"], quant_state["m"], sd_cpu)
    quant_eval_model = base.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0,
        mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
    ).to(device).bfloat16()
    for module in quant_eval_model.modules():
        if isinstance(module, base.CastedLinear):
            module.float()
    base.restore_low_dim_params_to_fp32(quant_eval_model)
    quant_eval_model.load_state_dict(deq_state, strict=True)

    torch.cuda.synchronize()
    t_qeval = base.time.perf_counter()
    q_val_loss, q_val_bpb = base.eval_val(
        args,
        quant_eval_model,
        0,
        1,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        eval_seq_len=effective_eval_seq_len,
    )
    torch.cuda.synchronize()
    _frontier_log(
        f"pre_ema_int6_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (base.time.perf_counter() - t_qeval):.0f}ms"
    )
    _frontier_log(f"pre_ema_int6_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")


base.Hyperparameters = FrontierHyperparameters
base.CastedLinear = FrontierCastedLinear
base.GPT.load_state_dict = _frontier_gpt_load_state_dict
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
        _run_pre_ema_export_diagnostic(args)
    finally:
        base.__file__ = original_file


if __name__ == "__main__":
    main()
