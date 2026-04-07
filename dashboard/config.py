"""Dashboard constants and configuration."""
from __future__ import annotations

from pathlib import Path

# Project root (one level up from dashboard/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Log directories
SLURM_LOG_DIR = PROJECT_ROOT / "output" / "output"
MLX_LOG_DIR = PROJECT_ROOT / "logs"
SLURM_GLOB = "pg-*.out"
MLX_GLOB = "*.txt"

# Competition constraints
SIZE_LIMIT_BYTES = 16_777_216  # 16 MB
SOTA_BPB = 1.1147  # Current best: March 25, AR Self-Gen GPTQ + XSA-all

# QAT mode color palette
MODE_COLORS = {
    "noqat": "#636EFA",
    "legacy-int6-late": "#EF553B",
    "legacy-int6-always": "#AB63FA",
    "int4-late": "#00CC96",
    "int4-always": "#FFA15A",
}

# Fallback color for unknown modes
DEFAULT_COLOR = "#B6B6B6"

# Cache TTL for data loading (seconds)
CACHE_TTL = 60
