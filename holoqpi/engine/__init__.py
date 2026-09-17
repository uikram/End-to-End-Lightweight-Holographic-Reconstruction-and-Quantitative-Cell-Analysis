"""Training, evaluation and the off-axis / Gabor comparison protocol."""

from .compare import compare_modalities, run_single_modality
from .evaluator import Evaluator, composite_score, save_per_cell
from .trainer import Trainer, apply_learned_physics, load_checkpoint

__all__ = [
    "Trainer",
    "apply_learned_physics",
    "load_checkpoint",
    "Evaluator",
    "composite_score",
    "save_per_cell",
    "compare_modalities",
    "run_single_modality",
]
