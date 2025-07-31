"""
Training components for Dual-Head model.
"""

from .losses import DualHeadLoss, DualHeadLossConfig, ARMLoss, WeightedLoss
from .trainer import DualHeadTrainer, DualHeadTrainingArguments
from .data_processing import (
    DualHeadDataCollator, 
    prepare_preference_dataset,
    prepare_sft_dataset,
    create_mixed_dataset
)

__all__ = [
    "DualHeadLoss",
    "DualHeadLossConfig", 
    "ARMLoss",
    "WeightedLoss",
    "DualHeadTrainer",
    "DualHeadTrainingArguments",
    "DualHeadDataCollator",
    "prepare_preference_dataset",
    "prepare_sft_dataset",
    "create_mixed_dataset",
]