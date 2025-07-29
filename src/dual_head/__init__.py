"""
Dual-Head: Compact and Efficient Alignment for Large Language Models via Dual-Head Architecture

This package implements the Dual-Head architecture for test-time alignment of large language models
with compact parameter overhead and context-aware gating mechanisms.
"""

from .dual_head_model import DualHeadModel
from .gating_mechanism import ContextAwareGating
from .heads import LanguageModelingHead, RewardModelingHead

__version__ = "0.1.0"
__all__ = [
    "DualHeadModel",
    "ContextAwareGating", 
    "LanguageModelingHead",
    "RewardModelingHead",
]