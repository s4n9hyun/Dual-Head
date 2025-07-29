"""
Evaluation framework for Dual-Head models.

This module provides comprehensive evaluation capabilities:
1. HH-RLHF evaluation for preference alignment assessment
2. Multi-objective evaluation across helpfulness, harmlessness, and honesty
3. Efficiency benchmarking for parameter and computational efficiency
4. Pairwise comparison with baseline models

Usage:
    from dual_head.evaluation import HHRLHFEvaluator, MultiObjectiveEvaluator
    
    # Run HH-RLHF evaluation
    evaluator = HHRLHFEvaluator(model_path="path/to/model")
    results = evaluator.evaluate_full_dataset()
    
    # Run multi-objective evaluation
    mo_evaluator = MultiObjectiveEvaluator(model_path="path/to/model")
    mo_results = mo_evaluator.evaluate_all_objectives()
"""

from .hh_rlhf_eval import HHRLHFEvaluator, EvaluationResult
from .multi_objective_eval import MultiObjectiveEvaluator, MultiObjectiveResult
from .efficiency_benchmark import EfficiencyBenchmark, EfficiencyMetrics
from .pairwise_comparison import PairwiseComparator, PairwiseResult, ComparisonSummary
from .gpt4_integration import GPT4Evaluator, create_gpt4_evaluator

__all__ = [
    # Main evaluator classes
    "HHRLHFEvaluator",
    "MultiObjectiveEvaluator", 
    "EfficiencyBenchmark",
    "PairwiseComparator",
    
    # Result classes
    "EvaluationResult",
    "MultiObjectiveResult",
    "EfficiencyMetrics",
    "PairwiseResult",
    "ComparisonSummary",
    
    # GPT-4 integration
    "GPT4Evaluator",
    "create_gpt4_evaluator",
]