#!/usr/bin/env python3
"""
Comprehensive evaluation script for Dual-Head models.

This script runs all evaluation components:
1. HH-RLHF evaluation
2. Multi-objective evaluation
3. Efficiency benchmarking
4. Pairwise comparison with baselines

Usage:
    python run_evaluation.py --model_path /path/to/dual_head_model --output_dir ./eval_results
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dual_head.evaluation import (
    HHRLHFEvaluator,
    MultiObjectiveEvaluator,
    EfficiencyBenchmark,
    PairwiseComparator,
)

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path):
    """Setup logging configuration."""
    log_file = output_dir / "evaluation.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Dual-Head model")
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the Dual-Head model to evaluate"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name for the model (defaults to basename of model_path)"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for evaluation (auto, cuda, cpu)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Maximum number of samples to evaluate"
    )
    
    # Evaluation components
    parser.add_argument(
        "--eval_hh_rlhf",
        action="store_true",
        help="Run HH-RLHF evaluation"
    )
    parser.add_argument(
        "--eval_multi_objective",
        action="store_true", 
        help="Run multi-objective evaluation"
    )
    parser.add_argument(
        "--eval_efficiency",
        action="store_true",
        help="Run efficiency benchmark"
    )
    parser.add_argument(
        "--eval_pairwise",
        action="store_true",
        help="Run pairwise comparison"
    )
    parser.add_argument(
        "--eval_all",
        action="store_true",
        help="Run all evaluation components"
    )
    
    # Baseline models for comparison
    parser.add_argument(
        "--baseline_models",
        type=str,
        nargs="*",
        default=[],
        help="Paths to baseline models for comparison"
    )
    parser.add_argument(
        "--baseline_names",
        type=str,
        nargs="*",
        default=[],
        help="Names for baseline models"
    )
    
    # Generation arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature"
    )
    
    # Analysis options
    parser.add_argument(
        "--generate_plots",
        action="store_true",
        help="Generate analysis plots"
    )
    parser.add_argument(
        "--generate_report",
        action="store_true",
        help="Generate comprehensive evaluation report"
    )
    
    # GPT-4 evaluation (optional)
    parser.add_argument(
        "--use_gpt4_judge",
        action="store_true",
        help="Use GPT-4 as quality judge (requires API key)"
    )
    parser.add_argument(
        "--gpt4_api_key",
        type=str,
        default=None,
        help="OpenAI API key for GPT-4 evaluation"
    )
    
    return parser.parse_args()


def run_hh_rlhf_evaluation(
    model_path: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Run HH-RLHF evaluation."""
    logger.info("Starting HH-RLHF evaluation...")
    
    evaluator = HHRLHFEvaluator(
        model_path=model_path,
        device=args.device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_gpt4_judge=args.use_gpt4_judge,
        gpt4_api_key=args.gpt4_api_key,
    )
    
    results = evaluator.evaluate_full_dataset(
        max_samples=args.max_samples,
        save_path=output_dir / "hh_rlhf_results.json",
    )
    
    logger.info("HH-RLHF evaluation completed")
    return {
        "metrics": results.metrics,
        "sample_count": len(results.sample_outputs),
    }


def run_multi_objective_evaluation(
    model_path: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Run multi-objective evaluation."""
    logger.info("Starting multi-objective evaluation...")
    
    evaluator = MultiObjectiveEvaluator(
        model_path=model_path,
        device=args.device,
        batch_size=args.batch_size,
        enable_efficiency_eval=True,
    )
    
    results = evaluator.evaluate_all_objectives(
        save_path=output_dir / "multi_objective_results.json",
        generate_plots=args.generate_plots,
    )
    
    logger.info("Multi-objective evaluation completed")
    return {
        "objective_scores": results.objective_scores,
        "aggregate_scores": results.aggregate_scores,
    }


def run_efficiency_benchmark(
    model_path: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Run efficiency benchmark."""
    logger.info("Starting efficiency benchmark...")
    
    # Prepare baseline models
    baseline_paths = {}
    if args.baseline_models:
        baseline_names = args.baseline_names or [f"baseline_{i}" for i in range(len(args.baseline_models))]
        baseline_paths = dict(zip(baseline_names, args.baseline_models))
    
    benchmark = EfficiencyBenchmark(
        dual_head_model_path=model_path,
        baseline_model_paths=baseline_paths,
        device=args.device,
    )
    
    results = benchmark.run_full_benchmark(
        save_path=output_dir / "efficiency_results.json",
        generate_plots=args.generate_plots,
    )
    
    logger.info("Efficiency benchmark completed")
    return {
        "parameter_count": results.parameter_count,
        "memory_usage": results.memory_usage,
        "inference_speed": results.inference_speed,
    }


def run_pairwise_comparison(
    model_path: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Run pairwise comparison."""
    logger.info("Starting pairwise comparison...")
    
    # Prepare models for comparison
    model_paths = {args.model_name or "dual_head": model_path}
    
    if args.baseline_models:
        baseline_names = args.baseline_names or [f"baseline_{i}" for i in range(len(args.baseline_models))]
        for name, path in zip(baseline_names, args.baseline_models):
            model_paths[name] = path
    
    if len(model_paths) < 2:
        logger.warning("Pairwise comparison requires at least 2 models. Skipping...")
        return {}
    
    comparator = PairwiseComparator(
        model_paths=model_paths,
        comparison_method="automated",  # Use automated for reproducibility
        device=args.device,
        batch_size=1,  # Use smaller batch for pairwise comparison
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    
    results = comparator.compare_all_pairs(
        dataset_name="custom_prompts",
        max_comparisons=min(50, args.max_samples),  # Limit for pairwise comparison
        save_path=output_dir / "pairwise_results.json",
    )
    
    # Generate leaderboard
    leaderboard = comparator.generate_leaderboard(
        results,
        save_path=output_dir / "leaderboard.json",
    )
    
    logger.info("Pairwise comparison completed")
    return {
        "leaderboard": [(name, score) for name, score, _ in leaderboard],
        "total_comparisons": sum(
            sum(summary.total_comparisons for summary in comparisons.values())
            for comparisons in results.values()
        ),
    }


def generate_comprehensive_report(
    results: Dict[str, Dict[str, Any]],
    model_path: str,
    output_dir: Path,
    args: argparse.Namespace,
):
    """Generate comprehensive evaluation report."""
    logger.info("Generating comprehensive evaluation report...")
    
    report = {
        "model_info": {
            "model_path": model_path,
            "model_name": args.model_name or Path(model_path).name,
            "evaluation_date": str(Path().cwd()),
        },
        "evaluation_config": {
            "device": args.device,
            "max_samples": args.max_samples,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
        },
        "results": results,
    }
    
    # Add summary statistics
    summary = {}
    
    if "hh_rlhf" in results:
        hh_metrics = results["hh_rlhf"]["metrics"]
        summary["hh_rlhf_win_rate"] = hh_metrics.get("win_rate_vs_chosen", 0.0)
        summary["hh_rlhf_safety"] = hh_metrics.get("mean_safety_score", 0.0)
    
    if "multi_objective" in results:
        mo_scores = results["multi_objective"]["aggregate_scores"]
        summary["overall_quality"] = mo_scores.get("overall_average", 0.0)
        summary["consistency"] = mo_scores.get("consistency_primary_score", 0.0)
    
    if "efficiency" in results:
        eff_metrics = results["efficiency"]
        summary["parameter_efficiency"] = eff_metrics["parameter_count"].get("efficiency_ratio", 0.0)
        summary["inference_speed"] = eff_metrics["inference_speed"].get("mean_tokens_per_sec", 0.0)
    
    if "pairwise" in results:
        leaderboard = results["pairwise"]["leaderboard"]
        if leaderboard:
            dual_head_rank = next((i for i, (name, score) in enumerate(leaderboard) 
                                 if "dual_head" in name.lower()), None)
            summary["leaderboard_position"] = dual_head_rank + 1 if dual_head_rank is not None else None
    
    report["summary"] = summary
    
    # Save report
    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    # Generate markdown report
    markdown_report = generate_markdown_report(report)
    with open(output_dir / "evaluation_report.md", "w") as f:
        f.write(markdown_report)
    
    logger.info(f"Comprehensive evaluation report saved to {report_path}")


def generate_markdown_report(report: Dict[str, Any]) -> str:
    """Generate markdown evaluation report."""
    model_name = report["model_info"]["model_name"]
    
    md = f"""# Dual-Head Model Evaluation Report

## Model Information
- **Model Name**: {model_name}
- **Model Path**: {report["model_info"]["model_path"]}
- **Evaluation Date**: {report["model_info"]["evaluation_date"]}

## Evaluation Configuration
- **Device**: {report["evaluation_config"]["device"]}
- **Max Samples**: {report["evaluation_config"]["max_samples"]}
- **Batch Size**: {report["evaluation_config"]["batch_size"]}
- **Max New Tokens**: {report["evaluation_config"]["max_new_tokens"]}
- **Temperature**: {report["evaluation_config"]["temperature"]}

## Summary Results
"""
    
    summary = report.get("summary", {})
    if summary:
        md += "| Metric | Score |\n|--------|-------|\n"
        for metric, value in summary.items():
            if isinstance(value, float):
                md += f"| {metric.replace('_', ' ').title()} | {value:.4f} |\n"
            else:
                md += f"| {metric.replace('_', ' ').title()} | {value} |\n"
    
    # Add detailed results for each evaluation component
    results = report.get("results", {})
    
    if "hh_rlhf" in results:
        md += f"""
## HH-RLHF Evaluation
- **Samples Evaluated**: {results["hh_rlhf"]["sample_count"]}
- **Key Metrics**:
"""
        hh_metrics = results["hh_rlhf"]["metrics"]
        for metric, value in hh_metrics.items():
            if isinstance(value, (int, float)):
                md += f"  - {metric.replace('_', ' ').title()}: {value:.4f}\n"
    
    if "multi_objective" in results:
        md += f"""
## Multi-Objective Evaluation
- **Objective Scores**:
"""
        obj_scores = results["multi_objective"]["objective_scores"]
        for objective, scores in obj_scores.items():
            md += f"  - **{objective.title()}**:\n"
            for metric, value in scores.items():
                if isinstance(value, (int, float)):
                    md += f"    - {metric}: {value:.4f}\n"
    
    if "efficiency" in results:
        md += f"""
## Efficiency Benchmark
- **Parameter Statistics**:
"""
        param_stats = results["efficiency"]["parameter_count"]
        for metric, value in param_stats.items():
            if isinstance(value, int):
                md += f"  - {metric.replace('_', ' ').title()}: {value:,}\n"
            elif isinstance(value, float):
                md += f"  - {metric.replace('_', ' ').title()}: {value:.4f}\n"
    
    if "pairwise" in results:
        md += f"""
## Pairwise Comparison
- **Total Comparisons**: {results["pairwise"]["total_comparisons"]}
- **Leaderboard**:

| Rank | Model | Win Rate |
|------|-------|----------|
"""
        leaderboard = results["pairwise"]["leaderboard"]
        for i, (name, score) in enumerate(leaderboard):
            md += f"| {i+1} | {name} | {score:.4f} |\n"
    
    md += """
## Conclusion

This evaluation provides a comprehensive assessment of the Dual-Head model across multiple dimensions including preference alignment, safety, efficiency, and comparison with baseline models. The results demonstrate the model's performance characteristics and trade-offs in different evaluation contexts.
"""
    
    return md


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir)
    
    logger.info(f"Starting comprehensive evaluation of {args.model_path}")
    logger.info(f"Results will be saved to {output_dir}")
    
    # Determine which evaluations to run
    eval_components = []
    if args.eval_all:
        eval_components = ["hh_rlhf", "multi_objective", "efficiency", "pairwise"]
    else:
        if args.eval_hh_rlhf:
            eval_components.append("hh_rlhf")
        if args.eval_multi_objective:
            eval_components.append("multi_objective")
        if args.eval_efficiency:
            eval_components.append("efficiency")
        if args.eval_pairwise:
            eval_components.append("pairwise")
    
    if not eval_components:
        logger.error("No evaluation components specified. Use --eval_all or specific flags.")
        return
    
    # Set model name
    if not args.model_name:
        args.model_name = Path(args.model_path).name
    
    # Run evaluations
    results = {}
    
    try:
        if "hh_rlhf" in eval_components:
            results["hh_rlhf"] = run_hh_rlhf_evaluation(args.model_path, output_dir, args)
        
        if "multi_objective" in eval_components:
            results["multi_objective"] = run_multi_objective_evaluation(args.model_path, output_dir, args)
        
        if "efficiency" in eval_components:
            results["efficiency"] = run_efficiency_benchmark(args.model_path, output_dir, args)
        
        if "pairwise" in eval_components:
            results["pairwise"] = run_pairwise_comparison(args.model_path, output_dir, args)
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
    
    # Generate comprehensive report
    if args.generate_report or len(eval_components) > 1:
        generate_comprehensive_report(results, args.model_path, output_dir, args)
    
    logger.info("Evaluation completed successfully!")
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()