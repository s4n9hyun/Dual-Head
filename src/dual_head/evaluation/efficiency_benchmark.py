"""
Efficiency benchmark for Dual-Head models.

This module provides comprehensive efficiency evaluation comparing:
1. Dual-Head vs traditional approaches (DPO, RLHF)
2. Memory usage comparison
3. Inference speed benchmarks
4. Parameter efficiency analysis
5. Training efficiency metrics
"""

import os
import time
import json
import logging
import psutil
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

from ..inference import DualHeadInference
from ..dual_head_model import DualHeadModel

logger = logging.getLogger(__name__)


@dataclass
class EfficiencyMetrics:
    """Container for efficiency metrics."""
    model_name: str
    parameter_count: Dict[str, int]
    memory_usage: Dict[str, float]  # in GB
    inference_speed: Dict[str, float]  # tokens/second
    training_efficiency: Dict[str, float]
    comparison_baselines: Dict[str, Dict[str, float]]


class EfficiencyBenchmark:
    """
    Comprehensive efficiency benchmark for Dual-Head models.
    
    This benchmark evaluates:
    - Parameter efficiency vs full model fine-tuning
    - Memory efficiency during training and inference
    - Inference speed across different batch sizes
    - Training speed comparison
    - Hardware utilization efficiency
    """
    
    def __init__(
        self,
        dual_head_model_path: str,
        baseline_model_paths: Optional[Dict[str, str]] = None,
        device: str = "auto",
        precision: str = "fp16",
    ):
        """
        Initialize efficiency benchmark.
        
        Args:
            dual_head_model_path: Path to Dual-Head model
            baseline_model_paths: Dict of baseline model names to paths
            device: Device for benchmarking
            precision: Precision for evaluation ("fp16", "fp32", "bf16")
        """
        self.dual_head_model_path = dual_head_model_path
        self.baseline_model_paths = baseline_model_paths or {}
        self.device = device
        self.precision = precision
        
        # Initialize models
        self.models = {}
        self._load_models()
        
        # Benchmark configurations
        self.batch_sizes = [1, 2, 4, 8, 16]
        self.sequence_lengths = [128, 256, 512, 1024, 2048]
        self.generation_lengths = [50, 100, 200, 500]
    
    def _load_models(self):
        """Load all models for benchmarking."""
        logger.info("Loading models for efficiency benchmark...")
        
        # Load Dual-Head model
        try:
            self.models["dual_head"] = DualHeadInference(
                model_path=self.dual_head_model_path,
                device=self.device,
            )
            logger.info("Loaded Dual-Head model")
        except Exception as e:
            logger.error(f"Failed to load Dual-Head model: {e}")
        
        # Load baseline models
        for name, path in self.baseline_model_paths.items():
            try:
                tokenizer = AutoTokenizer.from_pretrained(path)
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=torch.float16 if self.precision == "fp16" else torch.float32,
                    device_map="auto" if self.device == "auto" else None,
                )
                self.models[name] = {"model": model, "tokenizer": tokenizer}
                logger.info(f"Loaded baseline model: {name}")
            except Exception as e:
                logger.warning(f"Failed to load baseline model {name}: {e}")
    
    def run_full_benchmark(
        self,
        save_path: Optional[str] = None,
        generate_plots: bool = True,
    ) -> EfficiencyMetrics:
        """
        Run complete efficiency benchmark.
        
        Args:
            save_path: Path to save benchmark results
            generate_plots: Whether to generate performance plots
            
        Returns:
            EfficiencyMetrics with comprehensive results
        """
        logger.info("Starting comprehensive efficiency benchmark...")
        
        # 1. Parameter efficiency analysis
        logger.info("Analyzing parameter efficiency...")
        parameter_metrics = self._analyze_parameter_efficiency()
        
        # 2. Memory usage benchmark
        logger.info("Benchmarking memory usage...")
        memory_metrics = self._benchmark_memory_usage()
        
        # 3. Inference speed benchmark
        logger.info("Benchmarking inference speed...")
        speed_metrics = self._benchmark_inference_speed()
        
        # 4. Training efficiency estimation
        logger.info("Analyzing training efficiency...")
        training_metrics = self._analyze_training_efficiency()
        
        # 5. Compare with baselines
        logger.info("Comparing with baseline models...")
        baseline_comparisons = self._compare_with_baselines()
        
        # Create efficiency metrics object
        efficiency_metrics = EfficiencyMetrics(
            model_name=self.dual_head_model_path,
            parameter_count=parameter_metrics,
            memory_usage=memory_metrics,
            inference_speed=speed_metrics,
            training_efficiency=training_metrics,
            comparison_baselines=baseline_comparisons,
        )
        
        # Save results
        if save_path:
            self._save_benchmark_results(efficiency_metrics, save_path)
        
        # Generate plots
        if generate_plots and save_path:
            plot_dir = Path(save_path).parent / "efficiency_plots"
            self._generate_efficiency_plots(efficiency_metrics, plot_dir)
        
        # Log summary
        self._log_benchmark_summary(efficiency_metrics)
        
        logger.info("Efficiency benchmark completed")
        return efficiency_metrics
    
    def _analyze_parameter_efficiency(self) -> Dict[str, int]:
        """Analyze parameter efficiency of Dual-Head model."""
        if "dual_head" not in self.models:
            logger.warning("Dual-Head model not loaded. Cannot analyze parameter efficiency.")
            return {
                "total_parameters": 0,
                "trainable_parameters": 0,
                "frozen_parameters": 0,
                "efficiency_ratio": 0.0,
                "head_parameters": 0,
                "gating_parameters": 0,
                "error": "Model not loaded"
            }
        
        # Get Dual-Head model instance
        dual_head_inference = self.models["dual_head"]
        model = dual_head_inference.model
        
        # Count parameters
        param_stats = model.get_parameter_statistics()
        
        # Calculate efficiency ratios
        total_params = param_stats["total_parameters"]
        trainable_params = param_stats["trainable_parameters"]
        frozen_params = param_stats["frozen_parameters"]
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": frozen_params,
            "efficiency_ratio": frozen_params / total_params if total_params > 0 else 0,
            "head_parameters": param_stats.get("head_parameters", 0),
            "gating_parameters": param_stats.get("gating_parameters", 0),
        }
    
    def _benchmark_memory_usage(self) -> Dict[str, float]:
        """Benchmark memory usage across different configurations."""
        memory_metrics = {}
        
        if "dual_head" not in self.models:
            logger.warning("Dual-Head model not loaded. Cannot benchmark memory usage.")
            return {
                "error": "Model not loaded",
                "mean_memory_usage": 0.0,
                "max_memory_usage": 0.0,
                "min_memory_usage": 0.0
            }
        
        dual_head_inference = self.models["dual_head"]
        
        # Test different batch sizes and sequence lengths
        for batch_size in [1, 4, 8]:
            for seq_len in [256, 512, 1024]:
                config_name = f"batch_{batch_size}_seq_{seq_len}"
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                
                # Generate test input
                test_messages = [
                    [{"role": "user", "content": "Test prompt " * (seq_len // 20)}]
                    for _ in range(batch_size)
                ]
                
                try:
                    # Run inference
                    start_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                    
                    for messages in test_messages:
                        _ = dual_head_inference.generate_text(
                            messages=messages,
                            max_new_tokens=50,
                            temperature=0.7,
                        )
                    
                    peak_memory = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                    memory_used = peak_memory - start_memory
                    
                    memory_metrics[config_name] = memory_used
                    
                except Exception as e:
                    logger.warning(f"Memory benchmark failed for {config_name}: {e}")
                    memory_metrics[config_name] = -1
        
        # Calculate memory statistics
        valid_measurements = [v for v in memory_metrics.values() if v >= 0]
        if valid_measurements:
            memory_metrics["mean_memory_usage"] = np.mean(valid_measurements)
            memory_metrics["max_memory_usage"] = max(valid_measurements)
            memory_metrics["min_memory_usage"] = min(valid_measurements)
        
        return memory_metrics
    
    def _benchmark_inference_speed(self) -> Dict[str, float]:
        """Benchmark inference speed across different configurations."""
        speed_metrics = {}
        
        if "dual_head" not in self.models:
            logger.warning("Dual-Head model not loaded. Cannot benchmark inference speed.")
            return {
                "error": "Model not loaded",
                "mean_tokens_per_sec": 0.0,
                "max_tokens_per_sec": 0.0,
                "min_tokens_per_sec": 0.0
            }
        
        dual_head_inference = self.models["dual_head"]
        
        # Test different generation lengths
        for gen_length in self.generation_lengths:
            speeds = []
            
            for _ in range(5):  # Multiple runs for averaging
                test_messages = [{"role": "user", "content": "Write a detailed response about artificial intelligence."}]
                
                # Warm-up run
                _ = dual_head_inference.generate_text(
                    messages=test_messages,
                    max_new_tokens=10,
                    temperature=0.7,
                )
                
                # Timed run
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                response = dual_head_inference.generate_text(
                    messages=test_messages,
                    max_new_tokens=gen_length,
                    temperature=0.7,
                )
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                # Calculate tokens per second
                generated_tokens = len(response["text"].split())  # Approximate
                generation_time = end_time - start_time
                
                if generation_time > 0:
                    tokens_per_second = generated_tokens / generation_time
                    speeds.append(tokens_per_second)
            
            if speeds:
                speed_metrics[f"tokens_per_sec_{gen_length}"] = np.mean(speeds)
        
        # Calculate overall statistics
        all_speeds = [v for k, v in speed_metrics.items() if k.startswith("tokens_per_sec_")]
        if all_speeds:
            speed_metrics["mean_tokens_per_sec"] = np.mean(all_speeds)
            speed_metrics["max_tokens_per_sec"] = max(all_speeds)
            speed_metrics["min_tokens_per_sec"] = min(all_speeds)
        
        return speed_metrics
    
    def _analyze_training_efficiency(self) -> Dict[str, float]:
        """Analyze training efficiency based on parameter counts."""
        if "dual_head" not in self.models:
            logger.warning("Dual-Head model not loaded. Cannot analyze training efficiency.")
            return {
                "parameter_efficiency_ratio": 0.0,
                "estimated_memory_reduction": 0.0,
                "estimated_training_speedup": 1.0,
                "trainable_parameters_millions": 0.0,
                "total_parameters_millions": 0.0,
                "error": "Model not loaded"
            }
        
        # Get parameter statistics
        dual_head_inference = self.models["dual_head"]
        model = dual_head_inference.model
        param_stats = model.get_parameter_statistics()
        
        total_params = param_stats["total_parameters"]
        trainable_params = param_stats["trainable_parameters"]
        
        # Estimate training efficiency
        # These are theoretical calculations based on parameter counts
        training_efficiency = {
            "parameter_efficiency_ratio": trainable_params / total_params if total_params > 0 else 0,
            "estimated_memory_reduction": 1 - (trainable_params / total_params) if total_params > 0 else 0,
            "estimated_training_speedup": total_params / trainable_params if trainable_params > 0 else 1,
            "trainable_parameters_millions": trainable_params / 1e6,
            "total_parameters_millions": total_params / 1e6,
        }
        
        return training_efficiency
    
    def _compare_with_baselines(self) -> Dict[str, Dict[str, float]]:
        """Compare Dual-Head efficiency with baseline models."""
        comparisons = {}
        
        # Get Dual-Head metrics
        if "dual_head" not in self.models:
            return comparisons
        
        dual_head_inference = self.models["dual_head"]
        dual_head_params = dual_head_inference.model.get_parameter_statistics()["total_parameters"]
        
        # Compare with each baseline
        for baseline_name, baseline_data in self.models.items():
            if baseline_name == "dual_head":
                continue
            
            try:
                baseline_model = baseline_data["model"]
                baseline_params = sum(p.numel() for p in baseline_model.parameters())
                
                # Parameter comparison
                param_ratio = dual_head_params / baseline_params if baseline_params > 0 else 0
                
                # Speed comparison (simplified)
                baseline_speed = self._benchmark_baseline_speed(baseline_data)
                dual_head_speed = self.models["dual_head"]
                
                comparisons[baseline_name] = {
                    "parameter_ratio": param_ratio,
                    "parameter_reduction": 1 - param_ratio,
                    "baseline_parameters": baseline_params,
                    "dual_head_parameters": dual_head_params,
                }
                
                if baseline_speed > 0:
                    # Add speed comparison if available
                    comparisons[baseline_name]["speed_comparison"] = baseline_speed
                
            except Exception as e:
                logger.warning(f"Failed to compare with baseline {baseline_name}: {e}")
        
        return comparisons
    
    def _benchmark_baseline_speed(self, baseline_data: Dict) -> float:
        """Benchmark speed of a baseline model."""
        try:
            model = baseline_data["model"]
            tokenizer = baseline_data["tokenizer"]
            
            # Simple speed test
            test_input = "Write a response about machine learning."
            inputs = tokenizer(test_input, return_tensors="pt")
            
            # Move to device
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                model = model.cuda()
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            # Calculate tokens per second
            generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
            generation_time = end_time - start_time
            
            if generation_time > 0:
                return generated_tokens / generation_time
            
        except Exception as e:
            logger.warning(f"Baseline speed benchmark failed: {e}")
        
        return 0.0
    
    def _save_benchmark_results(self, metrics: EfficiencyMetrics, save_path: str):
        """Save benchmark results to file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_dict = {
            "model_name": metrics.model_name,
            "parameter_count": metrics.parameter_count,
            "memory_usage": metrics.memory_usage,
            "inference_speed": metrics.inference_speed,
            "training_efficiency": metrics.training_efficiency,
            "comparison_baselines": metrics.comparison_baselines,
            "benchmark_metadata": {
                "precision": self.precision,
                "device": self.device,
                "batch_sizes_tested": self.batch_sizes,
                "sequence_lengths_tested": self.sequence_lengths,
                "generation_lengths_tested": self.generation_lengths,
            }
        }
        
        with open(save_path, "w") as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        logger.info(f"Efficiency benchmark results saved to {save_path}")
    
    def _generate_efficiency_plots(self, metrics: EfficiencyMetrics, plot_dir: Path):
        """Generate efficiency analysis plots."""
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. Parameter efficiency comparison
            self._plot_parameter_efficiency(metrics, plot_dir / "parameter_efficiency.png")
            
            # 2. Memory usage analysis
            self._plot_memory_usage(metrics, plot_dir / "memory_usage.png")
            
            # 3. Speed performance
            self._plot_speed_performance(metrics, plot_dir / "speed_performance.png")
            
            # 4. Baseline comparison
            self._plot_baseline_comparison(metrics, plot_dir / "baseline_comparison.png")
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping efficiency plots")
    
    def _plot_parameter_efficiency(self, metrics: EfficiencyMetrics, save_path: Path):
        """Plot parameter efficiency breakdown."""
        param_counts = metrics.parameter_count
        
        if not param_counts:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart of parameter breakdown
        if "trainable_parameters" in param_counts and "frozen_parameters" in param_counts:
            sizes = [param_counts["trainable_parameters"], param_counts["frozen_parameters"]]
            labels = ["Trainable", "Frozen"]
            colors = ["lightcoral", "lightblue"]
            
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title("Parameter Distribution")
        
        # Bar chart of parameter counts
        categories = ["Total", "Trainable", "Frozen", "Head", "Gating"]
        values = [
            param_counts.get("total_parameters", 0) / 1e6,
            param_counts.get("trainable_parameters", 0) / 1e6,
            param_counts.get("frozen_parameters", 0) / 1e6, 
            param_counts.get("head_parameters", 0) / 1e6,
            param_counts.get("gating_parameters", 0) / 1e6,
        ]
        
        bars = ax2.bar(categories, values, alpha=0.7)
        ax2.set_ylabel("Parameters (Millions)")
        ax2.set_title("Parameter Count Breakdown")
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            if value > 0:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{value:.1f}M', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_memory_usage(self, metrics: EfficiencyMetrics, save_path: Path):
        """Plot memory usage analysis."""
        memory_usage = metrics.memory_usage
        
        if not memory_usage:
            return
        
        # Filter out summary statistics
        config_metrics = {k: v for k, v in memory_usage.items() 
                         if k.startswith("batch_") and v >= 0}
        
        if not config_metrics:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        configs = list(config_metrics.keys())
        values = list(config_metrics.values())
        
        bars = ax.bar(configs, values, alpha=0.7)
        ax.set_ylabel("Memory Usage (GB)")
        ax.set_title("Memory Usage Across Different Configurations")
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.2f}GB', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_speed_performance(self, metrics: EfficiencyMetrics, save_path: Path):
        """Plot speed performance analysis."""
        speed_metrics = metrics.inference_speed
        
        # Filter generation length specific metrics
        gen_metrics = {k: v for k, v in speed_metrics.items() 
                      if k.startswith("tokens_per_sec_")}
        
        if not gen_metrics:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        gen_lengths = [int(k.split("_")[-1]) for k in gen_metrics.keys()]
        speeds = list(gen_metrics.values())
        
        ax.plot(gen_lengths, speeds, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel("Generation Length (tokens)")
        ax.set_ylabel("Speed (tokens/second)")
        ax.set_title("Inference Speed vs Generation Length")
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for x, y in zip(gen_lengths, speeds):
            ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                       xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_baseline_comparison(self, metrics: EfficiencyMetrics, save_path: Path):
        """Plot comparison with baseline models."""
        comparisons = metrics.comparison_baselines
        
        if not comparisons:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Parameter comparison
        models = list(comparisons.keys())
        param_ratios = [comparisons[model]["parameter_ratio"] for model in models]
        
        bars1 = ax1.bar(models, param_ratios, alpha=0.7)
        ax1.set_ylabel("Parameter Ratio (Dual-Head / Baseline)")
        ax1.set_title("Parameter Efficiency vs Baselines")
        ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Equal parameters')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # Parameter reduction percentage
        param_reductions = [comparisons[model]["parameter_reduction"] * 100 for model in models]
        
        bars2 = ax2.bar(models, param_reductions, alpha=0.7, color='green')
        ax2.set_ylabel("Parameter Reduction (%)")
        ax2.set_title("Parameter Reduction vs Baselines")
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars2, param_reductions):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _log_benchmark_summary(self, metrics: EfficiencyMetrics):
        """Log benchmark summary."""
        logger.info("=== Efficiency Benchmark Summary ===")
        logger.info(f"Model: {metrics.model_name}")
        
        logger.info("\nParameter Efficiency:")
        for key, value in metrics.parameter_count.items():
            if isinstance(value, int):
                logger.info(f"  {key}: {value:,} ({value/1e6:.1f}M)")
            else:
                logger.info(f"  {key}: {value:.4f}")
        
        logger.info("\nMemory Usage:")
        for key, value in metrics.memory_usage.items():
            if value >= 0:
                logger.info(f"  {key}: {value:.3f} GB")
        
        logger.info("\nInference Speed:")
        for key, value in metrics.inference_speed.items():
            logger.info(f"  {key}: {value:.2f}")
        
        logger.info("\nTraining Efficiency:")
        for key, value in metrics.training_efficiency.items():
            logger.info(f"  {key}: {value:.4f}")
        
        if metrics.comparison_baselines:
            logger.info("\nBaseline Comparisons:")
            for baseline, comparison in metrics.comparison_baselines.items():
                logger.info(f"  vs {baseline}:")
                for metric, value in comparison.items():
                    logger.info(f"    {metric}: {value:.4f}")
        
        logger.info("=" * 50)