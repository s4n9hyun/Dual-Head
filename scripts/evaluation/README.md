# Dual-Head Model Evaluation Framework

This directory contains comprehensive evaluation tools for Dual-Head models, implementing the evaluation methodology described in the ICLR 2026 paper.

## Overview

The evaluation framework provides four main evaluation components:

1. **HH-RLHF Evaluation**: Preference alignment assessment using the HH-RLHF dataset
2. **Multi-Objective Evaluation**: Comprehensive assessment across helpfulness, harmlessness, honesty, knowledge, and consistency
3. **Efficiency Benchmark**: Parameter and computational efficiency analysis
4. **Pairwise Comparison**: Head-to-head comparison with baseline models

## Quick Start

### Basic Evaluation

Run complete evaluation on a Dual-Head model:

```bash
./evaluate_dual_head.sh --model_path /path/to/dual_head_model --output_dir ./results
```

### Specific Evaluation Components

Run only specific evaluation components:

```bash
# HH-RLHF evaluation only
./evaluate_dual_head.sh --model_path /path/to/model --eval_components hh_rlhf

# Multi-objective evaluation only
./evaluate_dual_head.sh --model_path /path/to/model --eval_components multi_objective

# Efficiency benchmark only
./evaluate_dual_head.sh --model_path /path/to/model --eval_components efficiency

# Multiple components
./evaluate_dual_head.sh --model_path /path/to/model --eval_components "hh_rlhf,efficiency"
```

### Comparison with Baselines

Compare Dual-Head model with baseline models:

```bash
./evaluate_dual_head.sh \
    --model_path /path/to/dual_head_model \
    --baseline_models "/path/to/dpo_model /path/to/genarm_model" \
    --baseline_names "DPO GenARM" \
    --eval_components pairwise
```

## Evaluation Components

### 1. HH-RLHF Evaluation

Evaluates the model on the HH-RLHF test dataset, measuring:
- Win rate against reference responses
- Safety scores
- Helpfulness metrics
- Response quality

**Key Metrics:**
- `win_rate_vs_chosen`: Win rate against chosen responses
- `mean_safety_score`: Average safety score (0-1)
- `mean_helpfulness_score`: Average helpfulness score (0-1)

### 2. Multi-Objective Evaluation

Comprehensive evaluation across multiple objectives:

**Objectives:**
- **Helpfulness**: How well the model assists users
- **Harmlessness**: Safety and non-toxicity of responses
- **Honesty**: Accuracy and truthfulness
- **Knowledge**: Factual correctness (using MMLU)
- **Consistency**: Response consistency across similar inputs

**Key Metrics:**
- Individual objective scores (0-1 scale)
- `overall_average`: Average across all objectives
- Pareto analysis of trade-offs

### 3. Efficiency Benchmark

Analyzes computational and parameter efficiency:

**Metrics:**
- Parameter counts (total, trainable, frozen)
- Memory usage across different configurations
- Inference speed (tokens/second)
- Training efficiency estimates
- Comparison with baseline models

**Key Features:**
- Parameter efficiency ratio (frozen/total parameters)
- Memory usage profiling
- Speed benchmarking across generation lengths

### 4. Pairwise Comparison

Head-to-head comparison with baseline models:

**Methods:**
- Automated quality metrics comparison
- GPT-4 as judge (optional, requires API key)
- Hybrid approach combining both methods

**Outputs:**
- Win rates between model pairs
- Confidence scores
- Detailed comparison results
- Overall leaderboard

## Configuration Options

### Command Line Arguments

```bash
./evaluate_dual_head.sh [OPTIONS]
```

**Required:**
- `--model_path`: Path to the Dual-Head model

**Optional:**
- `--output_dir`: Output directory (default: ./eval_results)
- `--eval_components`: Components to run (default: all)
- `--max_samples`: Maximum samples to evaluate (default: 500)
- `--batch_size`: Batch size (default: 4)
- `--device`: Device (auto/cuda/cpu, default: auto)
- `--baseline_models`: Paths to baseline models (space-separated)
- `--baseline_names`: Names for baseline models (space-separated)

### Python Script Usage

For more advanced usage, use the Python script directly:

```bash
python run_evaluation.py \
    --model_path /path/to/model \
    --output_dir ./results \
    --eval_all \
    --generate_plots \
    --generate_report \
    --use_gpt4_judge \
    --gpt4_api_key "your-api-key"
```

## Output Files

The evaluation generates several output files:

### JSON Results
- `hh_rlhf_results.json`: HH-RLHF evaluation results
- `multi_objective_results.json`: Multi-objective evaluation results
- `efficiency_results.json`: Efficiency benchmark results
- `pairwise_results.json`: Pairwise comparison results
- `leaderboard.json`: Model leaderboard
- `evaluation_report.json`: Comprehensive report

### Analysis Files
- `evaluation_report.md`: Human-readable evaluation report
- `plots/`: Directory containing analysis plots (if enabled)
- `evaluation.log`: Detailed evaluation logs

### Example Report Structure

```markdown
# Dual-Head Model Evaluation Report

## Summary Results
| Metric | Score |
|--------|-------|
| HH RLHF Win Rate | 0.8450 |
| Overall Quality | 0.7823 |
| Parameter Efficiency | 0.9341 |
| Leaderboard Position | 1 |

## Detailed Results
- HH-RLHF Evaluation: 500 samples, 84.5% win rate
- Multi-Objective: Strong performance across all objectives
- Efficiency: 93% parameter reduction, 156 tokens/sec
- Pairwise: Ranked #1 with 78% average win rate
```

## Extending the Framework

### Adding Custom Evaluators

1. Create a new evaluator class in `src/dual_head/evaluation/`
2. Implement the required interface methods
3. Add imports to `__init__.py`
4. Update evaluation scripts

### Adding Custom Metrics

Modify the respective evaluator classes to include additional metrics in the `_compute_*_metrics` methods.

### Adding Custom Datasets

Update the `_load_evaluation_datasets` methods in the evaluator classes to include new datasets.

## Performance Considerations

- **Memory**: Evaluation can be memory-intensive. Reduce `batch_size` if needed
- **Speed**: Use `--max_samples` to limit evaluation size for faster runs
- **GPU**: Set `--device cuda` for GPU acceleration
- **Parallel**: Some components support parallel processing automatically

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or max samples
2. **Missing Dependencies**: Install required packages (`pip install -r requirements.txt`)
3. **Model Loading Errors**: Ensure model path is correct and model is compatible
4. **Dataset Loading Errors**: Check internet connection for dataset downloads

### Debug Mode

Add debug logging by setting environment variable:
```bash
export PYTHONPATH="${PYTHONPATH}:src"
CUDA_LAUNCH_BLOCKING=1 python run_evaluation.py --model_path /path/to/model --eval_hh_rlhf
```

## Citation

If you use this evaluation framework in your research, please cite:

```bibtex
@inproceedings{dualhead2026,
  title={Dual-Head: Compact and Efficient Alignment for Large Language Models via Dual-Head Architecture},
  author={[Authors]},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```