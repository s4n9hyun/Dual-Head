# Dual-Head: Compact and Efficient Alignment for Large Language Models via Dual-Head Architecture

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

This repository contains the official implementation of **Dual-Head**, a novel architecture for compact and efficient alignment of large language models, as described in our ICLR 2026 submission. The approach achieves test-time alignment with only 131M additional parameters (50× reduction compared to GenARM) while maintaining competitive performance on preference alignment tasks.

## 🚀 Overview

The Dual-Head architecture enables efficient test-time alignment through:
- **Compact 131M parameter heads** (50× reduction vs GenARM's 6.7B)
- **Context-aware dynamic gating** with multi-head attention
- **Single forward pass** inference efficiency
- **Multi-objective alignment** capabilities
- **Frozen backbone** preserving pre-trained knowledge

## ✨ Key Features

- ✅ **Parameter Efficient**: Only 131M trainable parameters for alignment
- ✅ **Context-Aware Gating**: Multi-head attention for dynamic head selection
- ✅ **Dual-Head Architecture**: LM and RM heads with frozen backbone
- ✅ **Multi-Objective Training**: Combined fluency and preference losses
- ✅ **Efficient Inference**: Single-pass generation with alignment analysis
- ✅ **Comprehensive Evaluation**: Multi-objective assessment framework
- ✅ **Multiple Backbones**: Support for LLaMA, Mistral, and other architectures
- ✅ **Production Ready**: Complete training, inference, and evaluation pipeline

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Dual-Head

# Install dependencies
pip install -e .

# For training (optional)
pip install -e ".[train]"

# For evaluation (optional)  
pip install -e ".[eval]"
```

## Quick Start

### Training

```bash
# Train Dual-Head model on HH-RLHF dataset
python scripts/training/train_dual_head.py \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --dataset_name "Anthropic/hh-rlhf" \
    --output_dir "./outputs/dual_head_llama2_7b" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-5
```

### Inference

```python
from dual_head import DualHeadModel
from transformers import AutoTokenizer

# Load model and tokenizer
model = DualHeadModel.from_pretrained("./outputs/dual_head_llama2_7b")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Generate aligned response
prompt = "How can I improve my productivity?"
response = model.generate(
    prompt,
    tokenizer=tokenizer,
    max_new_tokens=128,
    temperature=0.7
)
print(response)
```

### Evaluation

```bash
# Evaluate on HH-RLHF benchmark
python scripts/evaluation/evaluate_hh_rlhf.py \
    --model_path "./outputs/dual_head_llama2_7b" \
    --baseline_models "DPO,GenARM,ARGS" \
    --num_samples 300
```

## Architecture

The Dual-Head architecture consists of:

1. **Frozen Backbone**: Pre-trained LLM providing contextual representations
2. **Dual Heads**: Compact LM and RM heads (131M parameters each)
3. **Context-Aware Gating**: Multi-head attention mechanism for dynamic fusion
4. **Training Objective**: Multi-objective loss balancing fluency and alignment

## Results

Our approach achieves:
- **64.8% win rate** vs GenARM on HH-RLHF
- **52.3% win rate** vs DPO while maintaining test-time flexibility
- **50× parameter reduction** compared to GenARM
- **1.7× speedup** in inference compared to existing test-time methods

## Directory Structure

```
Dual-Head/
├── src/dual_head/          # Core implementation
│   ├── dual_head_model.py  # Main model class
│   ├── gating_mechanism.py # Context-aware gating
│   ├── heads.py            # LM and RM heads
│   ├── training/           # Training components
│   ├── inference/          # Inference system
│   └── evaluation/         # Evaluation utilities
├── scripts/                # Training and evaluation scripts
├── configs/                # Configuration files
├── evaluation/             # Benchmark results
└── tests/                  # Unit tests
```

## Citation

```bibtex
@inproceedings{dualhead2026,
  title={Dual-Head: Compact and Efficient Alignment for Large Language Models via Dual-Head Architecture},
  author={Anonymous},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.