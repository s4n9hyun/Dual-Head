from setuptools import setup, find_packages

setup(
    name="dual-head",
    version="0.1.0",
    description="Dual-Head: Compact and Efficient Alignment for Large Language Models",
    author="ICLR 2026 Submission",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "datasets>=2.10.0",
        "peft>=0.4.0",
        "trl>=0.5.0",
        "wandb>=0.15.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.60.0",
        "einops>=0.6.0",
        "safetensors>=0.3.0",
    ],
    extras_require={
        "train": [
            "bitsandbytes>=0.40.0",
            "deepspeed>=0.9.0",
            "flash-attn>=2.0.0",
        ],
        "eval": [
            "nltk>=3.8.0",
            "rouge-score>=0.1.2",
            "openai>=1.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)