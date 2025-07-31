#!/usr/bin/env python3
"""
Simple setup script for experimental comparison.
All real scripts are now in scripts/ directory.
"""

import os
import json
import argparse
from pathlib import Path

def check_setup():
    """Check if experimental setup is ready."""
    
    # Check required files
    required_files = [
        "scripts/generate_args.py",
        "scripts/generate_genarm.py", 
        "scripts/generate_dual_head.py",
        "scripts/run_experiment.py",
        "configs/experiment_config.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    print("✅ All required files present")
    
    # Check directories
    Path("results").mkdir(exist_ok=True)
    print("✅ Results directory ready")
    
    # Check configuration
    with open("configs/experiment_config.json") as f:
        config = json.load(f)
    
    print("✅ Configuration loaded successfully")
    print(f"  - Dataset: {config['experiment']['dataset']}")
    print(f"  - Max samples: {config['experiment']['max_samples']}")
    print(f"  - Base model: {config['base_model']}")
    
    return True

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Check experimental setup")
    parser.add_argument("--check", action="store_true", help="Check if setup is ready")
    
    args = parser.parse_args()
    
    if args.check:
        if check_setup():
            print("\n🎉 Setup is ready! Run experiments with:")
            print("  python scripts/run_experiment.py")
        else:
            print("\n❌ Setup incomplete")
            return 1
    else:
        print("Use --check to verify setup is ready")
        print("Run experiments with: python scripts/run_experiment.py")
    
    return 0

if __name__ == "__main__":
    exit(main())