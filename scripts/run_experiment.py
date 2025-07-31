#!/usr/bin/env python3
"""
Master experiment runner for Dual-Head vs ARGS vs GenARM comparison.
"""
import os
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

def create_results_dir(config):
    """Create timestamped results directory."""
    base_results_dir = Path(config["output"]["results_dir"])
    
    if config["output"]["create_timestamp_dir"]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = base_results_dir / timestamp
    else:
        results_dir = base_results_dir
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create latest symlink
    latest_link = base_results_dir / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(results_dir.name)
    
    return results_dir

def run_method(method_name, script_path, config_path, output_path):
    """Run a single method generation."""
    print(f"\n{'='*50}")
    print(f"Running {method_name.upper()}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run([
            "python", str(script_path), str(config_path), str(output_path)
        ], check=True, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        print(f"✅ {method_name} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {method_name} failed with return code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Run experimental comparison")
    parser.add_argument("--config", type=str, default="configs/experiment_config.json", 
                       help="Configuration file path")
    parser.add_argument("--methods", nargs="+", choices=["args", "genarm", "dual_head"], 
                       default=["args", "genarm", "dual_head"],
                       help="Methods to run (default: all)")
    parser.add_argument("--skip-dual-head-check", action="store_true",
                       help="Skip checking if Dual-Head model exists")
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {config_path}")
        return 1
    
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"Loaded configuration: {config_path}")
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Methods to run: {args.methods}")
    
    # Create results directory
    results_dir = create_results_dir(config)
    print(f"Results directory: {results_dir}")
    
    # Save config to results directory
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Check Dual-Head model if needed
    if "dual_head" in args.methods and not args.skip_dual_head_check:
        dual_head_path = Path(config["dual_head"]["checkpoint_path"])
        if not dual_head_path.exists():
            print(f"\nWARNING: Dual-Head model not found at: {dual_head_path}")
            print("Please train the model first using: ./train_with_argsearch.sh")
            print("Or use --skip-dual-head-check to ignore this check")
            return 1
    
    # Run experiments
    scripts_dir = Path("scripts")
    results = {}
    
    for method in args.methods:
        script_path = scripts_dir / f"generate_{method}.py"
        output_path = results_dir / f"{method}_responses.json"
        
        if not script_path.exists():
            print(f"ERROR: Script not found: {script_path}")
            continue
        
        success = run_method(method, script_path, config_path, output_path)
        results[method] = {
            "success": success,
            "output_file": str(output_path) if success else None
        }
    
    # Save experiment summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "methods_run": args.methods,
        "results": results,
        "success_count": sum(1 for r in results.values() if r["success"]),
        "total_methods": len(args.methods)
    }
    
    with open(results_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*50}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*50}")
    print(f"Results directory: {results_dir}")
    print(f"Successful methods: {summary['success_count']}/{summary['total_methods']}")
    
    for method, result in results.items():
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        print(f"  {method.upper()}: {status}")
    
    if summary['success_count'] == summary['total_methods']:
        print("\n🎉 All methods completed successfully!")
        return 0
    else:
        print(f"\n⚠️  {summary['total_methods'] - summary['success_count']} method(s) failed")
        return 1

if __name__ == "__main__":
    exit(main())