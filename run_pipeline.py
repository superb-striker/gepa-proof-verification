import argparse
import json
import os
import subprocess
import sys
import time


def print_header(title):
    print("\n" + "=" * 90)
    print(f" STAGE: {title}")
    print("=" * 90 + "\n")

def run_script(script_path):
    """Executes a python script as a subprocess, run from src/ so relative artifact paths resolve."""
    full_path = os.path.join("src", script_path)
    if not os.path.exists(full_path):
        print(f" Error: Could not find '{full_path}'. Make sure you are running this from the project root.")
        sys.exit(1)
    try:
        subprocess.check_call([sys.executable, script_path], cwd="src")
    except subprocess.CalledProcessError as e:
        print(f"\n Pipeline failed during execution of src/{script_path}.")
        print(f"Error code: {e.returncode}")
        sys.exit(1)

def display_saved_metrics():
    """Reads the saved JSON metrics/CI/run-log and prints a final summary,
    including which optimizer actually succeeded per judge and CI-qualified deltas."""
    results_dir = os.path.join("artifacts", "evaluation_results")
    paths = {
        "baseline": os.path.join(results_dir, "baseline_metrics.json"),
        "gepa": os.path.join(results_dir, "gepa_metrics.json"),
        "bootstrap": os.path.join(results_dir, "bootstrap_metrics.json"),
        "gepa_train": os.path.join(results_dir, "gepa_train_metrics.json"),
        "ci": os.path.join(results_dir, "bootstrap_ci_results.json"),
        "run_log": os.path.join(results_dir, "optimizer_run_log.json"),
    }
    if not os.path.exists(paths["baseline"]) or not os.path.exists(paths["gepa"]):
        print(" Metrics files not found. Ensure src/evaluator.py ran successfully.")
        return
    loaded = {k: (json.load(open(p)) if os.path.exists(p) else {}) for k, p in paths.items()}
    print_header("FINAL METRICS: BASELINE vs. GEPA vs. BOOTSTRAPFEWSHOT, ON HELD-OUT TEST INEQUALITIES")
    print("Test set: triangle inequality + Jensen's inequality — never used for optimization.")
    print("Train set (cauchy_schwarz, bernoulli) was used only to compile/optimize the judges.\n")
    if loaded["run_log"].get("judges"):
        print("Optimizer status per judge:")
        for judge, statuses in loaded["run_log"]["judges"].items():
            gepa_status = statuses.get("gepa", "not run")
            bs_status = statuses.get("bootstrap", "not run")
            gepa_flag = "✅" if gepa_status == "ok" else f"⚠️  {gepa_status}"
            bs_flag = "✅" if bs_status == "ok" else f"⚠️  {bs_status}"
            print(f"  {judge:<22} GEPA: {gepa_flag}   Bootstrap: {bs_flag}")
        print()
    print(f"{'Judge':<22}{'Baseline F1':>13}{'GEPA F1':>10}{'Bootstrap F1':>14}{'In-dist F1':>12}{'GEPA CI (95%)':>20}")
    print("-" * 90)
    n_test = None
    for judge in loaded["baseline"]:
        b = loaded["baseline"][judge]["f1"]
        g = loaded["gepa"].get(judge, {}).get("f1", float("nan"))
        bs = loaded["bootstrap"].get(judge, {}).get("f1", float("nan"))
        t = loaded["gepa_train"].get(judge, {}).get("f1", None)
        ci = loaded["ci"].get(judge)
        n_test = loaded["baseline"][judge].get("n", n_test)
        t_str = f"{t:>11.1f}" if t is not None else f"{'n/a':>11}"
        ci_str = f"[{ci['ci_low']:+.1f}, {ci['ci_high']:+.1f}]" if ci else "n/a"
        single_class_flag = " *" if loaded["baseline"][judge].get("single_class") else ""
        print(f"{judge+single_class_flag:<22}{b:>13}{g:>10}{bs:>14}{t_str}{ci_str:>20}")
    print("-" * 90)
    print("\n* = single-class dataset (every label is True by construction — see README); F1/precision/recall")
    print("    are not meaningful for this judge, refer to 'accuracy' in the raw JSON instead.")
    print(f"\nHeld-out test set size: n={n_test} per judge — small-sample demonstration, not a statistically")
    print("powered benchmark (see README Limitations). The 95% CI column shows how much the GEPA-vs-baseline")
    print("delta could plausibly swing on a set this small; a CI that crosses zero means don't read the point")
    print("estimate as a confident win. 'In-dist F1' scores the same GEPA-optimized judges on the TRAIN")
    print("inequalities — a much higher in-distribution score than held-out score is itself evidence of")
    print("overfitting during optimization.")
    chart_path = os.path.join(results_dir, "f1_comparison.png")
    if os.path.exists(chart_path):
        print(f"\nChart: {chart_path}")
    print("=" * 90 + "\n")

def main():
    parser = argparse.ArgumentParser(description="IneqMath agentic evaluation pipeline")
    parser.add_argument("--skip-generation", action="store_true",
                         help="Reuse existing artifacts/data_curated/raw_proofs.json instead of calling Gemini again")
    parser.add_argument("--skip-optimization", action="store_true",
                         help="Reuse existing artifacts/compiled_judges/*.json instead of re-running GEPA/Bootstrap")
    parser.add_argument("--report-only", action="store_true",
                         help="Skip straight to printing the summary table from existing evaluation_results/*.json")
    args = parser.parse_args()
    print("Starting the IneqMath Agentic Evaluation Pipeline...")
    if not os.path.exists(".env"):
        print(" Error: '.env' file not found in the root directory.")
        print("Please create one with GEMINI_API_KEY and NVIDIA_API_KEY at minimum.")
        sys.exit(1)
    os.makedirs(os.path.join("artifacts", "data_curated"), exist_ok=True)
    os.makedirs(os.path.join("artifacts", "compiled_judges"), exist_ok=True)
    os.makedirs(os.path.join("artifacts", "evaluation_results"), exist_ok=True)
    start_time = time.time()
    if args.report_only:
        display_saved_metrics()
        return
    if not args.skip_generation:
        print_header("PHASE 1 & 2 - DATASET GENERATION")
        print("Running src/proof_generator.py (calls Gemini; ~1 request per proof, rate-limited)...")
        run_script("proof_generator.py")
    else:
        print("Skipping generation, reusing existing artifacts/data_curated/raw_proofs.json")
    if not args.skip_optimization:
        print_header("PHASE 3 - GEPA + BOOTSTRAPFEWSHOT OPTIMIZATION (both, deliberately, for comparison)")
        print("Running src/optimizers.py...")
        run_script("optimizers.py")
    else:
        print("Skipping optimization, reusing existing artifacts/compiled_judges/*.json")
    print_header("PHASE 4 - BASELINE vs. GEPA vs. BOOTSTRAP EVALUATION (held-out test set)")
    print("Running src/evaluator.py...")
    run_script("evaluator.py")
    display_saved_metrics()
    elapsed = round(time.time() - start_time, 2)
    print(f" Done. Total execution time: {elapsed} seconds.")
    print("See 'artifacts/' for the dataset, compiled judges, metrics JSONs, and f1_comparison.png.")

if __name__ == "__main__":
    main()
