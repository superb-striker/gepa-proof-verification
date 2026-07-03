"""
Evaluates baseline, GEPA-optimized, and BootstrapFewShot-optimized judges,
all five including the final-answer judge, all scored ONLY on the held-out
test split (triangle, jensens) - inequalities none of the three variants
ever saw during optimization.

Also computes:
  - optimized-on-train metrics (in-distribution), for an overfitting check
  - a bootstrap CI on the F1 delta (optimized - baseline) per judge, since a
    single point estimate on n=10 invites over-reading noise
  - a bar chart image comparing all three variants per judge
"""

import json
import os

import dspy
from dotenv import load_dotenv

from data_utils import load_raw_dataset, split_dataset, to_dspy_examples
from dspy_modules import (
    ApproximationJudge,
    ComputationJudge,
    FinalAnswerJudge,
    LogicalGapJudge,
    ToyCaseJudge,
)
from metrics import bootstrap_f1_delta_ci, compute_prf1

load_dotenv()
nim_key = os.getenv("NVIDIA_API_KEY")
if not nim_key:
    raise ValueError("NVIDIA_API_KEY not found. Please check your .env file.")

task_lm = dspy.LM(
    model=os.getenv("TASK_MODEL", "openai/meta/llama-3.1-70b-instruct"),
    api_key=nim_key,
    api_base="https://integrate.api.nvidia.com/v1",
)
dspy.settings.configure(lm=task_lm)

# name -> (module class, target output field, input fields the module's forward() needs)
JUDGES = {
    "final_answer_judge": (FinalAnswerJudge, "is_equivalent", ("problem", "proof", "ground_truth")),
    "toy_case_judge": (ToyCaseJudge, "uses_toy_case", ("problem", "proof")),
    "logical_gap_judge": (LogicalGapJudge, "has_logical_gap", ("problem", "proof")),
    "approximation_judge": (ApproximationJudge, "uses_illegal_approximation", ("problem", "proof")),
    "computation_judge": (ComputationJudge, "has_computation_error", ("problem", "proof")),
}

COMPILED_DIR = os.path.join("artifacts", "compiled_judges")
RESULTS_DIR = os.path.join("artifacts", "evaluation_results")


def run_judge(judge, examples, input_fields):
    """Calls a judge module on each example, passing only the fields it declared as inputs."""
    predictions = []
    for ex in examples:
        kwargs = {field: getattr(ex, field) for field in input_fields}
        try:
            pred = judge(**kwargs)
        except Exception as e:
            print(f"    Judge call failed on example, treating as False: {e}")
            pred = dspy.Prediction()
        predictions.append(pred)
    return predictions


def evaluate_variant(variant, raw_examples):
    """
    variant: "baseline", "gepa", or "bootstrap".
    Builds the correctly-shaped dspy examples per judge (final_answer_judge
    needs an extra `ground_truth` input the others don't) and returns
    (metrics_dict, expected_by_judge, predicted_by_judge) so callers can also
    compute bootstrap CIs from the raw predictions.
    """
    results = {}
    expected_by_judge = {}
    predicted_by_judge = {}
    examples_cache = {}

    for name, (judge_cls, field_name, input_fields) in JUDGES.items():
        judge = judge_cls()
        if variant in ("gepa", "bootstrap"):
            compiled_path = os.path.join(COMPILED_DIR, f"{name}_{variant}.json")
            if not os.path.exists(compiled_path):
                print(f"  {name}: no compiled {variant} artifact found, skipping")
                continue
            judge.load(compiled_path)

        if input_fields not in examples_cache:
            examples_cache[input_fields] = to_dspy_examples(raw_examples, input_fields=input_fields)
        examples = examples_cache[input_fields]

        preds = run_judge(judge, examples, input_fields)
        expected = [getattr(ex, field_name) for ex in examples]
        actual = [getattr(p, field_name, None) for p in preds]

        m = compute_prf1(expected, actual)
        results[name] = m
        expected_by_judge[name] = expected
        predicted_by_judge[name] = actual

        label = " (single-class dataset - precision/recall not meaningful, see accuracy)" if m["single_class"] else ""
        print(f"  {name}: F1={m['f1']} Acc={m['accuracy']}{label} (n={m['n']})")

    return results, expected_by_judge, predicted_by_judge


def plot_results(baseline_res, gepa_res, bootstrap_res, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping chart. `pip install matplotlib` to enable it.")
        return

    judges = list(JUDGES.keys())
    baseline_f1 = [baseline_res.get(j, {}).get("f1", 0) for j in judges]
    gepa_f1 = [gepa_res.get(j, {}).get("f1", 0) for j in judges]
    bootstrap_f1 = [bootstrap_res.get(j, {}).get("f1", 0) for j in judges]

    x = list(range(len(judges)))
    width = 0.27

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width for i in x], baseline_f1, width, label="Baseline")
    ax.bar(x, gepa_f1, width, label="GEPA")
    ax.bar([i + width for i in x], bootstrap_f1, width, label="BootstrapFewShot")

    ax.set_ylabel("F1 (%) on held-out test set")
    ax.set_title("Judge performance: baseline vs. GEPA vs. BootstrapFewShot\n(held-out inequalities: triangle, jensens)")
    ax.set_xticks(x)
    ax.set_xticklabels([j.replace("_judge", "") for j in judges], rotation=20, ha="right")
    ax.set_ylim(0, 105)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nChart saved to {out_path}")


def main():
    raw = load_raw_dataset()
    train_raw, test_raw = split_dataset(raw)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n BASELINE judges on HELD-OUT test set ({len(test_raw)} examples: triangle, jensens) ")
    baseline_res, baseline_exp, baseline_pred = evaluate_variant("baseline", test_raw)
    with open(os.path.join(RESULTS_DIR, "baseline_metrics.json"), "w") as f:
        json.dump(baseline_res, f, indent=2)

    print("\n GEPA-optimized judges on HELD-OUT test set ")
    gepa_res, _, gepa_pred = evaluate_variant("gepa", test_raw)
    with open(os.path.join(RESULTS_DIR, "gepa_metrics.json"), "w") as f:
        json.dump(gepa_res, f, indent=2)

    print("\n BootstrapFewShot-optimized judges on HELD-OUT test set ")
    bootstrap_res, _, _ = evaluate_variant("bootstrap", test_raw)
    with open(os.path.join(RESULTS_DIR, "bootstrap_metrics.json"), "w") as f:
        json.dump(bootstrap_res, f, indent=2)

    print("\n GEPA-optimized judges on IN-DISTRIBUTION train set "
          f"({len(train_raw)} examples: cauchy_schwarz, bernoulli) -- overfitting check only ")
    gepa_train_res, _, _ = evaluate_variant("gepa", train_raw)
    with open(os.path.join(RESULTS_DIR, "gepa_train_metrics.json"), "w") as f:
        json.dump(gepa_train_res, f, indent=2)

    #  Bootstrap CIs on the F1 delta (GEPA - baseline), per judge 
    print("\n Bootstrap 95% CIs on F1 delta (GEPA - baseline), held-out set ")
    ci_results = {}
    for name in JUDGES:
        if name not in gepa_pred or name not in baseline_exp:
            continue
        ci = bootstrap_f1_delta_ci(baseline_exp[name], baseline_pred[name], gepa_pred[name])
        ci_results[name] = ci
        print(f"  {name}: delta={ci['point_estimate']:+.1f}  "
              f"95% CI=[{ci['ci_low']:+.1f}, {ci['ci_high']:+.1f}]  (n={ci['n_examples']})")
    with open(os.path.join(RESULTS_DIR, "bootstrap_ci_results.json"), "w") as f:
        json.dump(ci_results, f, indent=2)

    #  Chart 
    plot_results(baseline_res, gepa_res, bootstrap_res, os.path.join(RESULTS_DIR, "f1_comparison.png"))

    #  Summary 
    print("\n SUMMARY (held-out test set - the headline numbers) ")
    print(f"{'Judge':<22}{'Baseline F1':>13}{'GEPA F1':>10}{'Bootstrap F1':>14}{'GEPA CI (95%)':>20}")
    for name in JUDGES:
        b = baseline_res.get(name, {}).get("f1", float("nan"))
        g = gepa_res.get(name, {}).get("f1", float("nan"))
        bs = bootstrap_res.get(name, {}).get("f1", float("nan"))
        ci = ci_results.get(name)
        ci_str = f"[{ci['ci_low']:+.1f}, {ci['ci_high']:+.1f}]" if ci else "n/a"
        print(f"{name:<22}{b:>13}{g:>10}{bs:>14}{ci_str:>20}")

    print(f"\nNote: n={len(test_raw)} held-out examples per judge. The 95% CIs above show how much the")
    print("delta could plausibly swing on a dataset this small -- a wide or zero-crossing CI means the")
    print("point estimate alone should not be read as a confident win. See README Limitations.")


if __name__ == "__main__":
    main()
