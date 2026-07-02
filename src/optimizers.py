"""
Optimizes all five judges (final-answer + four flaw judges) using BOTH GEPA
and BootstrapFewShot. Both compiled variants are saved so evaluator.py can
score them separately against the same held-out test set.

Why GEPA over BootstrapFewShot, in principle: BootstrapFewShot only selects
which successful traces to inject as few-shot demonstrations - it never
rewrites the judge's instructions. GEPA runs a reflection LM over rollouts
and proposes revised natural-language instructions for the signature itself.
Given the task here is "does this proof exhibit flaw X" - a judgment call
that hinges on how precisely the flaw is defined in the prompt, not just on
pattern-matching to examples - instruction refinement is the more relevant
lever. Running both and reporting both is what actually substantiates that
claim instead of just asserting it.
"""

import json
import os

import dspy
from dotenv import load_dotenv
from dspy.teleprompt import GEPA, BootstrapFewShot

from data_utils import (
    load_raw_dataset,
    split_dataset,
    to_dspy_examples,
    train_val_split,
)
from dspy_modules import (
    ApproximationJudge,
    ComputationJudge,
    FinalAnswerJudge,
    LogicalGapJudge,
    ToyCaseJudge,
)

load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")
nim_key = os.getenv("NVIDIA_API_KEY")

if not gemini_key:
    raise ValueError("GEMINI_API_KEY not found. Please check your .env file.")
if not nim_key:
    raise ValueError("NVIDIA_API_KEY not found. Please check your .env file.")

task_lm = dspy.LM(
    model=os.getenv("TASK_MODEL", "openai/meta/llama-3.1-70b-instruct"),
    api_key=nim_key,
    api_base="https://integrate.api.nvidia.com/v1",
)

# GEPA's own guidance is to use a distinct (often stronger) reflection model than the task LM being optimized, 
# so the critique step isn't bottlenecked by the same model/quota it's trying to improve.
reflection_lm = dspy.LM(
    model=os.getenv("REFLECTION_MODEL", "gemini/gemini-2.5-flash"),
    api_key=gemini_key,
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


def make_gepa_metric(field_name):
    """
    GEPA feedback metric for one judge's output field. 
    Returns a Prediction with `score` and `feedback` (not a bare bool)
    GEPA's reflection step reads the feedback text to decide how to revise instructions,
    so a plain 0/1 signal throws away the information GEPA is designed to use.
    """
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None):
        expected = getattr(gold, field_name)
        actual = getattr(pred, field_name, None)
        correct = expected == actual
        reasoning = getattr(pred, "reasoning", getattr(pred, "assessment_reasoning", "(no reasoning captured)"))
        feedback = (
            f"Expected {field_name}={expected}, judge predicted {actual}. "
            f"Judge's stated reasoning: {reasoning}"
        )
        return dspy.Prediction(score=1.0 if correct else 0.0, feedback=feedback)
    return metric


def make_plain_metric(field_name):
    """Boolean-only metric for BootstrapFewShot, which doesn't consume feedback text."""
    def metric(example, pred, trace=None):
        return getattr(example, field_name) == getattr(pred, field_name, None)
    return metric


def run_gepa(judge_cls, field_name, gepa_train, gepa_val):
    student = judge_cls()
    optimizer = GEPA(
        metric=make_gepa_metric(field_name),
        reflection_lm=reflection_lm,
        auto="light",  # small budget appropriate to this dataset's size
        num_threads=1,
    )
    return optimizer.compile(student, trainset=gepa_train, valset=gepa_val)


def run_bootstrap(judge_cls, field_name, gepa_train, gepa_val):
    student = judge_cls()
    teleprompter = BootstrapFewShot(
        metric=make_plain_metric(field_name),
        max_bootstrapped_demos=3,
        max_labeled_demos=1,
    )
    return teleprompter.compile(student, trainset=gepa_train + gepa_val)


def optimize_judges():
    print("Loading dataset...")
    raw = load_raw_dataset()
    train_raw, test_raw = split_dataset(raw)
    print(f"Train inequalities (used for optimization): "
          f"{sorted({d['inequality'] for d in train_raw})} ({len(train_raw)} examples)")
    print(f"Held-out test inequalities (never optimized on): "
          f"{sorted({d['inequality'] for d in test_raw})} ({len(test_raw)} examples)")

    gepa_train_raw, gepa_val_raw = train_val_split(train_raw)

    os.makedirs(os.path.join("artifacts", "compiled_judges"), exist_ok=True)
    run_log = {"judges": {}}

    for name, (judge_cls, field_name, input_fields) in JUDGES.items():
        print(f"\n Compiling {name} (fields: {input_fields}) ")
        gepa_train = to_dspy_examples(gepa_train_raw, input_fields=input_fields)
        gepa_val = to_dspy_examples(gepa_val_raw, input_fields=input_fields)

        run_log["judges"][name] = {}

        # GEPA 
        try:
            compiled_gepa = run_gepa(judge_cls, field_name, gepa_train, gepa_val)
            out_path = os.path.join("artifacts", "compiled_judges", f"{name}_gepa.json")
            compiled_gepa.save(out_path)
            run_log["judges"][name]["gepa"] = "ok"
            print(f"  GEPA: saved -> {out_path}")
        except Exception as e:
            run_log["judges"][name]["gepa"] = f"failed: {e}"
            print(f"GEPA failed for {name}: {e}")

        # BootstrapFewShot (always run, for a real comparison, not just a fallback) 
        try:
            compiled_bootstrap = run_bootstrap(judge_cls, field_name, gepa_train, gepa_val)
            out_path = os.path.join("artifacts", "compiled_judges", f"{name}_bootstrap.json")
            compiled_bootstrap.save(out_path)
            run_log["judges"][name]["bootstrap"] = "ok"
            print(f"  BootstrapFewShot: saved -> {out_path}")
        except Exception as e:
            run_log["judges"][name]["bootstrap"] = f"failed: {e}"
            print(f"BootstrapFewShot failed for {name}: {e}")

    log_path = os.path.join("artifacts", "evaluation_results", "optimizer_run_log.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2)

    print(f"\nCompilation complete. Run log saved to {log_path}")
    return run_log


if __name__ == "__main__":
    optimize_judges()
