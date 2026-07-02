"""
Shared dataset utilities.

Design decision: the train/test split is done by INEQUALITY, not by random
sampling of examples. All examples for cauchy_schwarz and bernoulli are used
for GEPA optimization; all examples for triangle and jensens are held out
entirely and only ever used for final evaluation.

Why split this way instead of a random 80/20 split: a random split would let
GEPA's few-shot bootstrapping see proofs about the same inequality it's later
tested on, so a high score could just mean it memorized surface patterns of
that specific problem (e.g. "Cauchy-Schwarz answers are usually (D)"). Splitting
by inequality forces the judges to generalize to problem types they never saw
a labeled example of, which is a meaningfully harder and more honest test of
whether the optimized prompts learned the underlying flaw categories (toy-case
reasoning, logical gaps, illegal approximation, computation errors) rather than
the specific inequalities.
"""

import json
import os

import dspy

TRAIN_INEQUALITIES = {"cauchy_schwarz", "bernoulli"}
TEST_INEQUALITIES = {"triangle", "jensens"}

FLAW_FIELD_MAP = {
    "toy_case": "uses_toy_case",
    "logical_gap": "has_logical_gap",
    "approximation": "uses_illegal_approximation",
    "computation_error": "has_computation_error",
}


def load_raw_dataset():
    filepath = os.path.join("artifacts", "data_curated", "raw_proofs.json")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def split_dataset(dataset):
    """Splits raw dataset dicts into (train, test) lists by inequality identity."""
    train = [d for d in dataset if d["inequality"] in TRAIN_INEQUALITIES]
    test = [d for d in dataset if d["inequality"] in TEST_INEQUALITIES]
    unknown = [d for d in dataset if d["inequality"] not in TRAIN_INEQUALITIES | TEST_INEQUALITIES]
    if unknown:
        raise ValueError(
            f"Found examples for inequalities not assigned to a split: "
            f"{sorted({d['inequality'] for d in unknown})}. "
            f"Add them to TRAIN_INEQUALITIES or TEST_INEQUALITIES in data_utils.py."
        )
    return train, test


def to_dspy_examples(raw_examples, input_fields=("problem", "proof")):
    """
    Converts raw proof dicts into dspy.Example objects carrying all five
    boolean labels (four flaw types + final-answer equivalence) plus
    ground_truth, and marks `input_fields` as the Example's inputs.
    """
    out = []
    for data in raw_examples:
        flaw = data["flaw_type"]
        example = dspy.Example(
            problem=data["prompt_used"],
            proof=data["generated_proof"],
            ground_truth=data["ground_truth"],
            uses_toy_case=(flaw == "toy_case"),
            has_logical_gap=(flaw == "logical_gap"),
            uses_illegal_approximation=(flaw == "approximation"),
            has_computation_error=(flaw == "computation_error"),
            is_equivalent=True,
        ).with_inputs(*input_fields)
        out.append(example)
    return out


def train_val_split(train_raw, val_fraction_instructions=("approximation", "computation_error")):
    """
    Within the TRAIN inequalities, carves out a small validation set for GEPA's
    Pareto candidate selection, distinct from the examples GEPA bootstraps reflective few-shot demonstrations from.
    """
    gepa_train = [d for d in train_raw if d["flaw_type"] not in val_fraction_instructions]
    gepa_val = [d for d in train_raw if d["flaw_type"] in val_fraction_instructions]
    return gepa_train, gepa_val
