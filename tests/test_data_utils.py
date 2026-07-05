import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data_utils


def _fake_dataset():
    fake = []
    for ineq in ["cauchy_schwarz", "bernoulli", "triangle", "jensens"]:
        for flaw in ["none", "toy_case", "logical_gap", "approximation", "computation_error"]:
            fake.append({
                "id": f"{ineq}_{flaw}",
                "inequality": ineq,
                "task_type": "relation",
                "ground_truth": "(D) >=",
                "proof_type": "good" if flaw == "none" else "bad",
                "flaw_type": flaw,
                "prompt_used": f"prompt for {ineq} {flaw}",
                "generated_proof": f"proof text for {ineq} {flaw}",
            })
    return fake


def test_split_dataset_partitions_by_inequality():
    train, test = data_utils.split_dataset(_fake_dataset())
    assert len(train) == 10
    assert len(test) == 10
    assert {d["inequality"] for d in train} == data_utils.TRAIN_INEQUALITIES
    assert {d["inequality"] for d in test} == data_utils.TEST_INEQUALITIES
    # no overlap
    assert set(d["id"] for d in train).isdisjoint(d["id"] for d in test)


def test_split_dataset_raises_on_unknown_inequality():
    fake = _fake_dataset()
    fake.append({
        "id": "mystery_none", "inequality": "mystery_inequality", "task_type": "relation",
        "ground_truth": "(A) <", "proof_type": "good", "flaw_type": "none",
        "prompt_used": "p", "generated_proof": "proof",
    })
    try:
        data_utils.split_dataset(fake)
        assert False, "expected ValueError for unassigned inequality"
    except ValueError:
        pass


def test_train_val_split_disjoint_and_covers_train():
    train, _ = data_utils.split_dataset(_fake_dataset())
    gepa_train, gepa_val = data_utils.train_val_split(train)
    assert len(gepa_train) + len(gepa_val) == len(train)
    assert set(d["id"] for d in gepa_train).isdisjoint(d["id"] for d in gepa_val)


def test_to_dspy_examples_default_fields():
    _, test = data_utils.split_dataset(_fake_dataset())
    examples = data_utils.to_dspy_examples(test)
    assert len(examples) == 10
    inputs = examples[0].inputs()
    assert set(inputs.keys()) == {"problem", "proof"}


def test_to_dspy_examples_with_ground_truth_input():
    _, test = data_utils.split_dataset(_fake_dataset())
    examples = data_utils.to_dspy_examples(test, input_fields=("problem", "proof", "ground_truth"))
    inputs = examples[0].inputs()
    assert set(inputs.keys()) == {"problem", "proof", "ground_truth"}


def test_to_dspy_examples_flaw_labels_are_mutually_exclusive():
    _, test = data_utils.split_dataset(_fake_dataset())
    examples = data_utils.to_dspy_examples(test)
    for ex in examples:
        flags = [ex.uses_toy_case, ex.has_logical_gap, ex.uses_illegal_approximation, ex.has_computation_error]
        # a "good" (flaw_type=none) example has all four False; a "bad" example has exactly one True
        assert sum(flags) in (0, 1)


def test_to_dspy_examples_is_equivalent_always_true():
    # every synthetic proof (even flawed ones) is instructed to conclude with the
    # correct final answer -- the injected flaw is in the reasoning, not the conclusion
    _, test = data_utils.split_dataset(_fake_dataset())
    examples = data_utils.to_dspy_examples(test)
    assert all(ex.is_equivalent is True for ex in examples)
