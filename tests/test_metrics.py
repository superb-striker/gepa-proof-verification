import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from metrics import bootstrap_f1_delta_ci, compute_prf1


def test_perfect_predictions():
    expected = [True, False, True, False]
    predicted = [True, False, True, False]
    m = compute_prf1(expected, predicted)
    assert m["precision"] == 100.0
    assert m["recall"] == 100.0
    assert m["f1"] == 100.0
    assert m["accuracy"] == 100.0
    assert m["single_class"] is False


def test_all_wrong_predictions():
    expected = [True, False, True, False]
    predicted = [False, True, False, True]
    m = compute_prf1(expected, predicted)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
    assert m["accuracy"] == 0.0


def test_mixed_predictions_known_values():
    # 2 true positives, 1 false positive, 1 false negative, 0 true negatives
    expected = [True, True, False, True]
    predicted = [True, True, True, False]
    m = compute_prf1(expected, predicted)
    assert m["tp"] == 2
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["tn"] == 0
    # precision = 2/3, recall = 2/3, f1 = 2/3
    assert abs(m["precision"] - 66.7) < 0.5
    assert abs(m["recall"] - 66.7) < 0.5
    assert abs(m["f1"] - 66.7) < 0.5


def test_single_class_dataset_flagged():
    # every expected value is True, as with the final-answer judge on this project's dataset
    expected = [True, True, True, True]
    predicted = [True, True, False, True]
    m = compute_prf1(expected, predicted)
    assert m["single_class"] is True
    # precision is trivially 100% here (no negatives exist to be falsely flagged);
    # accuracy is the informative number for a single-class set
    assert m["precision"] == 100.0
    assert m["accuracy"] == 75.0


def test_empty_input():
    m = compute_prf1([], [])
    assert m["n"] == 0
    assert m["accuracy"] == 0.0


def test_mismatched_length_raises():
    try:
        compute_prf1([True, False], [True])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_bootstrap_ci_zero_delta_when_identical():
    expected = [True, False, True, False, True, False, True, False, True, False]
    preds = [True, False, True, False, True, False, True, False, True, False]
    ci = bootstrap_f1_delta_ci(expected, preds, preds, n_resamples=200, seed=1)
    assert ci["point_estimate"] == 0.0
    # with identical baseline and optimized predictions every resample delta is 0
    assert ci["ci_low"] == 0.0
    assert ci["ci_high"] == 0.0


def test_bootstrap_ci_positive_when_optimized_strictly_better():
    expected = [True] * 5 + [False] * 5
    baseline_preds = [False] * 10  # gets everything wrong
    optimized_preds = expected[:]   # gets everything right
    ci = bootstrap_f1_delta_ci(expected, baseline_preds, optimized_preds, n_resamples=200, seed=1)
    assert ci["point_estimate"] > 0
    assert ci["ci_low"] > 0  # CI shouldn't cross zero when the improvement is this stark and consistent


def test_bootstrap_ci_deterministic_with_seed():
    expected = [True, False, True, True, False]
    baseline_preds = [True, True, True, False, False]
    optimized_preds = [True, False, True, True, True]
    ci_a = bootstrap_f1_delta_ci(expected, baseline_preds, optimized_preds, n_resamples=500, seed=42)
    ci_b = bootstrap_f1_delta_ci(expected, baseline_preds, optimized_preds, n_resamples=500, seed=42)
    assert ci_a == ci_b
