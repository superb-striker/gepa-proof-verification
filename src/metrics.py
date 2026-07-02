import random


def compute_prf1(expected_values, predicted_values):
    """
    Precision/recall/F1/accuracy for a boolean flaw-detection field.
    """
    if len(expected_values) != len(predicted_values):
        raise ValueError("expected_values and predicted_values must be the same length")
    tp = fp = fn = tn = 0
    for expected, actual in zip(expected_values, predicted_values):
        if expected and actual:
            tp += 1
        elif not expected and actual:
            fp += 1
        elif expected and not actual:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    n = len(expected_values)
    accuracy = (tp + tn) / n if n else 0.0
    single_class = (tp + fn) == n or (tn + fp) == n
    return {
        "precision": round(precision * 100, 1),
        "recall": round(recall * 100, 1),
        "f1": round(f1 * 100, 1),
        "accuracy": round(accuracy * 100, 1),
        "n": n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "single_class": single_class,
    }


def bootstrap_f1_delta_ci(expected_values, baseline_preds, optimized_preds, n_resamples=2000, ci=0.95, seed=13):
    """
    Bootstrap confidence interval on the F1 delta (optimized - baseline),
    resampling test examples with replacement.
    """
    n = len(expected_values)
    if n == 0:
        return {"ci_low": None, "ci_high": None, "point_estimate": None, "n_resamples": 0}
    rng = random.Random(seed)
    indices = list(range(n))
    deltas = []
    base_f1 = compute_prf1(expected_values, baseline_preds)["f1"]
    opt_f1 = compute_prf1(expected_values, optimized_preds)["f1"]
    point_estimate = round(opt_f1 - base_f1, 1)
    for _ in range(n_resamples):
        sample_idx = [rng.choice(indices) for _ in range(n)]
        exp_s = [expected_values[i] for i in sample_idx]
        base_s = [baseline_preds[i] for i in sample_idx]
        opt_s = [optimized_preds[i] for i in sample_idx]
        b_f1 = compute_prf1(exp_s, base_s)["f1"]
        o_f1 = compute_prf1(exp_s, opt_s)["f1"]
        deltas.append(o_f1 - b_f1)
    deltas.sort()
    lower_idx = int((1 - ci) / 2 * n_resamples)
    upper_idx = int((1 - (1 - ci) / 2) * n_resamples) - 1
    upper_idx = min(upper_idx, n_resamples - 1)
    return {
        "point_estimate": point_estimate,
        "ci_low": round(deltas[lower_idx], 1),
        "ci_high": round(deltas[upper_idx], 1),
        "ci_level": ci,
        "n_resamples": n_resamples,
        "n_examples": n,
    }
