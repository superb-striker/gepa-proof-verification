# Granular Proof Verification + GEPA Prompt Optimization

**Multi-agent judges that catch specific flaw types in LLM-generated inequality proofs - evaluated on inequalities they never saw during optimization.**

---

## What this is

An LLM (Gemini) generates proofs - some rigorous, some deliberately flawed in a specific way - for four classical inequalities.
A set of five DSPy judges, mirroring the granular verification approach from [IneqMath](https://ineqmath.github.io/), check for: whether the final answer is correct (`FinalAnswerJudge`), and four specific *reasoning* flaws - toy-case reasoning, logical gaps, illegal numerical approximation, and computation errors.
The judges' prompts are optimized two different ways - [GEPA](https://github.com/gepa-ai/gepa) (reflective prompt optimization) and plain `BootstrapFewShot` - run **deliberately side by side**, and both scored **on inequalities the optimizer never saw**, to measure whether the optimization actually generalizes rather than just memorizing the training proofs.

## Why granular, per-flaw judges instead of one "is this proof correct" judge

IneqMath's core finding is that a single end-to-end correctness judge misses *how* a proof is wrong - a model can flag "incorrect" without knowing whether the issue is a logical gap, an invalid approximation, or an outright computation error, which makes the judge's output far less useful as a training or debugging signal. Decomposing verification into flaw-specific judges (`ToyCaseJudge`, `LogicalGapJudge`, `ApproximationJudge`, `ComputationJudge`) means each one only has to answer a narrow, well-defined question, which is both easier to prompt for and easier to optimize independently - GEPA can improve the "does this use a toy-case shortcut" judge without needing to also relearn what a computation error looks like.

## Why the flawed proofs are synthetically generated, not hand-written

Getting labeled examples of *specific* proof flaws is expensive to source and annotate by hand. Instead, Gemini is prompted with explicit instructions to produce a proof with one deliberately injected flaw type (e.g. "solve this using only a single numerical test case, don't generalize"), which gives labeled ground truth for free. The trade-off: these are flaws an LLM was told to produce, not flaws an LLM organically makes - they're a reasonable proxy for realistic error patterns, but not identical to naturally-occurring mistakes. See **Limitations** below.

## Why the test set is two entirely different inequalities, not a random split

The original version of this project trained and evaluated on the same 10 examples (2 inequalities × 5 proof variants), which measures memorization, not generalization - a judge can score well by picking up on incidental phrasing in those specific proofs rather than learning the actual flaw pattern.

This version splits by **inequality identity**: Cauchy-Schwarz and Bernoulli's inequality are used for GEPA optimization; the Triangle inequality and Jensen's inequality are held out completely and only ever touched during final evaluation. A judge that improves on the held-out set had to learn something about what a "toy-case argument" or "illegal approximation" looks like in general, not just in the two problems it was tuned on.

## Why GEPA and BootstrapFewShot are run as a real comparison, not GEPA-with-a-silent-fallback

`BootstrapFewShot` only selects which successful rollouts to inject as few-shot demonstrations - it never touches the judge's actual instructions. GEPA runs a reflection LM over rollouts (successes and failures) and proposes revised natural-language instructions for the signature itself. Since the judging task here hinges on how precisely a flaw is *defined* in the prompt (what exactly counts as a "logical gap" vs. an acceptable proof shortcut), instruction refinement is the more relevant lever - few-shot examples alone are a weaker fit for that kind of judgment call.

That's a claim, though, not a fact - so `optimizers.py` runs **both** optimizers for every judge and saves both compiled variants (`<judge>_gepa.json`, `<judge>_bootstrap.json`), and `evaluator.py` scores both against the same held-out set. The summary table reports GEPA F1, Bootstrap F1, and baseline F1 side by side, so whether GEPA actually earns its extra cost is a number you can point to, not an assumption.

GEPA is also meaningfully more expensive (many more reflection-LM calls) and can hit free-tier quota mid-run. When it fails outright for a judge, `optimizers.py` catches it, logs `"gepa": "failed: <error>"` in `optimizer_run_log.json`, and `evaluator.py` simply skips the GEPA column for that judge rather than silently substituting the bootstrap result and calling it GEPA.

## Architecture

```
proof_generator.py  →  raw_proofs.json  →      optimizers.py       →  compiled_judges/
  (Gemini generates      (20 labeled       (GEPA AND BootstrapFewShot    <judge>_gepa.json
   4 inequalities ×        proofs)          each compile all 5 judges    <judge>_bootstrap.json
   5 flaw variants)                         on the 2 train inequalities,
                                             validating on a held-out
                                             slice of those same 2)
                                                        │
                                                        ▼
                                              evaluator.py
                              (scores baseline, GEPA, AND Bootstrap judges on the
                               2 held-out test inequalities - never seen above;
                               also computes bootstrap CIs and a comparison chart)
                                                        │
                                                        ▼
                                    evaluation_results/*.json, f1_comparison.png
```

Run the whole thing with `python run_pipeline.py`, or `python run_pipeline.py --report-only` to just reprint the last saved results without calling any APIs.

## Dataset

| Inequality | Role | Task type | Ground truth |
|---|---|---|---|
| Cauchy-Schwarz | GEPA train/val | relation | `(D) ≥` |
| Bernoulli's | GEPA train/val | bound | `C = 1` |
| Triangle inequality | **held-out test** | relation | `(B) ≤` |
| Jensen's inequality | **held-out test** | relation | `(D) ≥` |

Each inequality gets 5 generated proofs: one rigorous ("good") and four with a specific injected flaw (toy-case, logical gap, illegal approximation, computation error) - 20 proofs total, 10 train, 10 held-out test.

## Judges

| Judge | Detects | Signature | Extra input |
|---|---|---|---|
| `FinalAnswerJudge` | Whether the extracted final answer matches ground truth | `ExtractFinalAnswer` → `AssessEquivalence` | `ground_truth` |
| `ToyCaseJudge` | Proof only checks a specific numeric instance, doesn't generalize | `AssessToyCase` | - |
| `LogicalGapJudge` | Proof skips the actual derivation step | `AssessLogicalGap` | - |
| `ApproximationJudge` | Proof substitutes a decimal approximation for an exact symbolic value mid-derivation | `AssessApproximation` | - |
| `ComputationJudge` | Proof contains an arithmetic/algebraic error | `AssessComputation` | - |

**`FinalAnswerJudge`'s test set is single-class by construction, and that's reported explicitly, not hidden.** Every synthetic proof - including the flawed ones - is instructed to *conclude* with the correct final answer (see `proof_generator.py`'s `INSTRUCTIONS`); the injected flaw lives in the reasoning, not the conclusion. That's a deliberate mirror of IneqMath's point that a correct final answer doesn't imply a sound proof. It also means every `is_equivalent` label is `True`, so precision is trivially 100% and recall collapses to accuracy - `metrics.py` flags this via a `single_class` field, and the pipeline's summary table marks it with a `*` rather than presenting a meaningless F1 as if it were informative.

## Evaluation methodology

- **Metric**: precision, recall, F1, and accuracy per judge, computed only on the held-out test inequalities (see `metrics.compute_prf1`, unit-tested in `tests/test_metrics.py`).
- **Baseline**: judges with their original (unoptimized) DSPy signatures.
- **GEPA / Bootstrap**: judges compiled by each optimizer independently, loaded from `compiled_judges/<judge>_gepa.json` and `compiled_judges/<judge>_bootstrap.json`, scored on the same held-out set.
- **In-distribution comparison**: the GEPA-optimized judges are *also* scored on the train inequalities (`gepa_train_metrics.json`) purely so the gap between in-distribution and held-out performance is visible - a large gap is itself evidence of overfitting during optimization, and is reported rather than hidden.
- **Bootstrap confidence intervals**: with only 10 held-out examples per judge, a single point-estimate delta invites over-reading noise. `metrics.bootstrap_f1_delta_ci` resamples the test set with replacement (2000 resamples) and reports a 95% CI on the GEPA-minus-baseline F1 delta per judge (`bootstrap_ci_results.json`). A CI that crosses zero means the point estimate should not be read as a confident win - this is stated in the pipeline's own printed summary, not just in this README.
- **Chart**: `evaluator.py` also saves `artifacts/evaluation_results/f1_comparison.png`, a grouped bar chart of baseline/GEPA/Bootstrap F1 per judge on the held-out set.

Run `python run_pipeline.py` to regenerate all of this from scratch, or see `artifacts/evaluation_results/` for the results of the last run.

## Tests

```bash
pytest tests/
```

`tests/test_metrics.py` and `tests/test_data_utils.py` cover the split logic (train/test partition by inequality, unknown-inequality handling, train/val disjointness), the DSPy example conversion (correct input fields per judge, mutually-exclusive flaw labels), and the metric math (precision/recall/F1 against hand-computed cases, the single-class flag, and the bootstrap CI's determinism and directional correctness). None of this requires API keys - it's pure logic, which is exactly the part of the original project that had no verification at all.

## Limitations - stated plainly

- **Small dataset.** 10 train / 10 held-out examples is enough to demonstrate the methodology (granular judges, dual optimizer comparison, held-out evaluation, bootstrap CIs) but not enough for statistically significant deltas. The bootstrap CIs make this visible directly (expect wide intervals), rather than papering over it. Scaling this up would mean adding more inequalities to each split and generating more proof variants per flaw type.
- **Synthetic flaws.** The flawed proofs were generated by explicitly instructing Gemini to produce a specific error type, which is a controllable proxy for realistic mistakes but not identical to how a model naturally errs when trying (and failing) to get a proof right unprompted.
- **`FinalAnswerJudge`'s test set is single-class**, since every synthetic proof (flawed or not) concludes with the correct answer. This judge's F1/precision/recall are not meaningful - its accuracy is the number that matters, and this is flagged programmatically (`single_class: true`) rather than left for the reader to notice.
- **GEPA cost/quota.** GEPA's reflection loop makes meaningfully more LLM calls than BootstrapFewShot. On a free-tier key this can hit rate limits mid-run; when it fails for a judge, that judge's GEPA column is simply absent from the results rather than silently backfilled with a bootstrap result.
- **Two held-out inequalities is a small generalization test.** It shows whether judges generalize beyond the exact problems they were tuned on, but it's still only two novel problem instances, not a broad sweep of inequality types.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in GEMINI_API_KEY and NVIDIA_API_KEY
python run_pipeline.py
```

Useful flags:
- `--skip-generation` - reuse the existing `artifacts/data_curated/raw_proofs.json` instead of calling Gemini again
- `--skip-optimization` - reuse existing `artifacts/compiled_judges/*.json` instead of re-running GEPA/Bootstrap
- `--report-only` - just reprint the last saved summary table, no API calls at all

## Reference

Based on the granular verification approach described in [IneqMath](https://ineqmath.github.io/) (paper included in `resources/`), applied here to four classical inequalities (Cauchy-Schwarz, Bernoulli's, Triangle, Jensen's) rather than the original AM-GM case, with judge prompts optimized via [GEPA](https://arxiv.org/abs/2507.19457) instead of hand-written or few-shot-only prompts.
