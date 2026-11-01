"""
compute_confidence_intervals.py

Turns an MRResult (estimate + se from mr_estimators.py) into a normal-
approximation confidence interval and a clean summary.

IMPORTANT — THESE ARE NAIVE, NOT DP-AWARE, CONFIDENCE INTERVALS. When fed
a DP-privatized MRResult (from the run_federated_private.py /
dp_mechanism.py path), the se here is the ORDINARY post-privatization
sampling standard error — it does NOT add the DP noise's own variance or
account for clipping bias. That means CIs computed on privatized results
UNDER-STATE the true uncertainty (empirically: coverage measured in
privacy_budget_sweep.py's output can sit noticeably below the nominal
95%, or in some regimes appear to float above it by coincidence — either
way, the number is not meaningful for the private pipeline as-is). Do
not describe or label these as "DP confidence intervals" anywhere until
a DP-aware version (adding DP-noise variance + clipping-bias correction
to the formula below) is actually implemented — that is still a TODO,
not done in this file.
"""

from dataclasses import dataclass
import numpy as np
from scipy import stats

from mr_estimators import MRResult


@dataclass
class CIResult:
    method: str
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    confidence_level: float


def confidence_interval(result: MRResult, confidence_level: float = 0.95) -> CIResult:
    z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    return CIResult(
        method=result.method,
        estimate=result.estimate,
        se=result.se,
        ci_low=result.estimate - z * result.se,
        ci_high=result.estimate + z * result.se,
        p_value=result.p_value,
        confidence_level=confidence_level,
    )


def summarize(results: list[MRResult], confidence_level: float = 0.95) -> list[CIResult]:
    return [confidence_interval(r, confidence_level) for r in results]


def print_summary(ci_results: list[CIResult], true_effect: float = None) -> None:
    header = f"{'method':15s} {'estimate':>9s} {'se':>8s} {'CI low':>9s} {'CI high':>9s} {'p':>10s}"
    print(header)
    print("-" * len(header))
    for r in ci_results:
        print(
            f"{r.method:15s} {r.estimate:9.4f} {r.se:8.4f} "
            f"{r.ci_low:9.4f} {r.ci_high:9.4f} {r.p_value:10.3g}"
        )
    if true_effect is not None:
        print(f"\n(true causal effect = {true_effect:.4f})")
        for r in ci_results:
            covers = r.ci_low <= true_effect <= r.ci_high
            print(f"  {r.method:15s} CI covers truth: {covers}")


if __name__ == "__main__":
    import config
    from data_simulator import simulate_all_sites
    from federated_aggregator import aggregate_sites_with_se
    from mr_estimators import ivw, mr_egger, weighted_median

    panel, exp_sites, out_sites = simulate_all_sites()
    ge = aggregate_sites_with_se(exp_sites)
    go = aggregate_sites_with_se(out_sites)

    results = [fn(ge.beta, ge.se, go.beta, go.se) for fn in (ivw, mr_egger, weighted_median)]
    ci_results = summarize(results)
    print_summary(ci_results, true_effect=config.TRUE_CAUSAL_EFFECT)
