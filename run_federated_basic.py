"""
run_federated_basic.py

Full non-private federated run, end to end:
  1. each site simulates its own individuals and derives local sufficient
     statistics (data_simulator.py) — raw individuals never leave the site
  2. the aggregator combines per-site sufficient stats into global
     per-SNP exposure/outcome associations (federated_aggregator.py)
  3. the MR estimator suite runs on those global associations
     (mr_estimators.py)
  4. confidence intervals are computed (compute_confidence_intervals.py)

This is the reference entry point for "just run the federated pipeline" —
no DP noise yet (that's Phase 4).
"""

import config
from data_simulator import simulate_all_sites
from federated_aggregator import aggregate_sites_with_se
from mr_estimators import ivw, mr_egger, weighted_median
from compute_confidence_intervals import summarize, print_summary


def run_federated_basic():
    panel, exposure_sites, outcome_sites = simulate_all_sites()

    print(f"Sites: {len(exposure_sites)} exposure, {len(outcome_sites)} outcome  |  "
          f"SNPs: {config.N_SNPS}  |  "
          f"weak instruments: {panel.is_weak_instrument.sum()}  |  "
          f"pleiotropic SNPs: {panel.is_pleiotropic.sum()}")

    global_exposure = aggregate_sites_with_se(exposure_sites)
    global_outcome = aggregate_sites_with_se(outcome_sites)

    results = [
        fn(global_exposure.beta, global_exposure.se, global_outcome.beta, global_outcome.se)
        for fn in (ivw, mr_egger, weighted_median)
    ]
    ci_results = summarize(results)

    return panel, global_exposure, global_outcome, ci_results


if __name__ == "__main__":
    panel, global_exposure, global_outcome, ci_results = run_federated_basic()
    print("\n=== Federated MR result (non-private) ===")
    print_summary(ci_results, true_effect=config.TRUE_CAUSAL_EFFECT)
