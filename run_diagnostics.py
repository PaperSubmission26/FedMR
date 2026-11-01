"""
run_diagnostics.py

Runs the full Phase 4 validity-diagnostics suite and prints a report
using DISTINCT statuses (PASS, WARN, FAIL, NOT_APPLICABLE) — these are
NOT collapsed into one boolean, since a WARN (e.g. some weak instruments
present, but overall usable) and a FAIL (e.g. significant SNP-level
heterogeneity, i.e. likely pleiotropy) mean very different things for
whether to trust the result, and conflating them would hide that
distinction from whoever reads the report.

Every report is built ONCE here and the exact same objects are used both
to compute the overall status AND to print — nothing is recomputed for
display, so the printed output can never diverge from what the status
was actually based on. Checks that need at least 3 SNPs (MR-Egger,
weighted-median) are computed via their "safe" wrapper, which returns a
NOT_APPLICABLE report instead of letting mr_estimators.py's ValueError
crash this entire function.

IMPORTANT — "Overall (worst individual status)" IS NOT A FORMAL,
MULTIPLE-TESTING-ADJUSTED JOINT HYPOTHESIS TEST. It's a simple, easy-
to-read summary: the worst status among several FORMAL, INDEPENDENT
checks, each run at its own nominal alpha. Taking the max/worst across k
independent tests inflates the chance that AT LEAST ONE shows FAIL/WARN
well above any single test's own alpha (e.g. with 5 independent tests
each at alpha=0.05, the chance at least one is "significant" under a
true null is close to 1-(0.95)^5 ~ 23%, not 5%). An overall FAIL here
means "at least one individual check's own result was concerning", NOT
"there is a 5%-controlled, formal, family-wise finding of invalidity".
Treat it as a triage signal telling you WHICH per-check report to
actually go read — not as a single calibrated p-value in its own right.

"site_causal_effect_heterogeneity" is EXCLUDED from this overall
computation: it is explicitly EXPLORATORY (see heterogeneity_check.py's
is_approximate flag) because its Cochran's Q treats per-site causal
estimates as independent when they actually share the same
global_exposure and are therefore correlated — its own p-value is not
formally calibrated, so it must never be allowed to drive a claim that
"a formal test FAILED". Its status is still shown per-check, just not
folded into the formal worst-status summary.

Runs against the NON-PRIVATE federated pipeline by default. These
diagnostics are federated-safe individually, but running them
repeatedly against the SAME real (not synthetic) dataset would need its
own privacy accounting under Phase 3's composition requirement — not
addressed here, flagged as a gap for the eventual real-data phase.
"""

import config
from data_simulator import simulate_all_sites
from federated_aggregator import aggregate_sites_with_se
from instrument_strength_check import check_instrument_strength, print_report as print_strength_report
from heterogeneity_check import (
    check_snp_level_heterogeneity, check_site_level_heterogeneity,
    check_site_causal_effect_heterogeneity,
    print_snp_report, print_site_report, print_site_causal_report,
)
from sensitivity_analysis import (
    safe_egger_intercept_check, leave_one_out_ivw, ivw_excluding_weak_instruments,
    compare_estimators, check_i_squared_gx, print_sensitivity_reports,
)

_STATUS_SEVERITY = {"PASS": 0, "NOT_APPLICABLE": 0, "WARN": 1, "FAIL": 2}


def run_diagnostics():
    panel, exposure_sites, outcome_sites = simulate_all_sites()

    global_exposure = aggregate_sites_with_se(exposure_sites)
    global_outcome = aggregate_sites_with_se(outcome_sites)

    reports = {
        "instrument_strength": check_instrument_strength(global_exposure),
        # SNP-level Q now uses modified iterative weights internally (see
        # heterogeneity_check.py) — empirically brought the false-positive
        # rate from ~23% down to ~8-9% under a clean null, vs. the naive
        # first-order-weighted Q that used to be surfaced here.
        "snp_heterogeneity": check_snp_level_heterogeneity(global_exposure, global_outcome),
        "exposure_site_heterogeneity": check_site_level_heterogeneity(exposure_sites),
        "outcome_site_heterogeneity": check_site_level_heterogeneity(outcome_sites),
        # DIFFERENT question from the two site-heterogeneity checks above:
        # tests whether the DERIVED CAUSAL ESTIMATE (not the raw per-SNP
        # association) differs across outcome sites.
        "site_causal_effect_heterogeneity": check_site_causal_effect_heterogeneity(
            global_exposure, outcome_sites
        ),
        # safe_* wrapper: computes mr_egger itself only if n_snps>=3 and
        # the design matrix isn't singular; also downgrades a raw FAIL to
        # WARN when I^2_GX indicates the underlying regression is unreliable
        "egger_intercept": safe_egger_intercept_check(global_exposure, global_outcome),
        "cross_estimator": compare_estimators(global_exposure, global_outcome),
        "weak_snp_excluded_ivw": ivw_excluding_weak_instruments(global_exposure, global_outcome),
        "i_squared_gx": check_i_squared_gx(global_exposure),
    }
    reports["leave_one_out"] = (
        leave_one_out_ivw(global_exposure, global_outcome) if global_exposure.n_snps >= 3 else None
    )

    return global_exposure, global_outcome, reports


if __name__ == "__main__":
    print(f"=== Validity diagnostics ({config.N_SNPS} SNPs, "
          f"{config.N_EXPOSURE_SITES} exposure sites, {config.N_OUTCOME_SITES} outcome sites) ===\n")

    global_exposure, global_outcome, reports = run_diagnostics()

    print_strength_report(reports["instrument_strength"])
    print()
    print_snp_report(reports["snp_heterogeneity"])
    print()
    print_site_report(reports["exposure_site_heterogeneity"])
    print_site_report(reports["outcome_site_heterogeneity"])
    print_site_causal_report(reports["site_causal_effect_heterogeneity"])
    print()
    print_sensitivity_reports(
        reports["egger_intercept"], reports["i_squared_gx"], reports["leave_one_out"],
        reports["weak_snp_excluded_ivw"], reports["cross_estimator"],
    )
    if reports["egger_intercept"].low_i2gx_caveat:
        print("  (MR-Egger intercept: I^2_GX indicates the underlying regression may be "
              "unreliable — this does NOT change the intercept test's own PASS/FAIL status "
              "above, it's a separate reliability caveat; see I^2_GX for details.)")

    # site_causal_effect_heterogeneity is EXPLORATORY (is_approximate=True,
    # see heterogeneity_check.py) — its p-value is not formally calibrated
    # (per-site estimates share the same global_exposure and are not
    # independent, violating Cochran's Q's own assumption), so it is shown
    # per-check but EXCLUDED from the formal overall worst-status: an
    # overall "FAIL" should only ever reflect a properly calibrated test
    # actually rejecting at its own alpha, not an exploratory diagnostic
    # that says outright it isn't calibrated.
    _EXCLUDED_FROM_OVERALL = {"site_causal_effect_heterogeneity"}
    statuses = {name: (r.status if r is not None else "NOT_APPLICABLE") for name, r in reports.items()}
    formal_statuses = {name: s for name, s in statuses.items() if name not in _EXCLUDED_FROM_OVERALL}
    worst = max(formal_statuses.values(), key=lambda s: _STATUS_SEVERITY[s])

    print(f"\n=== Per-check status: {statuses} ===")
    print(f"    (site_causal_effect_heterogeneity is EXPLORATORY/approximate — shown above but "
          f"EXCLUDED from the overall status below; see its own [is_approximate] note.)")
    print(f"=== Overall (WORST individual status among the FORMAL checks — NOT a "
          f"multiple-testing-adjusted joint test; see module docstring): {worst} ===")
    if worst == "FAIL":
        print("At least one check formally FAILED at its OWN alpha — treat the causal estimate "
              "with real skepticism, not just caution. This does NOT mean a 5%-controlled "
              "family-wise error rate; go read the specific failing check(s) above.")
    elif worst == "WARN":
        print("No formal FAIL, but at least one check WARNed (an operational threshold, not a "
              "formal test) — worth a closer look, but not necessarily disqualifying.")
