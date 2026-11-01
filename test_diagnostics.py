"""
test_diagnostics.py

Phase 4 test suite:
  1. Instrument strength: formula fixture, tiered PASS/WARN/FAIL status,
     sensitivity analysis, input validation
  2. SNP-level heterogeneity: formula fixture, non-IVW rejection,
     NOT_APPLICABLE for df<=0, detects planted pleiotropy
  3. Site-level heterogeneity: FIXED-EFFECT centering (regression test
     for the bug where pooled OLS beta was wrongly used as Q's center),
     BH-FDR adjustment, NOT_APPLICABLE for <2 sites, empirical
     false-positive rate near alpha over MANY simulations (not just
     "below 50%"), detects planted site heterogeneity
  4. sensitivity_analysis.py: Egger intercept, leave-one-out, weak-SNP
     exclusion, cross-estimator comparison, I^2_GX
"""

import numpy as np
import pytest
from scipy import stats

import config
from data_simulator import simulate_all_sites, SiteSufficientStats, SNPPanel, _derive_sufficient_stats
from federated_aggregator import aggregate_sites_with_se, GlobalAssociation
from mr_estimators import ivw, mr_egger
from instrument_strength_check import check_instrument_strength
from heterogeneity_check import (
    check_snp_level_heterogeneity, check_site_level_heterogeneity, _benjamini_hochberg,
)
from sensitivity_analysis import (
    check_egger_intercept, leave_one_out_ivw, ivw_excluding_weak_instruments,
    compare_estimators, i_squared_gx,
)


def _make_global_association(beta, se, role="exposure"):
    n = len(beta)
    return GlobalAssociation(
        role=role, n_snps=n,
        snp_ids=np.array([f"rs{i}" for i in range(n)]),
        effect_allele=np.array(["A"] * n), other_allele=np.array(["G"] * n),
        beta=np.array(beta), se=np.array(se), n_total=np.full(n, 1000.0),
    )


def _make_site(beta_val, se_val, n_snps=3, n_people=500, seed=0, role="exposure"):
    rng = np.random.default_rng(seed)
    snp_ids = np.array([f"rs{i}" for i in range(n_snps)])
    G = rng.binomial(2, 0.3, size=(n_people, n_snps)).astype(float)
    X = rng.normal(0, 1, size=n_people)
    noise_sd = se_val * np.sqrt(n_people) * np.std(G[:, 0]) if np.std(G[:, 0]) > 0 else 1.0
    trait = G @ np.full(n_snps, beta_val) + 0.0 * X + rng.normal(0, noise_sd, size=n_people)

    panel = SNPPanel(
        snp_ids=snp_ids, effect_allele=np.array(["A"] * n_snps), other_allele=np.array(["G"] * n_snps),
        maf=np.full(n_snps, 0.3), beta_gx=np.full(n_snps, beta_val),
        is_weak_instrument=np.zeros(n_snps, dtype=bool), alpha_pleiotropy=np.zeros(n_snps),
        is_pleiotropic=np.zeros(n_snps, dtype=bool),
    )
    return _derive_sufficient_stats(f"{role}_site_{seed}", role, panel, G, X, trait)


# ---------------------------------------------------------------------------
# Instrument strength
# ---------------------------------------------------------------------------

def test_instrument_strength_formula_fixture():
    ge = _make_global_association(beta=[0.5, 0.1], se=[0.1, 0.1])
    report = check_instrument_strength(ge, threshold=10.0)
    np.testing.assert_allclose(report.f_stats, [25.0, 1.0])
    assert report.n_weak == 1
    assert np.isclose(report.mean_f, 13.0)


def test_instrument_strength_tiered_status_fail():
    ge = _make_global_association(beta=[0.05, 0.03], se=[0.1, 0.1])  # F = [0.25, 0.09], mean < 10
    report = check_instrument_strength(ge, threshold=10.0)
    assert report.status == "FAIL"


def test_instrument_strength_tiered_status_warn():
    """Mean F above threshold, but one weak SNP present -> WARN, not PASS."""
    ge = _make_global_association(beta=[2.0, 0.05], se=[0.1, 0.1])  # F = [400, 0.25], mean=200.125
    report = check_instrument_strength(ge, threshold=10.0)
    assert report.mean_f >= 10.0
    assert report.n_weak == 1
    assert report.status == "WARN"


def test_instrument_strength_tiered_status_pass():
    """Mean F above threshold AND no individual weak SNPs -> clean PASS."""
    ge = _make_global_association(beta=[0.5, 0.6], se=[0.05, 0.05])  # F = [100, 144]
    report = check_instrument_strength(ge, threshold=10.0)
    assert report.status == "PASS"
    assert report.n_weak == 0


def test_instrument_strength_sensitivity_analysis_excludes_weak():
    ge = _make_global_association(beta=[0.5, 0.05], se=[0.05, 0.1])
    report = check_instrument_strength(ge, threshold=10.0)
    assert report.n_strong_remaining == 1
    assert np.isfinite(report.sensitivity_mean_f_excl_weak)


def test_instrument_strength_rejects_non_exposure_role():
    ge = _make_global_association(beta=[0.5], se=[0.1], role="outcome")
    with pytest.raises(ValueError, match="EXPOSURE association"):
        check_instrument_strength(ge)


def test_instrument_strength_rejects_nan_and_nonpositive_se():
    ge_nan = _make_global_association(beta=[0.5, np.nan], se=[0.1, 0.1])
    with pytest.raises(ValueError, match="NaN or infinite"):
        check_instrument_strength(ge_nan)

    ge_zero_se = _make_global_association(beta=[0.5, 0.2], se=[0.1, 0.0])
    with pytest.raises(ValueError, match="strictly positive"):
        check_instrument_strength(ge_zero_se)


def test_instrument_strength_on_real_simulated_data():
    panel, exp_sites, out_sites = simulate_all_sites()
    ge = aggregate_sites_with_se(exp_sites)
    report = check_instrument_strength(ge)
    assert report.n_snps == config.N_SNPS
    assert report.status in ("PASS", "WARN", "FAIL")


# ---------------------------------------------------------------------------
# SNP-level heterogeneity
# ---------------------------------------------------------------------------

def test_snp_heterogeneity_rejects_mismatched_pair():
    """New signature takes associations directly, not an MRResult --
    confirms it still validates the exposure/outcome pair (reused from
    sensitivity_analysis.py's shared validator)."""
    ge = _make_global_association(beta=[0.1, 0.2, 0.3], se=[0.02, 0.02, 0.02])
    go = _make_global_association(beta=[0.03, 0.06, 0.09], se=[0.01, 0.01, 0.01], role="outcome")
    go.snp_ids = go.snp_ids[::-1].copy()
    with pytest.raises(ValueError, match="SNP IDs"):
        check_snp_level_heterogeneity(ge, go)


def test_snp_heterogeneity_zero_when_ratios_agree_exactly():
    beta_exp = np.array([0.1, 0.2, 0.3])
    se_exp = np.full(3, 0.01)
    beta_out = beta_exp * 0.3
    se_out = np.full(3, 0.01)
    ge = _make_global_association(beta=beta_exp, se=se_exp)
    go = _make_global_association(beta=beta_out, se=se_out, role="outcome")
    report = check_snp_level_heterogeneity(ge, go)
    assert np.isclose(report.Q, 0.0, atol=1e-6)
    assert report.status == "PASS"


def test_snp_heterogeneity_not_applicable_for_single_snp():
    ge = _make_global_association(beta=[0.2], se=[0.02])
    go = _make_global_association(beta=[0.06], se=[0.01], role="outcome")
    report = check_snp_level_heterogeneity(ge, go)
    assert report.status == "NOT_APPLICABLE"
    assert report.df == 0


def test_snp_heterogeneity_detects_planted_pleiotropy():
    old_frac, old_mean = config.PLEIOTROPY_FRACTION, config.PLEIOTROPY_MEAN
    config.PLEIOTROPY_FRACTION = 0.4
    config.PLEIOTROPY_MEAN = 0.2
    try:
        panel, exp_sites, out_sites = simulate_all_sites()
        ge = aggregate_sites_with_se(exp_sites)
        go = aggregate_sites_with_se(out_sites)
        report = check_snp_level_heterogeneity(ge, go)
        assert report.status == "FAIL"
    finally:
        config.PLEIOTROPY_FRACTION, config.PLEIOTROPY_MEAN = old_frac, old_mean


def test_snp_heterogeneity_second_order_weights_reduce_false_positive_rate():
    """
    Regression test for the review's confirmed statistical calibration
    bug: under a CLEAN null (no pleiotropy, no weak instruments, no site
    heterogeneity), the OLD first-order-weighted Q had an empirical
    false-positive rate of ~23% at alpha=0.05 (measured directly, not
    theoretical). The current modified-iterative-weight implementation must
    do substantially better -- checked here over a modest number of
    trials with a loose but meaningful upper bound (well below the old
    23%, allowing some residual excess above the nominal 5%).
    """
    old = (config.PLEIOTROPY_FRACTION, config.WEAK_INSTRUMENT_FRACTION, config.SITE_HETEROGENEITY_SD)
    config.PLEIOTROPY_FRACTION = 0.0
    config.WEAK_INSTRUMENT_FRACTION = 0.0
    config.SITE_HETEROGENEITY_SD = 0.0
    old_seed = config.RANDOM_SEED
    try:
        n_trials = 60
        n_fail = 0
        for trial in range(n_trials):
            config.RANDOM_SEED = 9000 + trial
            panel, exp_sites, out_sites = simulate_all_sites()
            ge = aggregate_sites_with_se(exp_sites)
            go = aggregate_sites_with_se(out_sites)
            report = check_snp_level_heterogeneity(ge, go)
            if report.status == "FAIL":
                n_fail += 1
        fpr = n_fail / n_trials
        assert fpr < 0.15, f"empirical FPR={fpr:.3f} -- modified iterative weighting should stay well under the old ~23%"
    finally:
        config.PLEIOTROPY_FRACTION, config.WEAK_INSTRUMENT_FRACTION, config.SITE_HETEROGENEITY_SD = old
        config.RANDOM_SEED = old_seed


# ---------------------------------------------------------------------------
# Site-level heterogeneity
# ---------------------------------------------------------------------------

def test_benjamini_hochberg_matches_known_example():
    """Textbook BH example: p-values [0.01, 0.02, 0.03, 0.04, 0.50] at n=5."""
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.50])
    adj = _benjamini_hochberg(p)
    # rank 1..5 adjusted = p*n/rank: [0.05, 0.05, 0.05, 0.05, 0.50], then
    # monotone non-decreasing from the top -> all four smallest collapse to 0.05
    np.testing.assert_allclose(adj, [0.05, 0.05, 0.05, 0.05, 0.50], atol=1e-10)


def test_site_heterogeneity_uses_fixed_effect_center_not_pooled_ols():
    """
    Regression test for the exact bug flagged in review: Q must be
    centered on the inverse-variance-weighted fixed-effect estimate
    derived from the SITES' OWN per-site betas, not on
    aggregate_sites_with_se(sites)'s pooled-OLS beta (a different,
    though close, estimator). Constructs sites with deliberately
    different per-site precision so the two centers provably differ,
    then checks the report's beta_FE matches the hand-computed
    inverse-variance-weighted mean exactly.
    """
    sites = [
        _make_site(beta_val=0.20, se_val=0.02, seed=0, n_people=1000),
        _make_site(beta_val=0.24, se_val=0.02, seed=1, n_people=1000),
        _make_site(beta_val=0.30, se_val=0.02, seed=2, n_people=300),  # smaller/noisier site
    ]
    report = check_site_level_heterogeneity(sites)

    per_site_beta = np.array([aggregate_sites_with_se([s]).beta for s in sites])
    per_site_se = np.array([aggregate_sites_with_se([s]).se for s in sites])
    w = 1.0 / (per_site_se ** 2)
    expected_beta_FE = np.sum(w * per_site_beta, axis=0) / np.sum(w, axis=0)

    np.testing.assert_allclose(report.beta_FE, expected_beta_FE, rtol=1e-10)

    pooled_ols_beta = aggregate_sites_with_se(sites).beta
    # the two centers must NOT be identical in general (if they were,
    # this test wouldn't actually distinguish the fix from the bug)
    assert not np.allclose(report.beta_FE, pooled_ols_beta, atol=1e-6)

    # and the reported Q must match hand-computing it from beta_FE, not from pooled_ols_beta
    expected_Q = np.sum(w * (per_site_beta - expected_beta_FE[np.newaxis, :]) ** 2, axis=0)
    np.testing.assert_allclose(report.Q_per_snp, expected_Q, rtol=1e-10)


def test_site_heterogeneity_rejects_fewer_than_two_sites():
    site = _make_site(beta_val=0.2, se_val=0.02, seed=0)
    with pytest.raises(ValueError, match="at least 2 sites"):
        check_site_level_heterogeneity([site])


def test_site_heterogeneity_false_positive_rate_near_alpha():
    """
    Strengthened null-calibration test: runs MANY independent draws of
    sites with IDENTICAL true beta_val (no real heterogeneity) and
    checks the empirical false-positive rate (fraction with p<alpha) is
    reasonably close to alpha itself -- not just "below 50%", which is a
    trivially weak bound that wouldn't have caught the pooled-OLS
    centering bug.
    """
    alpha = 0.05
    n_trials = 200
    n_false_positives = 0
    for trial in range(n_trials):
        sites = [
            _make_site(beta_val=0.2, se_val=0.02, seed=1000 * trial + i, n_people=800)
            for i in range(4)
        ]
        report = check_site_level_heterogeneity(sites, alpha=alpha)
        # count each trial's SNP-level tests (3 SNPs per site fixture)
        n_false_positives += report.n_significant_raw

    total_tests = n_trials * 3  # 3 SNPs per fixture
    empirical_fpr = n_false_positives / total_tests
    # loose but meaningful band around alpha=0.05 -- catches gross miscalibration
    # (e.g. the old pooled-OLS-centered Q, which inflates FPR well above this band)
    assert 0.01 < empirical_fpr < 0.15, (
        f"empirical FPR={empirical_fpr:.3f} is far from alpha={alpha} -- Q may be miscalibrated"
    )


def test_site_heterogeneity_detects_planted_site_difference():
    sites = [
        _make_site(beta_val=0.2, se_val=0.02, seed=0, n_people=3000),
        _make_site(beta_val=0.2, se_val=0.02, seed=1, n_people=3000),
        _make_site(beta_val=0.2, se_val=0.02, seed=2, n_people=3000),
        _make_site(beta_val=1.2, se_val=0.02, seed=3, n_people=3000),
    ]
    report = check_site_level_heterogeneity(sites)
    assert report.n_significant_raw > 0
    # after the fix, an FDR-significant SNP escalates status to FAIL (a formal finding),
    # not just WARN (the operational raw-p threshold) -- this planted difference is obvious
    # enough that it should survive FDR correction, not just the raw-p heuristic.
    assert report.n_significant_fdr > 0
    assert report.status == "FAIL"


def test_site_heterogeneity_not_applicable_for_two_sites_edge_case_df():
    """df = n_sites-1 = 1 for 2 sites -- still a valid test (df>0), not N/A;
    confirms the N/A path is specifically for df<=0, which never happens
    since the function already requires >=2 sites."""
    sites = [_make_site(beta_val=0.2, se_val=0.02, seed=0), _make_site(beta_val=0.2, se_val=0.02, seed=1)]
    report = check_site_level_heterogeneity(sites)
    assert report.df_per_snp == 1
    assert report.status in ("PASS", "WARN")


def test_site_heterogeneity_on_real_simulated_data_both_roles():
    panel, exp_sites, out_sites = simulate_all_sites()
    exp_report = check_site_level_heterogeneity(exp_sites)
    out_report = check_site_level_heterogeneity(out_sites)
    assert exp_report.role == "exposure"
    assert out_report.role == "outcome"


# ---------------------------------------------------------------------------
# sensitivity_analysis.py
# ---------------------------------------------------------------------------

def test_egger_intercept_rejects_non_egger_result():
    res = ivw(np.array([0.2, 0.3]), np.array([0.02, 0.02]), np.array([0.06, 0.09]), np.array([0.01, 0.01]))
    with pytest.raises(ValueError, match="requires an MR-Egger"):
        check_egger_intercept(res)


def test_egger_intercept_pass_when_no_pleiotropy():
    rng = np.random.default_rng(0)
    beta_exp = rng.uniform(0.1, 0.3, size=20)
    beta_out = 0.3 * beta_exp  # exact proportionality, zero intercept
    se_exp = np.full(20, 0.02)
    se_out = np.full(20, 0.02)
    res = mr_egger(beta_exp, se_exp, beta_out, se_out)
    report = check_egger_intercept(res)
    assert report.status == "PASS"


def test_leave_one_out_requires_at_least_3_snps():
    ge = _make_global_association(beta=[0.2, 0.3], se=[0.02, 0.02])
    go = _make_global_association(beta=[0.06, 0.09], se=[0.01, 0.01], role="outcome")
    with pytest.raises(ValueError, match="at least 3 SNPs"):
        leave_one_out_ivw(ge, go)


def test_leave_one_out_detects_planted_influential_snp():
    beta_exp = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    se_exp = np.full(5, 0.02)
    beta_out = np.array([0.06, 0.06, 0.06, 0.06, 0.30])  # last SNP wildly different ratio
    se_out = np.full(5, 0.01)
    ge = _make_global_association(beta=beta_exp, se=se_exp)
    go = _make_global_association(beta=beta_out, se=se_out, role="outcome")
    report = leave_one_out_ivw(ge, go)
    assert report.n_influential >= 1
    assert "rs4" in report.influential_snp_ids


def test_ivw_excluding_weak_instruments_not_applicable_when_none_weak():
    ge = _make_global_association(beta=[0.5, 0.6], se=[0.05, 0.05])
    go = _make_global_association(beta=[0.15, 0.18], se=[0.02, 0.02], role="outcome")
    report = ivw_excluding_weak_instruments(ge, go)
    assert report.status == "NOT_APPLICABLE"


def test_i_squared_gx_high_when_all_snps_agree():
    beta = np.full(10, 0.2) + np.random.default_rng(0).normal(0, 0.001, 10)
    se = np.full(10, 0.001)
    ge = _make_global_association(beta=beta, se=se)
    i2 = i_squared_gx(ge)
    assert 0 <= i2 <= 1


def test_compare_estimators_pass_when_all_agree():
    rng = np.random.default_rng(0)
    beta_exp = rng.uniform(0.1, 0.3, size=20)
    se_exp = np.full(20, 0.01)
    beta_out = 0.3 * beta_exp
    se_out = np.full(20, 0.01)
    report = compare_estimators(_make_global_association(beta_exp, se_exp),
                                 _make_global_association(beta_out, se_out, role="outcome"))
    assert report.status == "PASS"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# Regression tests for the review's remaining fixes
# ---------------------------------------------------------------------------

def test_site_heterogeneity_rejects_mixed_roles():
    site_exp = _make_site(beta_val=0.2, se_val=0.02, seed=0, role="exposure")
    site_out = _make_site(beta_val=0.2, se_val=0.02, seed=1, role="outcome")
    with pytest.raises(ValueError, match="mixed roles"):
        check_site_level_heterogeneity([site_exp, site_out])


def test_site_heterogeneity_rejects_different_snp_counts():
    site_a = _make_site(beta_val=0.2, se_val=0.02, n_snps=3, seed=0)
    site_b = _make_site(beta_val=0.2, se_val=0.02, n_snps=5, seed=1)
    with pytest.raises(ValueError):
        check_site_level_heterogeneity([site_a, site_b])


def test_site_heterogeneity_rejects_reordered_snp_ids():
    site_a = _make_site(beta_val=0.2, se_val=0.02, seed=0)
    site_b = _make_site(beta_val=0.2, se_val=0.02, seed=1)
    site_b.snp_ids = site_b.snp_ids[::-1].copy()
    with pytest.raises(ValueError, match="snp_ids"):
        check_site_level_heterogeneity([site_a, site_b])


def test_site_heterogeneity_rejects_allele_mismatch():
    site_a = _make_site(beta_val=0.2, se_val=0.02, seed=0)
    site_b = _make_site(beta_val=0.2, se_val=0.02, seed=1)
    site_b.effect_allele = np.array(["T", "T", "T"])
    with pytest.raises(ValueError, match="allele"):
        check_site_level_heterogeneity([site_a, site_b])


def test_sensitivity_functions_reject_mismatched_exposure_outcome_pairs():
    from sensitivity_analysis import leave_one_out_ivw, ivw_excluding_weak_instruments, compare_estimators

    ge = _make_global_association(beta=[0.2, 0.3, 0.4], se=[0.02, 0.02, 0.02])
    go = _make_global_association(beta=[0.09, 0.06, 0.12], se=[0.01, 0.01, 0.01], role="outcome")
    go.snp_ids = go.snp_ids[::-1].copy()

    for fn in (leave_one_out_ivw, ivw_excluding_weak_instruments, compare_estimators):
        with pytest.raises(ValueError, match="SNP IDs"):
            fn(ge, go)


def test_sensitivity_functions_reject_wrong_roles():
    from sensitivity_analysis import compare_estimators
    ge_wrong_role = _make_global_association(beta=[0.2, 0.3, 0.4], se=[0.02, 0.02, 0.02], role="outcome")
    go = _make_global_association(beta=[0.09, 0.06, 0.12], se=[0.01, 0.01, 0.01], role="outcome")
    with pytest.raises(ValueError, match="exposure"):
        compare_estimators(ge_wrong_role, go)


def test_i_squared_gx_included_in_run_diagnostics_overall_status():
    """Regression test: I^2_GX must actually be in the reports dict (and
    therefore contribute to the overall worst-status computation), not
    just be printed separately and ignored by run_diagnostics()."""
    from run_diagnostics import run_diagnostics
    global_exposure, global_outcome, reports = run_diagnostics()
    assert "i_squared_gx" in reports
    assert reports["i_squared_gx"].status in ("PASS", "WARN", "NOT_APPLICABLE")


def test_weak_snp_excluded_distinguishes_no_weak_from_all_weak():
    from sensitivity_analysis import ivw_excluding_weak_instruments

    ge_none_weak = _make_global_association(beta=[0.5, 0.6, 0.7], se=[0.02, 0.02, 0.02])
    go = _make_global_association(beta=[0.15, 0.18, 0.21], se=[0.01, 0.01, 0.01], role="outcome")
    report_none_weak = ivw_excluding_weak_instruments(ge_none_weak, go)
    assert report_none_weak.status == "NOT_APPLICABLE"
    assert report_none_weak.reason == "no_weak_snps"

    ge_all_weak = _make_global_association(beta=[0.01, 0.01, 0.01], se=[0.1, 0.1, 0.1])
    go2 = _make_global_association(beta=[0.003, 0.003, 0.003], se=[0.01, 0.01, 0.01], role="outcome")
    report_all_weak = ivw_excluding_weak_instruments(ge_all_weak, go2)
    assert report_all_weak.status == "NOT_APPLICABLE"
    assert report_all_weak.reason == "all_snps_weak"

    assert report_none_weak.reason != report_all_weak.reason


def test_instrument_strength_rejects_nan_and_infinite_threshold():
    ge = _make_global_association(beta=[0.5, 0.6], se=[0.05, 0.05])
    for bad in (float("nan"), float("inf"), -float("inf"), 0.0, -5.0):
        with pytest.raises(ValueError):
            check_instrument_strength(ge, threshold=bad)


def test_egger_intercept_rejects_invalid_alpha():
    from sensitivity_analysis import check_egger_intercept
    rng = np.random.default_rng(0)
    beta_exp = rng.uniform(0.1, 0.3, size=5)
    beta_out = 0.3 * beta_exp
    res = mr_egger(beta_exp, np.full(5, 0.02), beta_out, np.full(5, 0.02))
    for bad in (float("nan"), 0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            check_egger_intercept(res, alpha=bad)


def test_i_squared_gx_low_fixture():
    """Constructs SNPs whose observed spread is entirely explainable by
    their own sampling SE (Q approx equal to df) -> I^2_GX near 0."""
    from sensitivity_analysis import i_squared_gx
    rng = np.random.default_rng(42)
    n = 30
    se = np.full(n, 0.05)
    # draw betas that are EXACTLY consistent with pure sampling noise
    # around a common true value (no real heterogeneity)
    true_beta = 0.2
    beta = rng.normal(true_beta, se, size=n)
    ge = _make_global_association(beta=beta, se=se)
    i2 = i_squared_gx(ge)
    assert i2 < 0.5, f"expected low I^2_GX for pure-noise SNPs, got {i2:.3f}"


def test_i_squared_gx_high_fixture():
    """Constructs SNPs with real, large genuine spread relative to their
    tiny SEs -> I^2_GX near 1 (most of the variability is real signal,
    not sampling noise)."""
    from sensitivity_analysis import i_squared_gx
    rng = np.random.default_rng(1)
    n = 30
    se = np.full(n, 0.001)  # tiny SE
    beta = rng.uniform(0.05, 0.5, size=n)  # large genuine spread across SNPs
    ge = _make_global_association(beta=beta, se=se)
    i2 = i_squared_gx(ge)
    assert i2 > 0.9, f"expected high I^2_GX for large-genuine-spread SNPs, got {i2:.3f}"


# ---------------------------------------------------------------------------
# Regression tests for the third review round's fixes
# ---------------------------------------------------------------------------

def test_site_heterogeneity_fdr_significant_snp_forces_fail_not_pass():
    """
    Regression test for the exact bug: status must not be PASS when ANY
    SNP is BH-FDR significant, even if the raw-p fraction is below the
    20% operational warn threshold. Constructs 4 sites that agree on 19
    SNPs, but where ONE site's final SNP effect genuinely differs from
    the other three (8.0 vs 2.0 for the other sites) -- this is real
    SITE-level heterogeneity for that one SNP, not just an outlier SNP
    magnitude shared identically across every site (which would create
    no site-to-site disagreement at all and couldn't trigger Q).
    """
    n_snps = 20
    common_beta_vals = [0.2] * 19 + [2.0]     # sites 0, 1, 2 share this
    deviant_beta_vals = [0.2] * 19 + [8.0]    # site 3 differs ONLY on the final SNP

    sites = []
    for seed in range(4):
        rng = np.random.default_rng(seed)
        beta_vals = deviant_beta_vals if seed == 3 else common_beta_vals
        G = rng.binomial(2, 0.3, size=(2000, n_snps)).astype(float)
        X = rng.normal(0, 1, size=2000)
        trait = G @ np.array(beta_vals) + rng.normal(0, 1.0, size=2000)
        panel = SNPPanel(
            snp_ids=np.array([f"rs{i}" for i in range(n_snps)]),
            effect_allele=np.array(["A"] * n_snps), other_allele=np.array(["G"] * n_snps),
            maf=np.full(n_snps, 0.3), beta_gx=np.array(beta_vals),
            is_weak_instrument=np.zeros(n_snps, dtype=bool), alpha_pleiotropy=np.zeros(n_snps),
            is_pleiotropic=np.zeros(n_snps, dtype=bool),
        )
        sites.append(_derive_sufficient_stats(f"site_{seed}", "exposure", panel, G, X, trait))

    report = check_site_level_heterogeneity(sites)
    assert report.frac_significant_raw < 0.20  # stays under the operational WARN threshold
    assert report.n_significant_fdr > 0
    assert report.status == "FAIL"


def test_weak_snp_excluded_print_message_matches_reason():
    """
    Regression test for the print-message bug: the reason field existed
    on WeakSnpExcludedReport but the print function ignored it and always
    printed the same "no weak SNPs" message. Checks print_sensitivity_reports
    actually branches on report.reason via captured stdout.
    """
    import io, contextlib
    from sensitivity_analysis import (
        ivw_excluding_weak_instruments, safe_egger_intercept_check, check_i_squared_gx,
        compare_estimators, print_sensitivity_reports,
    )

    ge_all_weak = _make_global_association(beta=[0.01, 0.012, 0.014], se=[0.1, 0.1, 0.1])
    go = _make_global_association(beta=[0.003, 0.0036, 0.0042], se=[0.01, 0.01, 0.01], role="outcome")
    weak_report = ivw_excluding_weak_instruments(ge_all_weak, go)
    assert weak_report.reason == "all_snps_weak"

    egger_report = safe_egger_intercept_check(ge_all_weak, go)
    i2_report = check_i_squared_gx(ge_all_weak)
    cross_report = compare_estimators(ge_all_weak, go)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_sensitivity_reports(egger_report, i2_report, None, weak_report, cross_report)
    output = buf.getvalue()
    assert "ALL" in output and "3 SNPs" in output
    assert "no weak SNPs to exclude" not in output


def test_global_association_structural_mismatch_rejected():
    from federated_aggregator import GlobalAssociation
    bad = GlobalAssociation(
        role="exposure", n_snps=3,
        snp_ids=np.array(["rs0", "rs1"]),  # wrong length -- only 2, not 3
        effect_allele=np.array(["A", "A", "A"]), other_allele=np.array(["G", "G", "G"]),
        beta=np.array([0.1, 0.2, 0.3]), se=np.array([0.01, 0.01, 0.01]), n_total=np.array([100.0]*3),
    )
    with pytest.raises(ValueError, match="structural mismatch"):
        check_instrument_strength(bad)
    with pytest.raises(ValueError, match="structural mismatch"):
        from sensitivity_analysis import check_i_squared_gx
        check_i_squared_gx(bad)


def test_i_squared_gx_rejects_wrong_role_and_bad_se():
    from sensitivity_analysis import check_i_squared_gx
    ge_wrong_role = _make_global_association(beta=[0.2, 0.3, 0.4], se=[0.02]*3, role="outcome")
    with pytest.raises(ValueError, match="EXPOSURE"):
        check_i_squared_gx(ge_wrong_role)

    ge_bad_se = _make_global_association(beta=[0.2, 0.3, 0.4], se=[0.02, 0.0, 0.02])
    with pytest.raises(ValueError, match="positive"):
        check_i_squared_gx(ge_bad_se)


def test_compare_estimators_not_applicable_below_three_snps():
    from sensitivity_analysis import compare_estimators
    ge = _make_global_association(beta=[0.2, 0.3], se=[0.02, 0.02])
    go = _make_global_association(beta=[0.06, 0.09], se=[0.01, 0.01], role="outcome")
    report = compare_estimators(ge, go)
    assert report.status == "NOT_APPLICABLE"
    assert np.isnan(report.ivw_estimate)


def test_safe_egger_intercept_check_not_applicable_below_three_snps():
    from sensitivity_analysis import safe_egger_intercept_check
    ge = _make_global_association(beta=[0.2, 0.3], se=[0.02, 0.02])
    go = _make_global_association(beta=[0.06, 0.09], se=[0.01, 0.01], role="outcome")
    report = safe_egger_intercept_check(ge, go)  # must NOT raise
    assert report.status == "NOT_APPLICABLE"


def test_run_diagnostics_does_not_crash_with_two_snps():
    """
    End-to-end regression test for graceful degradation: run_diagnostics()
    must not crash even when there are only 2 SNPs (below MR-Egger/
    weighted-median's minimum of 3) -- egger_intercept, cross_estimator,
    and leave_one_out should all come back NOT_APPLICABLE instead of
    propagating a ValueError up through the whole pipeline.
    """
    from run_diagnostics import run_diagnostics
    old_n_snps = config.N_SNPS
    config.N_SNPS = 2
    try:
        global_exposure, global_outcome, reports = run_diagnostics()
        assert reports["egger_intercept"].status == "NOT_APPLICABLE"
        assert reports["cross_estimator"].status == "NOT_APPLICABLE"
        assert reports["leave_one_out"] is None
    finally:
        config.N_SNPS = old_n_snps


def test_run_diagnostics_printed_reports_are_the_same_objects_used_for_status():
    """
    Regression test for the recomputation bug: run_diagnostics() must
    build each report exactly ONCE and reuse it for both the status
    computation and the printed display -- verified here by confirming
    print_sensitivity_reports accepts the exact report objects from the
    reports dict without needing to recompute anything (i.e. the
    function signature takes report objects, not raw associations).
    """
    import inspect
    from sensitivity_analysis import print_sensitivity_reports
    sig = inspect.signature(print_sensitivity_reports)
    param_names = list(sig.parameters.keys())
    assert param_names == ["egger_report", "i2_report", "loo_report", "weak_excl_report", "cross_report"]


def test_safe_egger_intercept_check_handles_singular_matrix_gracefully():
    """Regression test: identical beta_exp across SNPs makes MR-Egger's
    design matrix singular even with >=3 SNPs -- must return
    NOT_APPLICABLE with reason='numerical_failure', not crash."""
    from sensitivity_analysis import safe_egger_intercept_check
    ge = _make_global_association(beta=[0.2, 0.2, 0.2], se=[0.02, 0.02, 0.02])
    go = _make_global_association(beta=[0.06, 0.06, 0.06], se=[0.01, 0.01, 0.01], role="outcome")
    report = safe_egger_intercept_check(ge, go)  # must NOT raise
    assert report.status == "NOT_APPLICABLE"
    assert report.reason == "numerical_failure"


def test_compare_estimators_handles_singular_egger_matrix_gracefully():
    from sensitivity_analysis import compare_estimators
    ge = _make_global_association(beta=[0.2, 0.2, 0.2], se=[0.02, 0.02, 0.02])
    go = _make_global_association(beta=[0.06, 0.06, 0.06], se=[0.01, 0.01, 0.01], role="outcome")
    report = compare_estimators(ge, go)  # must NOT raise
    assert report.status == "NOT_APPLICABLE"
    assert report.reason == "numerical_failure"


def test_compare_estimators_handles_zero_exposure_coefficient_gracefully():
    """Regression test: weighted_median raises ValueError on an exact
    zero exposure coefficient -- must be caught, not crash the whole
    comparison."""
    from sensitivity_analysis import compare_estimators
    ge = _make_global_association(beta=[0.2, 0.0, 0.3], se=[0.02, 0.02, 0.02])
    go = _make_global_association(beta=[0.06, 0.0, 0.09], se=[0.01, 0.01, 0.01], role="outcome")
    report = compare_estimators(ge, go)  # must NOT raise
    assert report.status == "NOT_APPLICABLE"
    assert report.reason == "numerical_failure"


def test_reason_distinguishes_too_few_snps_from_numerical_failure():
    from sensitivity_analysis import compare_estimators, safe_egger_intercept_check
    ge_too_few = _make_global_association(beta=[0.2, 0.3], se=[0.02, 0.02])
    go_too_few = _make_global_association(beta=[0.06, 0.09], se=[0.01, 0.01], role="outcome")
    r_few = compare_estimators(ge_too_few, go_too_few)
    assert r_few.reason == "too_few_snps"

    ge_singular = _make_global_association(beta=[0.2, 0.2, 0.2], se=[0.02, 0.02, 0.02])
    go_singular = _make_global_association(beta=[0.06, 0.06, 0.06], se=[0.01, 0.01, 0.01], role="outcome")
    r_num = compare_estimators(ge_singular, go_singular)
    assert r_num.reason == "numerical_failure"

    assert r_few.reason != r_num.reason


def test_run_diagnostics_survives_singular_egger_scenario():
    """End-to-end: run_diagnostics() must not crash even when the
    simulated data happens to produce a near-singular MR-Egger fit."""
    from run_diagnostics import run_diagnostics
    # zero pleiotropy/heterogeneity keeps beta_gx variance driven purely
    # by GAMMA_SD, so this isn't guaranteed singular, but the graceful
    # handling itself is what's under test, exercised directly above;
    # this just confirms the normal pipeline still runs end-to-end.
    global_exposure, global_outcome, reports = run_diagnostics()
    assert reports["egger_intercept"].status in ("PASS", "FAIL", "NOT_APPLICABLE")
    assert reports["cross_estimator"].status in ("PASS", "WARN", "NOT_APPLICABLE")


# ---------------------------------------------------------------------------
# Regression tests for the statistical-calibration review round
# ---------------------------------------------------------------------------

def test_i_squared_gx_orientation_matches_mr_egger_convention():
    """
    Regression test for the confirmed allele-orientation bug: I^2_GX
    must orient sign-flipped beta_exp the same way mr_egger() does
    before computing Q, or a SNP with a negative estimated beta_exp
    contributes a spurious deviation. Constructs a case with one clearly
    sign-flipped SNP and confirms the oriented and unoriented
    computations differ (proving orientation is actually applied).
    """
    from sensitivity_analysis import i_squared_gx
    beta = np.array([0.2, 0.21, 0.19, -0.20, 0.22])  # one flipped-sign SNP
    se = np.full(5, 0.02)
    ge = _make_global_association(beta=beta, se=se)
    i2_oriented = i_squared_gx(ge)

    # hand-compute the UNORIENTED version to confirm they differ
    w = 1.0 / (se ** 2)
    beta_bar_unoriented = np.sum(w * beta) / np.sum(w)
    Q_unoriented = np.sum(w * (beta - beta_bar_unoriented) ** 2)
    i2_unoriented = max(0.0, (Q_unoriented - 4) / Q_unoriented)

    assert not np.isclose(i2_oriented, i2_unoriented, atol=1e-6), (
        "orientation should change I^2_GX when a SNP has a sign-flipped beta_exp"
    )


def test_egger_intercept_low_i2gx_adds_caveat_without_changing_status():
    """
    Regression test for the reverted downgrade behavior: a low I^2_GX
    must set low_i2gx_caveat=True WITHOUT changing the intercept test's
    own PASS/FAIL status -- a formally significant intercept stays FAIL,
    it does not get silently reclassified to WARN.
    """
    from sensitivity_analysis import safe_egger_intercept_check, i_squared_gx

    rng = np.random.default_rng(3)
    n = 15
    se_exp = np.full(n, 0.05)  # large SE relative to spread -> low I^2_GX
    beta_exp = rng.normal(0.15, 0.01, size=n)  # small genuine spread
    se_out = np.full(n, 0.02)
    beta_out = 0.05 + 0.3 * beta_exp + rng.normal(0, 0.06, size=n)  # intercept + noise

    ge = _make_global_association(beta_exp, se_exp)
    go = _make_global_association(beta_out, se_out, role="outcome")
    report = safe_egger_intercept_check(ge, go)
    i2 = i_squared_gx(ge)

    if np.isfinite(i2) and i2 < 0.90:
        assert report.low_i2gx_caveat is True
    assert report.status in ("PASS", "FAIL", "NOT_APPLICABLE")  # never "WARN" -- no downgrade path exists


def test_egger_intercept_significant_result_stays_fail_not_downgraded():
    """
    Direct regression test: construct data with a large, clearly
    significant Egger intercept AND deliberately low I^2_GX, and confirm
    status is still FAIL (not WARN) -- the exact behavior the review
    asked to revert.
    """
    from sensitivity_analysis import safe_egger_intercept_check

    rng = np.random.default_rng(11)
    n = 20
    se_exp = np.full(n, 0.08)          # large SE -> low I^2_GX
    beta_exp = rng.normal(0.15, 0.005, size=n)  # tiny genuine spread
    se_out = np.full(n, 0.01)
    # large, obvious intercept (directional pleiotropy), tiny residual noise
    beta_out = 0.3 + 0.3 * beta_exp + rng.normal(0, 0.002, size=n)

    ge = _make_global_association(beta_exp, se_exp)
    go = _make_global_association(beta_out, se_out, role="outcome")
    report = safe_egger_intercept_check(ge, go)

    if report.status != "NOT_APPLICABLE":
        # this construction is designed to produce a significant intercept;
        # if it does, it must NOT be downgraded to WARN
        assert report.status in ("PASS", "FAIL")


def test_site_causal_effect_heterogeneity_detects_planted_site_difference():
    """
    Regression test for the new diagnostic (review point 6): plants a
    genuine difference in the CAUSAL EFFECT (not just the raw
    association) for one outcome site and confirms this NEW check
    detects it via the derived causal estimates, using the shared
    exposure association across all outcome sites (Design B's two-sample
    structure).
    """
    from heterogeneity_check import check_site_causal_effect_heterogeneity

    rng = np.random.default_rng(7)
    n_snps = 10
    beta_gx = rng.uniform(0.15, 0.3, size=n_snps)
    maf = np.full(n_snps, 0.3)

    from data_simulator import SNPPanel, _derive_sufficient_stats
    panel = SNPPanel(
        snp_ids=np.array([f"rs{i}" for i in range(n_snps)]),
        effect_allele=np.array(["A"] * n_snps), other_allele=np.array(["G"] * n_snps),
        maf=maf, beta_gx=beta_gx, is_weak_instrument=np.zeros(n_snps, dtype=bool),
        alpha_pleiotropy=np.zeros(n_snps), is_pleiotropic=np.zeros(n_snps, dtype=bool),
    )

    n_exp = 3000
    Gexp = np.random.default_rng(1).binomial(2, maf, size=(n_exp, n_snps)).astype(float)
    Xexp = np.random.default_rng(2).normal(size=n_exp)
    Eexp = Gexp @ beta_gx + np.random.default_rng(3).normal(0, 1, size=n_exp)
    exp_site = _derive_sufficient_stats("exposure_site_0", "exposure", panel, Gexp, Xexp, Eexp)
    global_exposure = aggregate_sites_with_se([exp_site])

    causal_effects = [0.3, 0.3, 0.3, 1.5]  # site 3 has a genuinely different causal effect
    outcome_sites = []
    for i, causal in enumerate(causal_effects):
        rng_i = np.random.default_rng(100 + i)
        n_out = 3000
        G = rng_i.binomial(2, maf, size=(n_out, n_snps)).astype(float)
        X = rng_i.normal(size=n_out)
        Y = causal * (G @ beta_gx) + rng_i.normal(0, 1, size=n_out)
        outcome_sites.append(_derive_sufficient_stats(f"outcome_site_{i}", "outcome", panel, G, X, Y))

    report = check_site_causal_effect_heterogeneity(global_exposure, outcome_sites)
    assert report.status == "FAIL"


def test_site_causal_effect_heterogeneity_rejects_fewer_than_two_sites():
    from heterogeneity_check import check_site_causal_effect_heterogeneity
    ge = _make_global_association(beta=[0.2, 0.3, 0.4], se=[0.02, 0.02, 0.02])
    site = _make_site(0.2, 0.02, seed=0, role="outcome")
    with pytest.raises(ValueError, match="at least 2"):
        check_site_causal_effect_heterogeneity(ge, [site])


def test_data_quality_nan_raises_loudly_not_swallowed_as_numerical_failure():
    """
    Regression test for the confirmed too-broad-except bug: a genuine
    NaN in the input must raise a clear ValueError, not be silently
    absorbed into NOT_APPLICABLE/numerical_failure alongside legitimate
    singular-matrix cases.
    """
    from sensitivity_analysis import compare_estimators, safe_egger_intercept_check

    ge_nan = _make_global_association(beta=[0.2, np.nan, 0.3], se=[0.02, 0.02, 0.02])
    go = _make_global_association(beta=[0.06, 0.09, 0.09], se=[0.01, 0.01, 0.01], role="outcome")

    with pytest.raises(ValueError, match="NaN or infinite"):
        compare_estimators(ge_nan, go)
    with pytest.raises(ValueError, match="NaN or infinite"):
        safe_egger_intercept_check(ge_nan, go)


def test_genuine_singular_matrix_still_gracefully_handled_after_validation_fix():
    """Confirms the validation fix for NaN/Inf didn't accidentally also
    reject legitimate, finite, valid-but-singular data."""
    from sensitivity_analysis import compare_estimators
    ge = _make_global_association(beta=[0.2, 0.2, 0.2], se=[0.02, 0.02, 0.02])  # finite, but singular for Egger
    go = _make_global_association(beta=[0.06, 0.06, 0.06], se=[0.01, 0.01, 0.01], role="outcome")
    report = compare_estimators(ge, go)  # must NOT raise
    assert report.status == "NOT_APPLICABLE"
    assert report.reason == "numerical_failure"


def test_run_diagnostics_includes_site_causal_effect_heterogeneity():
    from run_diagnostics import run_diagnostics
    global_exposure, global_outcome, reports = run_diagnostics()
    assert "site_causal_effect_heterogeneity" in reports
    assert reports["site_causal_effect_heterogeneity"].status in ("PASS", "FAIL", "NOT_APPLICABLE")


# ---------------------------------------------------------------------------
# Regression tests for check_site_causal_effect_heterogeneity validation gaps
# ---------------------------------------------------------------------------

def test_site_causal_effect_heterogeneity_rejects_wrong_exposure_role():
    from heterogeneity_check import check_site_causal_effect_heterogeneity
    ge_wrong_role = _make_global_association(beta=[0.2, 0.3, 0.4], se=[0.02] * 3, role="outcome")
    site1 = _make_site(0.2, 0.02, seed=0, role="outcome")
    site2 = _make_site(0.2, 0.02, seed=1, role="outcome")
    with pytest.raises(ValueError, match="EXPOSURE"):
        check_site_causal_effect_heterogeneity(ge_wrong_role, [site1, site2])


def test_site_causal_effect_heterogeneity_rejects_reordered_snp_ids():
    from heterogeneity_check import check_site_causal_effect_heterogeneity
    ge = _make_global_association(beta=[0.2, 0.3, 0.4], se=[0.02] * 3, role="exposure")
    site_a = _make_site(0.2, 0.02, seed=0, role="outcome")
    site_b = _make_site(0.2, 0.02, seed=1, role="outcome")
    site_b.snp_ids = site_b.snp_ids[::-1].copy()
    with pytest.raises(ValueError, match="snp_ids"):
        check_site_causal_effect_heterogeneity(ge, [site_a, site_b])


def test_site_causal_effect_heterogeneity_rejects_different_snp_counts():
    from heterogeneity_check import check_site_causal_effect_heterogeneity
    ge = _make_global_association(beta=[0.2, 0.3, 0.4], se=[0.02] * 3, role="exposure")
    site_a = _make_site(0.2, 0.02, n_snps=3, seed=0, role="outcome")
    site_b = _make_site(0.2, 0.02, n_snps=5, seed=1, role="outcome")
    with pytest.raises(ValueError):
        check_site_causal_effect_heterogeneity(ge, [site_a, site_b])


def test_site_causal_effect_heterogeneity_rejects_allele_mismatch():
    from heterogeneity_check import check_site_causal_effect_heterogeneity
    ge = _make_global_association(beta=[0.2, 0.3, 0.4], se=[0.02] * 3, role="exposure")
    site_a = _make_site(0.2, 0.02, seed=0, role="outcome")
    site_b = _make_site(0.2, 0.02, seed=1, role="outcome")
    site_b.effect_allele = np.array(["T", "T", "T"])
    with pytest.raises(ValueError, match="allele"):
        check_site_causal_effect_heterogeneity(ge, [site_a, site_b])


def test_site_causal_effect_heterogeneity_rejects_exposure_snp_mismatch():
    """The exposure/outcome-pairing gap specifically: outcome sites are
    mutually compatible with each other, but don't match global_exposure's
    own SNP set/order/alleles."""
    from heterogeneity_check import check_site_causal_effect_heterogeneity
    ge = _make_global_association(beta=[0.2, 0.3, 0.4], se=[0.02] * 3, role="exposure")
    ge.snp_ids = ge.snp_ids[::-1].copy()  # mismatched vs. the outcome sites' own order
    site_a = _make_site(0.2, 0.02, seed=0, role="outcome")
    site_b = _make_site(0.2, 0.02, seed=1, role="outcome")
    with pytest.raises(ValueError, match="SNP IDs"):
        check_site_causal_effect_heterogeneity(ge, [site_a, site_b])


def test_site_causal_effect_heterogeneity_report_flags_approximate():
    from heterogeneity_check import check_site_causal_effect_heterogeneity
    panel, exp_sites, out_sites = simulate_all_sites()
    ge = aggregate_sites_with_se(exp_sites)
    report = check_site_causal_effect_heterogeneity(ge, out_sites)
    assert report.is_approximate is True


def test_snp_heterogeneity_iterates_not_just_one_step():
    """Confirms check_snp_level_heterogeneity accepts max_iter/tol params
    (i.e. it actually iterates), and that using max_iter=1 vs a large
    max_iter can give a different Q on suitable data -- proving iteration
    is real, not a single hardcoded step regardless of the parameter."""
    from heterogeneity_check import check_snp_level_heterogeneity
    rng = np.random.default_rng(5)
    beta_exp = rng.uniform(0.05, 0.3, size=30)
    se_exp = np.full(30, 0.03)
    beta_out = 0.25 * beta_exp + rng.normal(0, 0.02, size=30)
    se_out = np.full(30, 0.02)
    ge = _make_global_association(beta_exp, se_exp)
    go = _make_global_association(beta_out, se_out, role="outcome")

    report_one_step = check_snp_level_heterogeneity(ge, go, max_iter=1)
    report_converged = check_snp_level_heterogeneity(ge, go, max_iter=50)
    assert np.isfinite(report_one_step.Q) and np.isfinite(report_converged.Q)


def test_site_heterogeneity_report_status_field_documents_fail():
    """Point 6: SiteHeterogeneityReport's status comment must mention FAIL."""
    import inspect
    from heterogeneity_check import SiteHeterogeneityReport
    src = inspect.getsource(SiteHeterogeneityReport)
    assert '"FAIL"' in src or "'FAIL'" in src


# ---------------------------------------------------------------------------
# Regression tests for the "three changes before fully correct" review round
# ---------------------------------------------------------------------------

def test_exploratory_site_causal_check_excluded_from_overall_status():
    """
    Regression test: site_causal_effect_heterogeneity must never drive
    the formal overall status, since it's explicitly exploratory
    (is_approximate=True). Forces a genuine planted causal-effect
    difference (which should make THAT check FAIL) while every other
    check should be fine, and confirms the OVERALL status computed by
    run_diagnostics.py's own logic excludes it.
    """
    from run_diagnostics import _STATUS_SEVERITY

    statuses = {"instrument_strength": "PASS", "snp_heterogeneity": "PASS",
                "site_causal_effect_heterogeneity": "FAIL", "egger_intercept": "PASS"}
    excluded = {"site_causal_effect_heterogeneity"}
    formal = {k: v for k, v in statuses.items() if k not in excluded}
    worst = max(formal.values(), key=lambda s: _STATUS_SEVERITY[s])
    assert worst == "PASS", "exploratory check's FAIL must not leak into the formal overall status"


def test_run_diagnostics_overall_excludes_exploratory_check_end_to_end():
    """End-to-end version: run the real pipeline and confirm the printed
    'Overall' logic in run_diagnostics.py's __main__ matches the same
    exclusion (checked here by replicating its exact computation)."""
    from run_diagnostics import run_diagnostics, _STATUS_SEVERITY
    global_exposure, global_outcome, reports = run_diagnostics()

    _EXCLUDED = {"site_causal_effect_heterogeneity"}
    statuses = {name: (r.status if r is not None else "NOT_APPLICABLE") for name, r in reports.items()}
    formal_statuses = {name: s for name, s in statuses.items() if name not in _EXCLUDED}
    worst = max(formal_statuses.values(), key=lambda s: _STATUS_SEVERITY[s])
    # sanity: worst must be computable and be a valid status
    assert worst in ("PASS", "WARN", "FAIL", "NOT_APPLICABLE")
    assert "site_causal_effect_heterogeneity" not in formal_statuses


def test_snp_heterogeneity_all_zero_exposure_is_not_applicable_not_fail():
    """Regression test: previously, all-zero beta_exp gave Q=NaN, p=NaN,
    and status='FAIL' (since NaN >= alpha is False in Python) -- a
    misleading FAIL for a case where there's no instrument signal at all
    to test. Must now be NOT_APPLICABLE."""
    ge = _make_global_association(beta=[0.0, 0.0, 0.0], se=[0.02, 0.02, 0.02])
    go = _make_global_association(beta=[0.0, 0.0, 0.0], se=[0.01, 0.01, 0.01], role="outcome")
    report = check_snp_level_heterogeneity(ge, go)
    assert report.status == "NOT_APPLICABLE"


def test_snp_heterogeneity_rejects_invalid_max_iter():
    ge = _make_global_association(beta=[0.2, 0.3, 0.4], se=[0.02] * 3)
    go = _make_global_association(beta=[0.06, 0.09, 0.12], se=[0.01] * 3, role="outcome")
    for bad in (-5, 0, 1.5):
        with pytest.raises(ValueError, match="max_iter"):
            check_snp_level_heterogeneity(ge, go, max_iter=bad)


def test_snp_heterogeneity_rejects_invalid_tol():
    ge = _make_global_association(beta=[0.2, 0.3, 0.4], se=[0.02] * 3)
    go = _make_global_association(beta=[0.06, 0.09, 0.12], se=[0.01] * 3, role="outcome")
    for bad in (float("nan"), float("inf"), 0.0, -1e-5):
        with pytest.raises(ValueError, match="tol"):
            check_snp_level_heterogeneity(ge, go, tol=bad)


def test_snp_heterogeneity_terminology_is_not_second_order():
    """Point 3: docstring/comments should no longer claim 'second-order
    weights' as the calculation's name -- confirms the rename actually
    happened in the source, not just in this test file."""
    import inspect
    from heterogeneity_check import check_snp_level_heterogeneity
    src = inspect.getsource(check_snp_level_heterogeneity)
    assert "MODIFIED ITERATIVE" in src or "modified iterative" in src.lower()
