"""
test_federated_matches_centralized.py

Phase 2 test suite:
  1. Federated vs. centralized correctness (independent solvers)
  2. Cross-process reproducibility (full panel + all site statistics)
  3. Formula fixture tests (hand-computed small examples) for IVW/Egger/median
  4. Allele-flip invariance
  5. Site-compatibility validation (role mismatch, SNP order/ID mismatch,
     allele mismatch, empty site list)
  6. MR estimator input validation (shape/length, NaN/Inf, non-positive SE,
     zero beta_exp, too few SNPs)
  7. Simulation scenario smoke tests: weak instruments, directional
     pleiotropy, site-level heterogeneity
"""

import subprocess
import sys
import hashlib
import numpy as np
import pytest

import config
from data_simulator import (
    simulate_all_sites, simulate_all_sites_with_individuals, SiteSufficientStats, build_snp_panel,
    _derive_sufficient_stats,
)
from federated_aggregator import aggregate_sites_with_se
from run_centralized_baseline import _pool_individuals, _pooled_ols_per_snp_lstsq
from mr_estimators import ivw, mr_egger, weighted_median, MRResult


# ---------------------------------------------------------------------------
# 1. Federated vs centralized correctness (independent numerical paths)
# ---------------------------------------------------------------------------

def get_both_paths():
    panel, exp_sites, out_sites, exp_indiv, out_indiv = simulate_all_sites_with_individuals()

    fed_exposure = aggregate_sites_with_se(exp_sites)
    fed_outcome = aggregate_sites_with_se(out_sites)

    G_exp, X_exp, E = _pool_individuals(exp_indiv)
    G_out, X_out, Y = _pool_individuals(out_indiv)
    cen_exposure = _pooled_ols_per_snp_lstsq(
        G_exp, X_exp, E, role="exposure", snp_ids=panel.snp_ids,
        effect_allele=panel.effect_allele, other_allele=panel.other_allele,
    )
    cen_outcome = _pooled_ols_per_snp_lstsq(
        G_out, X_out, Y, role="outcome", snp_ids=panel.snp_ids,
        effect_allele=panel.effect_allele, other_allele=panel.other_allele,
    )

    return fed_exposure, fed_outcome, cen_exposure, cen_outcome


def test_per_snp_associations_match():
    fed_exposure, fed_outcome, cen_exposure, cen_outcome = get_both_paths()
    for fed, cen, label in [(fed_exposure, cen_exposure, "EXPOSURE"),
                             (fed_outcome, cen_outcome, "OUTCOME")]:
        np.testing.assert_allclose(
            fed.beta, cen.beta, atol=config.CORRECTNESS_ATOL, rtol=config.CORRECTNESS_RTOL,
            err_msg=f"Federated vs centralized {label} beta mismatch",
        )
        np.testing.assert_allclose(
            fed.se, cen.se, atol=config.CORRECTNESS_ATOL, rtol=config.CORRECTNESS_RTOL,
            err_msg=f"Federated vs centralized {label} se mismatch",
        )


def test_mr_estimates_match():
    fed_exposure, fed_outcome, cen_exposure, cen_outcome = get_both_paths()
    for fn in (ivw, mr_egger, weighted_median):
        fed_res = fn(fed_exposure.beta, fed_exposure.se, fed_outcome.beta, fed_outcome.se)
        cen_res = fn(cen_exposure.beta, cen_exposure.se, cen_outcome.beta, cen_outcome.se)
        assert np.isclose(fed_res.estimate, cen_res.estimate,
                           atol=config.CORRECTNESS_ATOL, rtol=config.CORRECTNESS_RTOL)
        assert np.isclose(fed_res.se, cen_res.se,
                           atol=config.CORRECTNESS_ATOL, rtol=config.CORRECTNESS_RTOL)


# ---------------------------------------------------------------------------
# 2. Cross-process reproducibility
# ---------------------------------------------------------------------------

def test_rng_reproducible_across_processes():
    snippet = (
        "import config; "
        "print(config.get_rng('exposure_site_0').normal(0, 1, 5).tobytes().hex())"
    )
    outs = []
    for hashseed in ("0", "12345"):
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True, text=True, cwd=".",
            env={"PYTHONHASHSEED": hashseed, "PATH": "/usr/bin:/bin"},
        )
        assert proc.returncode == 0, proc.stderr
        outs.append(proc.stdout.strip())
    assert outs[0] == outs[1], "RNG stream differs across PYTHONHASHSEED values / processes"


def test_full_simulation_reproducible_across_processes():
    """
    Hashes the ENTIRE panel (SNP ids, alleles, maf, beta_gx, weak/pleio
    flags, pleiotropy effects) and every field of every exposure/outcome
    site's sufficient statistics — not just one array from one site — and
    compares that combined hash across two subprocesses with different
    PYTHONHASHSEED values. A regression that only changes, say, an
    outcome-site field or the panel's allele coding would be invisible to
    a narrower single-array check.
    """
    snippet = (
        "import hashlib\n"
        "from data_simulator import simulate_all_sites\n"
        "panel, exp_sites, out_sites = simulate_all_sites()\n"
        "h = hashlib.sha256()\n"
        "for arr in (panel.snp_ids, panel.effect_allele, panel.other_allele, panel.maf,\n"
        "            panel.beta_gx, panel.is_weak_instrument, panel.alpha_pleiotropy,\n"
        "            panel.is_pleiotropic):\n"
        "    h.update(str(list(arr)).encode())\n"
        "for site in list(exp_sites) + list(out_sites):\n"
        "    for field_name in ('site_name', 'site_role', 'n', 'sum_G', 'sum_X', 'sum_G2',\n"
        "                       'sum_GX', 'sum_X2', 'sum_Y', 'sum_GY', 'sum_XY', 'sum_Y2',\n"
        "                       'snp_ids', 'effect_allele', 'other_allele'):\n"
        "        val = getattr(site, field_name)\n"
        "        h.update(str(list(val) if hasattr(val, '__iter__') and not isinstance(val, str) else val).encode())\n"
        "print(h.hexdigest())\n"
    )
    outs = []
    for hashseed in ("1", "999"):
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True, text=True, cwd=".",
            env={"PYTHONHASHSEED": hashseed, "PATH": "/usr/bin:/bin"},
        )
        assert proc.returncode == 0, proc.stderr
        outs.append(proc.stdout.strip())
    assert outs[0] == outs[1], "Full simulation (panel + all site stats) differs across processes"


# ---------------------------------------------------------------------------
# 3. Formula fixture tests — small hand-checkable examples
# ---------------------------------------------------------------------------

def test_ivw_matches_hand_computed_fixed_effect():
    beta_exp = np.array([1.0, 2.0])
    se_exp = np.array([0.1, 0.1])
    beta_out = np.array([0.5, 1.0])   # ratios: 0.5, 0.5 -> should recover slope 0.5 exactly
    se_out = np.array([0.1, 0.1])

    res = ivw(beta_exp, se_exp, beta_out, se_out)
    assert np.isclose(res.estimate, 0.5, atol=1e-10)


def test_weighted_median_matches_simple_median_with_equal_weights():
    beta_exp = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    se_exp = np.full(5, 0.01)
    beta_out = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    se_out = np.full(5, 0.1)

    res = weighted_median(beta_exp, se_exp, beta_out, se_out, n_bootstrap=200,
                           rng=np.random.default_rng(0))
    assert np.isclose(res.estimate, 0.3, atol=1e-10)


def test_mr_egger_recovers_known_slope_and_intercept():
    rng = np.random.default_rng(1)
    beta_exp = rng.uniform(0.05, 0.3, size=20)
    se_exp = np.full(20, 0.02)
    true_intercept, true_slope = 0.01, 0.4
    beta_out = true_intercept + true_slope * beta_exp
    se_out = np.full(20, 0.05)

    res = mr_egger(beta_exp, se_exp, beta_out, se_out)
    assert np.isclose(res.estimate, true_slope, atol=1e-8)
    assert np.isclose(res.extra["intercept"], true_intercept, atol=1e-8)


# ---------------------------------------------------------------------------
# 4. Allele-flip invariance
# ---------------------------------------------------------------------------

def _flip_random_snps(beta_exp, beta_out, frac=0.4, seed=0):
    rng = np.random.default_rng(seed)
    n = len(beta_exp)
    flip = rng.random(n) < frac
    sign = np.where(flip, -1.0, 1.0)
    return beta_exp * sign, beta_out * sign


def test_ivw_is_allele_flip_invariant():
    rng = np.random.default_rng(2)
    beta_exp = rng.uniform(0.05, 0.3, size=15)
    se_exp = np.full(15, 0.02)
    beta_out = rng.uniform(-0.1, 0.2, size=15)
    se_out = np.full(15, 0.05)

    res_orig = ivw(beta_exp, se_exp, beta_out, se_out)
    beta_exp_f, beta_out_f = _flip_random_snps(beta_exp, beta_out)
    res_flipped = ivw(beta_exp_f, se_exp, beta_out_f, se_out)

    assert np.isclose(res_orig.estimate, res_flipped.estimate, atol=1e-10)
    assert np.isclose(res_orig.se, res_flipped.se, atol=1e-10)


def test_weighted_median_is_allele_flip_invariant():
    rng = np.random.default_rng(3)
    beta_exp = rng.uniform(0.05, 0.3, size=15)
    se_exp = np.full(15, 0.02)
    beta_out = rng.uniform(-0.1, 0.2, size=15)
    se_out = np.full(15, 0.05)

    res_orig = weighted_median(beta_exp, se_exp, beta_out, se_out, n_bootstrap=500,
                                rng=np.random.default_rng(0))
    beta_exp_f, beta_out_f = _flip_random_snps(beta_exp, beta_out)
    res_flipped = weighted_median(beta_exp_f, se_exp, beta_out_f, se_out, n_bootstrap=500,
                                   rng=np.random.default_rng(0))

    assert np.isclose(res_orig.estimate, res_flipped.estimate, atol=1e-8)


def test_mr_egger_slope_is_allele_flip_invariant():
    rng = np.random.default_rng(4)
    beta_exp = rng.uniform(0.05, 0.3, size=25)
    se_exp = np.full(25, 0.02)
    beta_out = 0.02 + 0.35 * beta_exp + rng.normal(0, 0.01, size=25)
    se_out = np.full(25, 0.05)

    res_orig = mr_egger(beta_exp, se_exp, beta_out, se_out)
    beta_exp_f, beta_out_f = _flip_random_snps(beta_exp, beta_out, frac=0.5, seed=7)
    res_flipped = mr_egger(beta_exp_f, se_exp, beta_out_f, se_out)

    assert np.isclose(res_orig.estimate, res_flipped.estimate, atol=1e-8)
    assert np.isclose(res_orig.extra["intercept"], res_flipped.extra["intercept"], atol=1e-8)


# ---------------------------------------------------------------------------
# 5. Site-compatibility validation (federated_aggregator.py)
# ---------------------------------------------------------------------------

def _make_site(role, n_snps=5, n_people=100, snp_ids=None, effect_allele=None, other_allele=None, seed=0):
    rng = np.random.default_rng(seed)
    if snp_ids is None:
        snp_ids = np.array([f"rs{i}" for i in range(n_snps)])
    if effect_allele is None:
        effect_allele = np.array(["A"] * n_snps)
    if other_allele is None:
        other_allele = np.array(["G"] * n_snps)
    n = np.full(n_snps, n_people, dtype=float)
    ones_arr = lambda: rng.normal(10, 1, size=n_snps)
    return SiteSufficientStats(
        site_name=f"site_{role}_{seed}", site_role=role, n_snps=n_snps,
        snp_ids=snp_ids, effect_allele=effect_allele, other_allele=other_allele,
        n=n, sum_G=ones_arr(), sum_X=ones_arr(), sum_G2=ones_arr(), sum_GX=ones_arr(),
        sum_X2=ones_arr(), sum_Y=ones_arr(), sum_GY=ones_arr(), sum_XY=ones_arr(),
        sum_Y2=np.full(n_snps, 500.0),
    )


def test_aggregator_rejects_empty_site_list():
    with pytest.raises(ValueError, match="empty site list"):
        aggregate_sites_with_se([])


def test_aggregator_rejects_mixed_roles():
    sites = [_make_site("exposure", seed=0), _make_site("outcome", seed=1)]
    with pytest.raises(ValueError, match="mixed roles"):
        aggregate_sites_with_se(sites)


def test_aggregator_rejects_reordered_snp_ids():
    ids_a = np.array(["rs1", "rs2", "rs3"])
    ids_b = np.array(["rs2", "rs1", "rs3"])  # same set, different order
    site_a = _make_site("exposure", n_snps=3, snp_ids=ids_a, seed=0)
    site_b = _make_site("exposure", n_snps=3, snp_ids=ids_b, seed=1)
    with pytest.raises(ValueError, match="snp_ids"):
        aggregate_sites_with_se([site_a, site_b])


def test_aggregator_rejects_mismatched_snp_ids():
    ids_a = np.array(["rs1", "rs2", "rs3"])
    ids_b = np.array(["rs1", "rs2", "rs4"])
    site_a = _make_site("exposure", n_snps=3, snp_ids=ids_a, seed=0)
    site_b = _make_site("exposure", n_snps=3, snp_ids=ids_b, seed=1)
    with pytest.raises(ValueError, match="snp_ids"):
        aggregate_sites_with_se([site_a, site_b])


def test_aggregator_rejects_mismatched_alleles():
    site_a = _make_site("exposure", n_snps=3, effect_allele=np.array(["A", "A", "A"]), seed=0)
    site_b = _make_site("exposure", n_snps=3, effect_allele=np.array(["A", "T", "A"]), seed=1)
    with pytest.raises(ValueError, match="allele"):
        aggregate_sites_with_se([site_a, site_b])


def test_aggregator_calls_each_sites_validate():
    """A structurally invalid site (duplicate SNP id) must be caught even
    if it's the only site in the list (no cross-site comparison possible)."""
    bad_ids = np.array(["rs1", "rs1", "rs3"])
    bad_site = _make_site("exposure", n_snps=3, snp_ids=bad_ids, seed=0)
    with pytest.raises(ValueError):
        aggregate_sites_with_se([bad_site])


def test_global_association_preserves_snp_metadata():
    panel, exp_sites, out_sites = simulate_all_sites()
    ge = aggregate_sites_with_se(exp_sites)
    assert np.array_equal(ge.snp_ids, panel.snp_ids)
    assert np.array_equal(ge.effect_allele, panel.effect_allele)
    assert np.array_equal(ge.other_allele, panel.other_allele)


def test_aggregator_raises_on_monomorphic_snp():
    panel = build_snp_panel()
    n_people = 100
    G = np.zeros((n_people, config.N_SNPS))
    X = np.random.default_rng(0).normal(size=n_people)
    Y = np.random.default_rng(1).normal(size=n_people)
    site = _derive_sufficient_stats("bad_site", "exposure", panel, G, X, Y)
    with pytest.raises(np.linalg.LinAlgError):
        aggregate_sites_with_se([site])


def test_site_validate_rejects_mismatched_snp_ids():
    from data_simulator import SiteSufficientStats
    n = 5
    bad = SiteSufficientStats(
        site_name="bad", site_role="exposure", n_snps=n,
        snp_ids=np.array(["rs1", "rs1", "rs3", "rs4", "rs5"]),
        effect_allele=np.array(["A"] * n), other_allele=np.array(["G"] * n),
        n=np.full(n, 100.0), sum_G=np.zeros(n), sum_X=np.zeros(n), sum_G2=np.zeros(n),
        sum_GX=np.zeros(n), sum_X2=np.zeros(n), sum_Y=np.zeros(n), sum_GY=np.zeros(n),
        sum_XY=np.zeros(n), sum_Y2=np.zeros(n),
    )
    with pytest.raises(ValueError):
        bad.validate()


def test_site_validate_rejects_non_finite_values():
    from data_simulator import SiteSufficientStats
    n = 3
    bad = SiteSufficientStats(
        site_name="bad", site_role="outcome", n_snps=n,
        snp_ids=np.array(["rs1", "rs2", "rs3"]),
        effect_allele=np.array(["A"] * n), other_allele=np.array(["G"] * n),
        n=np.full(n, 100.0), sum_G=np.array([1.0, np.nan, 3.0]), sum_X=np.zeros(n),
        sum_G2=np.zeros(n), sum_GX=np.zeros(n), sum_X2=np.zeros(n), sum_Y=np.zeros(n),
        sum_GY=np.zeros(n), sum_XY=np.zeros(n), sum_Y2=np.zeros(n),
    )
    with pytest.raises(ValueError):
        bad.validate()


# ---------------------------------------------------------------------------
# 6. MR estimator input validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [ivw, mr_egger, weighted_median])
def test_estimator_rejects_mismatched_array_lengths(fn):
    beta_exp = np.array([0.1, 0.2, 0.3])
    se_exp = np.array([0.01, 0.01])  # wrong length
    beta_out = np.array([0.05, 0.1, 0.15])
    se_out = np.array([0.02, 0.02, 0.02])
    with pytest.raises(ValueError, match="length mismatch"):
        fn(beta_exp, se_exp, beta_out, se_out)


@pytest.mark.parametrize("fn", [ivw, mr_egger, weighted_median])
def test_estimator_rejects_nan_input(fn):
    n = 5
    beta_exp = np.full(n, 0.1)
    beta_exp[2] = np.nan
    se_exp = np.full(n, 0.01)
    beta_out = np.full(n, 0.05)
    se_out = np.full(n, 0.02)
    with pytest.raises(ValueError, match="NaN or infinite"):
        fn(beta_exp, se_exp, beta_out, se_out)


@pytest.mark.parametrize("fn", [ivw, mr_egger, weighted_median])
def test_estimator_rejects_infinite_input(fn):
    n = 5
    beta_out = np.full(n, 0.05)
    beta_out[0] = np.inf
    beta_exp = np.full(n, 0.1)
    se_exp = np.full(n, 0.01)
    se_out = np.full(n, 0.02)
    with pytest.raises(ValueError, match="NaN or infinite"):
        fn(beta_exp, se_exp, beta_out, se_out)


@pytest.mark.parametrize("fn", [ivw, mr_egger, weighted_median])
def test_estimator_rejects_zero_se(fn):
    n = 5
    beta_exp = np.full(n, 0.1)
    se_exp = np.full(n, 0.01)
    beta_out = np.full(n, 0.05)
    se_out = np.full(n, 0.02)
    se_out[1] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        fn(beta_exp, se_exp, beta_out, se_out)


@pytest.mark.parametrize("fn", [ivw, mr_egger, weighted_median])
def test_estimator_rejects_negative_se(fn):
    n = 5
    beta_exp = np.full(n, 0.1)
    se_exp = np.full(n, 0.01)
    se_exp[0] = -0.01
    beta_out = np.full(n, 0.05)
    se_out = np.full(n, 0.02)
    with pytest.raises(ValueError, match="strictly positive"):
        fn(beta_exp, se_exp, beta_out, se_out)


def test_weighted_median_rejects_zero_beta_exp():
    n = 5
    beta_exp = np.full(n, 0.1)
    beta_exp[3] = 0.0
    se_exp = np.full(n, 0.01)
    beta_out = np.full(n, 0.05)
    se_out = np.full(n, 0.02)
    with pytest.raises(ValueError, match="beta_exp contains an exact 0"):
        weighted_median(beta_exp, se_exp, beta_out, se_out)


def test_ivw_allows_zero_beta_exp():
    """IVW's formula never divides by beta_exp per SNP (only sums w*beta_exp*beta_out
    and w*beta_exp^2), so a zero-effect SNP is valid input, not an error."""
    beta_exp = np.array([0.1, 0.0, 0.2])
    se_exp = np.full(3, 0.01)
    beta_out = np.array([0.05, 0.0, 0.1])
    se_out = np.full(3, 0.02)
    res = ivw(beta_exp, se_exp, beta_out, se_out)
    assert np.isfinite(res.estimate)


def test_ivw_accepts_single_snp():
    res = ivw(np.array([0.2]), np.array([0.02]), np.array([0.06]), np.array([0.01]))
    assert np.isfinite(res.estimate)


def test_mr_egger_rejects_too_few_snps():
    for n in (1, 2):
        with pytest.raises(ValueError, match="at least 3"):
            mr_egger(np.full(n, 0.1), np.full(n, 0.01), np.full(n, 0.05), np.full(n, 0.02))


def test_weighted_median_rejects_too_few_snps():
    for n in (1, 2):
        with pytest.raises(ValueError, match="at least 3"):
            weighted_median(np.full(n, 0.1), np.full(n, 0.01), np.full(n, 0.05), np.full(n, 0.02))


def test_estimator_rejects_non_1d_input():
    beta_exp = np.array([[0.1, 0.2], [0.3, 0.4]])  # 2-D by mistake
    se_exp = np.full(2, 0.01)
    beta_out = np.full(2, 0.05)
    se_out = np.full(2, 0.02)
    with pytest.raises(ValueError, match="1-D"):
        ivw(beta_exp, se_exp, beta_out, se_out)


# ---------------------------------------------------------------------------
# 7. Simulation scenario smoke tests
# ---------------------------------------------------------------------------

def test_weak_instrument_snps_have_attenuated_beta_gx():
    old_frac = config.WEAK_INSTRUMENT_FRACTION
    config.WEAK_INSTRUMENT_FRACTION = 0.3
    try:
        panel = build_snp_panel()
        weak_mean_abs = np.mean(np.abs(panel.beta_gx[panel.is_weak_instrument]))
        strong_mean_abs = np.mean(np.abs(panel.beta_gx[~panel.is_weak_instrument]))
        assert weak_mean_abs < strong_mean_abs
    finally:
        config.WEAK_INSTRUMENT_FRACTION = old_frac


def test_directional_pleiotropy_smoke():
    """
    Simulation smoke test, NOT a statistical power/calibration check: just
    confirms directional pleiotropy in the generative model produces a
    nonzero Egger intercept in this one draw. Proving the Egger intercept
    test statistically detects directional pleiotropy at a controlled
    false-positive rate requires evaluating power/FPR over MANY repeated
    simulations — that belongs to the "diagnostic calibration" success
    criterion in the problem statement, not this single-draw smoke test.
    """
    old_mean = config.PLEIOTROPY_MEAN
    old_frac = config.PLEIOTROPY_FRACTION
    config.PLEIOTROPY_MEAN = 0.15
    config.PLEIOTROPY_FRACTION = 0.4
    try:
        panel, exp_sites, out_sites = simulate_all_sites_with_individuals()[:3]
        ge = aggregate_sites_with_se(exp_sites)
        go = aggregate_sites_with_se(out_sites)
        egger_res = mr_egger(ge.beta, ge.se, go.beta, go.se)
        assert abs(egger_res.extra["intercept"]) > 1e-4
    finally:
        config.PLEIOTROPY_MEAN = old_mean
        config.PLEIOTROPY_FRACTION = old_frac


def test_site_heterogeneity_is_recorded_per_outcome_site():
    old_sd = config.SITE_HETEROGENEITY_SD
    config.SITE_HETEROGENEITY_SD = 0.1
    try:
        panel, exp_sites, out_sites = simulate_all_sites_with_individuals()[:3]
        realized = [s.site_realized_causal_effect for s in out_sites]
        assert all(r is not None for r in realized)
        assert len(set(np.round(realized, 6))) > 1
    finally:
        config.SITE_HETEROGENEITY_SD = old_sd


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
