"""
test_multicov_federated_equivalence.py

Phase (real-data prep) Milestone 3 exact-equivalence proof: the
generalized matrix-sufficient-statistic aggregator (multicov_federated.py)
must reproduce pooled centralized OLS with
p >= 2 covariates, the same way federated_aggregator.py was already
proven equivalent to centralized OLS for p == 1
(test_federated_matches_centralized.py). This must pass BEFORE the
generalized engine is ever pointed at real MAGE/1000G data
(Section 19, Milestone 3 / Milestone 4 ordering).
"""

import numpy as np
import pytest

from multicov_federated import (
    derive_matrix_sufficient_stats, SiteMatrixStats,
    aggregate_matrix_sites, GlobalMatrixAssociation,
)


def _make_snp_panel(rng, n_snps):
    snp_ids = np.array([f"rs{i}" for i in range(n_snps)])
    effect_allele = np.array(["A"] * n_snps)
    other_allele = np.array(["G"] * n_snps)
    maf = rng.uniform(0.1, 0.5, size=n_snps)
    return snp_ids, effect_allele, other_allele, maf


def _simulate_site(rng, n_people, n_snps, maf, p, true_beta_G, true_beta_C, true_intercept):
    """
    G here has n_snps GENUINELY DISTINCT genotype columns (independent
    binomial draws per SNP, as in the real panel) — only SNP 0 actually
    drives y (the others are "genotyped but non-causal", which is exactly
    what a real SNP panel looks like: most SNPs tested are not the causal
    one). This matters because the equivalence check is per-SNP: if every
    SNP column were identical (as an earlier version of this test did via
    np.tile), every SNP's federated-vs-centralized comparison would
    silently be re-testing the same single genotype vector instead of
    exercising the aggregator across distinct data.
    """
    G = rng.binomial(2, maf, size=(n_people, n_snps)).astype(float)
    C = rng.normal(0.0, 1.0, size=(n_people, p)) if p > 0 else np.zeros((n_people, 0))
    noise = rng.normal(0.0, 1.0, size=n_people)
    y = true_intercept + true_beta_G * G[:, 0] + (C @ true_beta_C if p > 0 else 0.0) + noise
    return G, C, y


def _centralized_ols_beta_G_se(G_col, C, y):
    """
    Independent reference path. Coefficients come from np.linalg.lstsq
    (SVD-based). Critically, the SE must NOT reuse
    np.linalg.inv(Z.T @ Z) — that is the exact same normal-equations +
    explicit-inverse pathway multicov_federated.py's aggregator uses internally, so
    reusing it here would let a shared bug in that pathway (e.g. in how
    coefficient index 1 maps to the genotype variance) pass undetected.
    Instead (Z'Z)^-1 is built from the SVD of Z itself, same technique
    run_centralized_baseline.py already uses for the original 1-covariate
    aggregator: Z = U S V', so (Z'Z)^-1 = V diag(1/S^2) V', never forming
    Z'Z or inverting it directly.
    """
    n = len(y)
    p = C.shape[1]
    ones = np.ones(n)
    Z = np.column_stack([ones, G_col, C]) if p > 0 else np.column_stack([ones, G_col])
    coef, _, rank, _ = np.linalg.lstsq(Z, y, rcond=None)
    d = Z.shape[1]
    if rank < d:
        raise np.linalg.LinAlgError("rank-deficient design in centralized reference")
    resid = y - Z @ coef
    rss = float(resid @ resid)
    dof = n - d
    sigma2 = rss / dof

    _, S, Vt = np.linalg.svd(Z, full_matrices=False)
    ZtZ_inv = (Vt.T * (1.0 / S ** 2)) @ Vt
    var_bG = sigma2 * ZtZ_inv[1, 1]
    return coef[1], np.sqrt(var_bG), sigma2


@pytest.mark.parametrize("p", [0, 1, 2, 4])
def test_federated_matches_centralized_multicov(p):
    """
    Core equivalence test: for p in {0, 1, 2, 4} covariates, aggregating
    per-site matrix sufficient statistics must give beta_G/se_G
    numerically identical (to solver tolerance) to pooling all
    individuals and running OLS directly via np.linalg.lstsq — an
    independent numerical path, mirroring
    run_centralized_baseline.py's use of lstsq vs. the aggregator's
    normal-equations solve.
    """
    rng = np.random.default_rng(1234 + p)
    n_snps = 5
    site_sizes = [300, 150, 500, 220]

    snp_ids, effect_allele, other_allele, maf = _make_snp_panel(rng, n_snps)
    true_beta_G = 0.4
    true_beta_C = rng.normal(0.0, 0.3, size=p) if p > 0 else np.zeros(0)
    true_intercept = 1.5
    covariate_names = tuple(f"C{k}" for k in range(p))

    sites = []
    G_all, C_all, y_all = [], [], []
    for i, n_people in enumerate(site_sizes):
        G_full, C, y = _simulate_site(rng, n_people, n_snps, maf, p, true_beta_G, true_beta_C, true_intercept)
        site = derive_matrix_sufficient_stats(
            site_name=f"site_{i}", site_role="exposure",
            snp_ids=snp_ids, effect_allele=effect_allele, other_allele=other_allele,
            G=G_full, C=C, y=y, covariate_names=covariate_names,
        )
        sites.append(site)
        G_all.append(G_full)
        C_all.append(C)
        y_all.append(y)

    global_assoc = aggregate_matrix_sites(sites)

    G_pooled = np.vstack(G_all)
    C_pooled = np.vstack(C_all) if p > 0 else np.zeros((sum(site_sizes), 0))
    y_pooled = np.concatenate(y_all)

    for j in range(n_snps):
        beta_ref, se_ref, sigma2_ref = _centralized_ols_beta_G_se(G_pooled[:, j], C_pooled, y_pooled)
        # atol=1e-10, rtol=1e-9: this is the project's definition of completion
        # for the multi-covariate generalization — federated must reproduce
        # pooled lstsq to numerical precision, not just "close enough".
        np.testing.assert_allclose(
            global_assoc.beta_G[j], beta_ref, atol=1e-10, rtol=1e-9,
            err_msg=f"SNP {j}, p={p}: federated beta_G != centralized beta_ref",
        )
        np.testing.assert_allclose(
            global_assoc.se_G[j], se_ref, atol=1e-10, rtol=1e-9,
            err_msg=f"SNP {j}, p={p}: federated se_G != centralized se_ref",
        )
        # Residual variance (sigma^2) is itself part of the real-data
        # correctness benchmark per the roadmap, not just an internal to
        # beta/se — check it explicitly rather than only via its effect on se.
        np.testing.assert_allclose(
            global_assoc.residual_variance[j], sigma2_ref, atol=1e-10, rtol=1e-9,
            err_msg=f"SNP {j}, p={p}: federated residual_variance != centralized sigma2_ref",
        )

    assert np.all(global_assoc.n_total == sum(site_sizes))
    assert global_assoc.covariate_names == covariate_names


def test_p1_matches_original_10sum_aggregator():
    """
    Sanity anchor: with exactly p=1 covariate, the generalized engine's
    beta_G/se_G must agree with the ORIGINAL federated_aggregator.py
    (the one already validated in test_federated_matches_centralized.py)
    on the same underlying data — proving the generalization is a strict
    superset of the old design, not an accidental behavior change for
    the case that was already shipped and trusted.
    """
    from data_simulator import build_snp_panel, simulate_exposure_site
    from federated_aggregator import aggregate_sites_with_se
    import config

    panel = build_snp_panel()
    exp_sites_old = [
        simulate_exposure_site(i, n, panel)
        for i, n in enumerate(config.EXPOSURE_SITE_SIZES)
    ]
    old_global = aggregate_sites_with_se(exp_sites_old)

    # Rebuild the identical individuals through the generalized path.
    # data_simulator's internal generator is reused via the private
    # helper so both paths see EXACTLY the same G/X/trait draws.
    from data_simulator import _simulate_exposure_individuals

    new_sites = []
    for i, n in enumerate(config.EXPOSURE_SITE_SIZES):
        d = _simulate_exposure_individuals(i, n, panel)
        site = derive_matrix_sufficient_stats(
            site_name=d["site_name"], site_role="exposure",
            snp_ids=panel.snp_ids, effect_allele=panel.effect_allele, other_allele=panel.other_allele,
            G=d["G"], C=d["X"][:, None], y=d["trait"], covariate_names=("X",),
        )
        new_sites.append(site)

    new_global = aggregate_matrix_sites(new_sites)

    assert np.allclose(old_global.beta, new_global.beta_G, rtol=1e-9, atol=1e-10)
    assert np.allclose(old_global.se, new_global.se_G, rtol=1e-9, atol=1e-10)


def test_covariate_name_mismatch_rejected():
    rng = np.random.default_rng(0)
    snp_ids, effect_allele, other_allele, maf = _make_snp_panel(rng, 3)
    G1, C1, y1 = _simulate_site(rng, 100, 3, maf, 2, 0.4, np.array([0.1, 0.2]), 1.0)
    G2, C2, y2 = _simulate_site(rng, 100, 3, maf, 2, 0.4, np.array([0.1, 0.2]), 1.0)

    site1 = derive_matrix_sufficient_stats(
        "s1", "exposure", snp_ids, effect_allele, other_allele, G1, C1, y1,
        covariate_names=("PC1", "PC2"),
    )
    site2 = derive_matrix_sufficient_stats(
        "s2", "exposure", snp_ids, effect_allele, other_allele, G2, C2, y2,
        covariate_names=("PC1", "sex"),  # deliberately different
    )

    with pytest.raises(ValueError, match="covariate_names"):
        aggregate_matrix_sites([site1, site2])


def test_singular_design_raises():
    """A covariate that's an exact copy of genotype makes A singular — must raise, not silently return garbage."""
    rng = np.random.default_rng(0)
    n_people, n_snps = 100, 2
    snp_ids, effect_allele, other_allele, maf = _make_snp_panel(rng, n_snps)
    G = rng.binomial(2, maf, size=(n_people, n_snps)).astype(float)
    C = G[:, [0]].copy()  # exact copy of SNP 0's genotype column
    y = rng.normal(size=n_people)

    site = derive_matrix_sufficient_stats(
        "s1", "exposure", snp_ids, effect_allele, other_allele, G, C, y,
        covariate_names=("dup_of_snp0",),
    )
    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        aggregate_matrix_sites([site])


def test_covariate_order_mismatch_rejected():
    """('PC1','PC2') and ('PC2','PC1') are different designs — column 2
    means something different in each — and must be rejected even though
    the two sites agree on which covariate NAMES are present."""
    rng = np.random.default_rng(0)
    snp_ids, effect_allele, other_allele, maf = _make_snp_panel(rng, 3)
    G1, C1, y1 = _simulate_site(rng, 100, 3, maf, 2, 0.4, np.array([0.1, 0.2]), 1.0)
    G2, C2, y2 = _simulate_site(rng, 100, 3, maf, 2, 0.4, np.array([0.1, 0.2]), 1.0)

    site1 = derive_matrix_sufficient_stats(
        "s1", "exposure", snp_ids, effect_allele, other_allele, G1, C1, y1,
        covariate_names=("PC1", "PC2"),
    )
    site2 = derive_matrix_sufficient_stats(
        "s2", "exposure", snp_ids, effect_allele, other_allele, G2, C2[:, ::-1], y2,
        covariate_names=("PC2", "PC1"),  # same names, reversed order
    )

    with pytest.raises(ValueError, match="covariate_names"):
        aggregate_matrix_sites([site1, site2])


def test_empty_site_list_rejected():
    with pytest.raises(ValueError, match="empty site list"):
        aggregate_matrix_sites([])


def test_mismatched_snp_count_rejected():
    rng = np.random.default_rng(0)
    snp_ids3, ea3, oa3, maf3 = _make_snp_panel(rng, 3)
    snp_ids5, ea5, oa5, maf5 = _make_snp_panel(rng, 5)
    G1, C1, y1 = _simulate_site(rng, 100, 3, maf3, 1, 0.4, np.array([0.1]), 1.0)
    G2, C2, y2 = _simulate_site(rng, 100, 5, maf5, 1, 0.4, np.array([0.1]), 1.0)

    site1 = derive_matrix_sufficient_stats(
        "s1", "exposure", snp_ids3, ea3, oa3, G1, C1, y1, covariate_names=("X",),
    )
    site2 = derive_matrix_sufficient_stats(
        "s2", "exposure", snp_ids5, ea5, oa5, G2, C2, y2, covariate_names=("X",),
    )
    with pytest.raises(ValueError, match="n_snps"):
        aggregate_matrix_sites([site1, site2])


def test_reordered_snp_ids_rejected():
    """Same SNP set, different order — summing these would silently add
    SNP j's stats at one site to a DIFFERENT SNP j's stats at another."""
    rng = np.random.default_rng(0)
    snp_ids, ea, oa, maf = _make_snp_panel(rng, 4)
    G1, C1, y1 = _simulate_site(rng, 100, 4, maf, 1, 0.4, np.array([0.1]), 1.0)
    G2, C2, y2 = _simulate_site(rng, 100, 4, maf, 1, 0.4, np.array([0.1]), 1.0)

    site1 = derive_matrix_sufficient_stats(
        "s1", "exposure", snp_ids, ea, oa, G1, C1, y1, covariate_names=("X",),
    )
    reordered = np.array([snp_ids[1], snp_ids[0], snp_ids[3], snp_ids[2]])
    site2 = derive_matrix_sufficient_stats(
        "s2", "exposure", reordered, ea, oa, G2, C2, y2, covariate_names=("X",),
    )
    with pytest.raises(ValueError, match="snp_ids"):
        aggregate_matrix_sites([site1, site2])


def test_allele_mismatch_rejected():
    rng = np.random.default_rng(0)
    snp_ids, ea, oa, maf = _make_snp_panel(rng, 3)
    G1, C1, y1 = _simulate_site(rng, 100, 3, maf, 1, 0.4, np.array([0.1]), 1.0)
    G2, C2, y2 = _simulate_site(rng, 100, 3, maf, 1, 0.4, np.array([0.1]), 1.0)

    site1 = derive_matrix_sufficient_stats(
        "s1", "exposure", snp_ids, ea, oa, G1, C1, y1, covariate_names=("X",),
    )
    ea_flipped = np.array(["T"] * 3)  # deliberately different effect allele coding
    site2 = derive_matrix_sufficient_stats(
        "s2", "exposure", snp_ids, ea_flipped, oa, G2, C2, y2, covariate_names=("X",),
    )
    with pytest.raises(ValueError, match="allele"):
        aggregate_matrix_sites([site1, site2])


def test_covariate_count_mismatch_rejected():
    """Different number of covariates -> different covariate_names length
    -> must be rejected (a same-length-but-wrong-name case is already
    covered by test_covariate_name_mismatch_rejected)."""
    rng = np.random.default_rng(0)
    snp_ids, ea, oa, maf = _make_snp_panel(rng, 3)
    G1, C1, y1 = _simulate_site(rng, 100, 3, maf, 1, 0.4, np.array([0.1]), 1.0)
    G2, C2, y2 = _simulate_site(rng, 100, 3, maf, 2, 0.4, np.array([0.1, 0.2]), 1.0)

    site1 = derive_matrix_sufficient_stats(
        "s1", "exposure", snp_ids, ea, oa, G1, C1, y1, covariate_names=("X",),
    )
    site2 = derive_matrix_sufficient_stats(
        "s2", "exposure", snp_ids, ea, oa, G2, C2, y2, covariate_names=("X", "Y"),
    )
    with pytest.raises(ValueError, match="covariate_names"):
        aggregate_matrix_sites([site1, site2])


# ---- derive_matrix_sufficient_stats input validation ----

def test_derive_rejects_row_count_mismatch_C():
    rng = np.random.default_rng(0)
    snp_ids, ea, oa, maf = _make_snp_panel(rng, 3)
    G = rng.binomial(2, maf, size=(100, 3)).astype(float)
    C = rng.normal(size=(99, 1))  # one fewer row than G/y
    y = rng.normal(size=100)
    with pytest.raises(ValueError, match="shape"):
        derive_matrix_sufficient_stats("s1", "exposure", snp_ids, ea, oa, G, C, y, covariate_names=("X",))


def test_derive_rejects_covariate_names_length_mismatch():
    rng = np.random.default_rng(0)
    snp_ids, ea, oa, maf = _make_snp_panel(rng, 3)
    G = rng.binomial(2, maf, size=(100, 3)).astype(float)
    C = rng.normal(size=(100, 2))
    y = rng.normal(size=100)
    with pytest.raises(ValueError, match="covariate_names"):
        derive_matrix_sufficient_stats("s1", "exposure", snp_ids, ea, oa, G, C, y, covariate_names=("X",))  # only 1 name for 2 columns


def test_derive_rejects_y_length_mismatch():
    rng = np.random.default_rng(0)
    snp_ids, ea, oa, maf = _make_snp_panel(rng, 3)
    G = rng.binomial(2, maf, size=(100, 3)).astype(float)
    C = rng.normal(size=(100, 1))
    y = rng.normal(size=99)  # wrong length
    with pytest.raises(ValueError, match="y must have shape"):
        derive_matrix_sufficient_stats("s1", "exposure", snp_ids, ea, oa, G, C, y, covariate_names=("X",))


def test_derive_rejects_nan_in_G():
    rng = np.random.default_rng(0)
    snp_ids, ea, oa, maf = _make_snp_panel(rng, 3)
    G = rng.binomial(2, maf, size=(100, 3)).astype(float)
    G[5, 1] = np.nan
    C = rng.normal(size=(100, 1))
    y = rng.normal(size=100)
    with pytest.raises(ValueError, match="non-finite|NaN"):
        derive_matrix_sufficient_stats("s1", "exposure", snp_ids, ea, oa, G, C, y, covariate_names=("X",))


def test_derive_rejects_inf_in_y():
    rng = np.random.default_rng(0)
    snp_ids, ea, oa, maf = _make_snp_panel(rng, 3)
    G = rng.binomial(2, maf, size=(100, 3)).astype(float)
    C = rng.normal(size=(100, 1))
    y = rng.normal(size=100)
    y[0] = np.inf
    with pytest.raises(ValueError, match="non-finite|Inf"):
        derive_matrix_sufficient_stats("s1", "exposure", snp_ids, ea, oa, G, C, y, covariate_names=("X",))


def test_wrong_length_effect_allele_rejected():
    rng = np.random.default_rng(0)
    snp_ids, ea, oa, maf = _make_snp_panel(rng, 4)
    G = rng.binomial(2, maf, size=(100, 4)).astype(float)
    C = rng.normal(size=(100, 1))
    y = rng.normal(size=100)
    short_ea = ea[:3]  # one short of n_snps=4
    with pytest.raises(ValueError, match="effect_allele"):
        derive_matrix_sufficient_stats(
            "s1", "exposure", snp_ids, short_ea, oa, G, C, y, covariate_names=("X",),
        )


def test_wrong_length_other_allele_rejected():
    rng = np.random.default_rng(0)
    snp_ids, ea, oa, maf = _make_snp_panel(rng, 4)
    G = rng.binomial(2, maf, size=(100, 4)).astype(float)
    C = rng.normal(size=(100, 1))
    y = rng.normal(size=100)
    short_oa = oa[:3]
    with pytest.raises(ValueError, match="other_allele"):
        derive_matrix_sufficient_stats(
            "s1", "exposure", snp_ids, ea, short_oa, G, C, y, covariate_names=("X",),
        )


def test_duplicate_covariate_names_rejected():
    rng = np.random.default_rng(0)
    snp_ids, ea, oa, maf = _make_snp_panel(rng, 3)
    G = rng.binomial(2, maf, size=(100, 3)).astype(float)
    C = rng.normal(size=(100, 2))
    y = rng.normal(size=100)
    with pytest.raises(ValueError, match="duplicate"):
        derive_matrix_sufficient_stats(
            "s1", "exposure", snp_ids, ea, oa, G, C, y, covariate_names=("PC1", "PC1"),
        )


def test_mismatched_role_rejected():
    rng = np.random.default_rng(0)
    snp_ids, effect_allele, other_allele, maf = _make_snp_panel(rng, 3)
    G1, C1, y1 = _simulate_site(rng, 100, 3, maf, 1, 0.4, np.array([0.1]), 1.0)
    G2, C2, y2 = _simulate_site(rng, 100, 3, maf, 1, 0.4, np.array([0.1]), 1.0)

    site1 = derive_matrix_sufficient_stats(
        "s1", "exposure", snp_ids, effect_allele, other_allele, G1, C1, y1, covariate_names=("PC1",),
    )
    site2 = derive_matrix_sufficient_stats(
        "s2", "outcome", snp_ids, effect_allele, other_allele, G2, C2, y2, covariate_names=("PC1",),
    )
    with pytest.raises(ValueError, match="mixed roles"):
        aggregate_matrix_sites([site1, site2])