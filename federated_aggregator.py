"""
federated_aggregator.py

Central "coordinator" logic: combines each site's local sufficient
statistics (the 10 raw sums per SNP from data_simulator.py) into
global per-SNP regression coefficients — WITHOUT ever seeing any
site's raw per-individual data.

For each SNP j, every site solves nothing locally beyond the raw sums;
all the actual linear algebra happens here, on already-summed totals.
This is what makes the Phase 2 correctness proof possible: because
OLS normal equations are additive across disjoint samples,

    sum_over_sites(X_site'X_site) == X_pooled'X_pooled
    sum_over_sites(X_site'Y_site) == X_pooled'Y_pooled

the global per-SNP regression
    trait ~ 1 + G_j + X
computed from summed sufficient statistics is numerically IDENTICAL
(not just "close") to the same regression run on centrally pooled
individuals — see run_centralized_baseline.py and
test_federated_matches_centralized.py.

Before any statistics are summed, every site's own validate() is called
and cross-site compatibility is checked (same role, same SNP count/IDs/
order, same effect/other alleles). Summing sufficient statistics across
sites that disagree on any of this would silently produce nonsense
(e.g. SNP j's exposure sum added to a different SNP j's outcome sum, or
an effect-allele mismatch silently flipping the sign of a genotype
count) — so this must be checked before aggregation, not caught later.
"""

from dataclasses import dataclass
import numpy as np

from data_simulator import SiteSufficientStats


@dataclass
class GlobalAssociation:
    """Global per-SNP association (genotype -> trait), one array per field."""
    role: str                # "exposure" or "outcome"
    n_snps: int
    snp_ids: np.ndarray        # (n_snps,) str — carried through from the sites, for downstream MR
    effect_allele: np.ndarray  # (n_snps,) str
    other_allele: np.ndarray   # (n_snps,) str
    beta: np.ndarray          # (n_snps,) estimated G -> trait effect, adjusted for X
    se: np.ndarray            # (n_snps,) standard error of beta
    n_total: np.ndarray       # (n_snps,) total individuals contributing (should be constant)


def _sum_stat_field(sites: list[SiteSufficientStats], field_name: str) -> np.ndarray:
    """Sums one sufficient-statistic field across all sites (elementwise over SNPs)."""
    return np.sum([getattr(s, field_name) for s in sites], axis=0)


def _validate_sites_compatible(sites: list[SiteSufficientStats]) -> None:
    """
    Cross-site checks required before any statistic is summed:
      - non-empty list
      - every site individually passes its own validate()
      - every site has the same role ("exposure" or "outcome")
      - every site reports the same SNP count, in the same order, with
        the same SNP IDs and the same effect/other allele coding
    """
    if len(sites) == 0:
        raise ValueError("aggregate_sites_with_se: got an empty site list — nothing to aggregate.")

    for s in sites:
        s.validate()

    roles = {s.site_role for s in sites}
    if len(roles) > 1:
        raise ValueError(
            f"Cannot aggregate sites with mixed roles: {sorted(roles)}. "
            "Exposure and outcome sites must be aggregated separately."
        )

    ref = sites[0]
    for s in sites[1:]:
        if s.n_snps != ref.n_snps:
            raise ValueError(
                f"{s.site_name}: n_snps={s.n_snps} does not match "
                f"{ref.site_name}'s n_snps={ref.n_snps}."
            )
        if not np.array_equal(s.snp_ids, ref.snp_ids):
            raise ValueError(
                f"{s.site_name}: snp_ids do not match {ref.site_name}'s snp_ids "
                "(different SNPs, or same SNPs in a different order)."
            )
        if not np.array_equal(s.effect_allele, ref.effect_allele) or \
           not np.array_equal(s.other_allele, ref.other_allele):
            raise ValueError(
                f"{s.site_name}: effect/other allele coding does not match "
                f"{ref.site_name}'s — genotype dosages are not comparable across "
                "these sites without allele harmonization first."
            )


def aggregate_sites_with_se(sites: list[SiteSufficientStats]) -> GlobalAssociation:
    """
    Combines a list of same-role site sufficient stats into one global
    per-SNP OLS regression: trait ~ 1 + G_j + X, solved via the 3x3
    normal equations, vectorized across SNPs. Uses sum_Y2 (sum of
    trait^2 per SNP, an additive sufficient statistic reported locally
    by each site, same as the other nine — no raw individual data is
    touched here) to get the residual variance and hence the SE.

    residual variance: sigma^2 = RSS / (n - 3)
    var(bG) = sigma^2 * [(X'X)^-1]_{GG}
    """
    _validate_sites_compatible(sites)

    role = sites[0].site_role
    n_snps = sites[0].n_snps
    snp_ids = sites[0].snp_ids
    effect_allele = sites[0].effect_allele
    other_allele = sites[0].other_allele

    n = _sum_stat_field(sites, "n")
    sum_G = _sum_stat_field(sites, "sum_G")
    sum_X = _sum_stat_field(sites, "sum_X")
    sum_G2 = _sum_stat_field(sites, "sum_G2")
    sum_GX = _sum_stat_field(sites, "sum_GX")
    sum_X2 = _sum_stat_field(sites, "sum_X2")
    sum_Y = _sum_stat_field(sites, "sum_Y")
    sum_GY = _sum_stat_field(sites, "sum_GY")
    sum_XY = _sum_stat_field(sites, "sum_XY")
    sum_trait2 = _sum_stat_field(sites, "sum_Y2")

    beta = np.empty(n_snps)
    se = np.empty(n_snps)

    for j in range(n_snps):
        XtX = np.array([
            [n[j],       sum_G[j],   sum_X[j]],
            [sum_G[j],   sum_G2[j],  sum_GX[j]],
            [sum_X[j],   sum_GX[j],  sum_X2[j]],
        ])
        XtY = np.array([sum_Y[j], sum_GY[j], sum_XY[j]])

        cond = np.linalg.cond(XtX)
        if not np.isfinite(cond) or cond > 1e12:
            raise np.linalg.LinAlgError(
                f"{role} SNP {snp_ids[j]} (index {j}): X'X is singular or near-singular "
                f"(cond={cond:.3g}) — likely a monomorphic SNP or zero-variance covariate."
            )
        XtX_inv = np.linalg.inv(XtX)
        coef = XtX_inv @ XtY
        b0, bG, bX = coef

        rss = sum_trait2[j] - coef @ XtY
        dof = n[j] - 3
        sigma2 = rss / dof
        if sigma2 <= 0:
            raise ValueError(
                f"{role} SNP {snp_ids[j]} (index {j}): residual variance estimate is "
                f"non-positive (sigma^2={sigma2:.4g}). Under DP noise this means the privacy "
                f"budget was too tight to identify this SNP's association at all (increase "
                f"epsilon, reduce the number of privatized fields, or exclude this SNP). "
                f"Without DP, this indicates a data or normal-equations problem upstream."
            )
        var_bG = sigma2 * XtX_inv[1, 1]
        if not np.isfinite(var_bG) or var_bG <= 0:
            raise ValueError(
                f"{role} SNP {snp_ids[j]} (index {j}): coefficient variance estimate is "
                f"non-finite or non-positive (var_bG={var_bG:.4g}). sigma^2 was positive "
                f"({sigma2:.4g}), so this means DP noise pushed (X'X)^-1's [1,1] entry "
                f"negative even though X'X itself passed the condition-number check above — "
                f"the noised design matrix is no longer behaving like a valid covariance "
                f"structure for this SNP. As with the sigma^2 check, this means the privacy "
                f"budget was too tight to identify this SNP (increase epsilon, reduce the "
                f"number of privatized fields, or exclude this SNP); without DP, it indicates "
                f"a data or normal-equations problem upstream."
            )

        beta[j] = bG
        se[j] = np.sqrt(var_bG)

    return GlobalAssociation(
        role=role, n_snps=n_snps, snp_ids=snp_ids,
        effect_allele=effect_allele, other_allele=other_allele,
        beta=beta, se=se, n_total=n,
    )


if __name__ == "__main__":
    from data_simulator import simulate_all_sites

    panel, exp_sites, out_sites = simulate_all_sites()

    global_exposure = aggregate_sites_with_se(exp_sites)
    global_outcome = aggregate_sites_with_se(out_sites)

    print("Global exposure associations (first 5 SNPs):")
    for j in range(5):
        print(f"  {global_exposure.snp_ids[j]}: beta={global_exposure.beta[j]:.4f}  "
              f"se={global_exposure.se[j]:.4f}")

    print("Global outcome associations (first 5 SNPs):")
    for j in range(5):
        print(f"  {global_outcome.snp_ids[j]}: beta={global_outcome.beta[j]:.4f}  "
              f"se={global_outcome.se[j]:.4f}")
