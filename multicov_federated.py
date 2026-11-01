"""
multicov_federated.py

Generalizes data_simulator.py's SiteSufficientStats (fixed design:
trait ~ 1 + G + X, exactly 10 scalar sums) to an arbitrary number of
adjustment covariates, per the roadmap's Milestone 3 / Section 7.1
("Generalized local statistics"). Site-level statistic derivation and the coordinator's aggregation logic
live together in this one file — they are small, always imported
together, and never used independently.

For each site s and SNP j, the local design matrix is
    Z_{s,j} = [1, G_j, C_1, ..., C_p]        (n_people_s x d), d = 2 + p
and the site releases, PER SNP:
    A_{s,j} = Z_{s,j}' Z_{s,j}      (d x d)
    b_{s,j} = Z_{s,j}' y_s          (d,)
    q_{s,j} = y_s' y_s              (scalar; identical across j since y_s
                                      doesn't depend on the SNP, but kept
                                      per-SNP for a uniform release shape)
    n_{s,j} = n_people_s            (scalar; public, as in the original design)

This is the direct matrix generalization of the old 10-sum design: the
old (n, sum_G, sum_X, sum_G2, sum_GX, sum_X2, sum_Y, sum_GY, sum_XY,
sum_Y2) are exactly the entries of A (3x3, d=3, p=1) and b (3,) and q,
just unpacked into named scalars. Summing A/b/q across sites and
solving A^-1 b reproduces pooled OLS by the same additivity argument
used in federated_aggregator.py's docstring — this file does not
change that argument, only the dimensionality it's applied at.

Covariates C_1..C_p are assumed identical in definition across every
site (e.g. same genotype-PC columns, same coding for sex) — this is a
precondition of the real MAGE experiment (Section 7.2: "Centralized and
federated comparisons must use exactly the same covariate matrix"), and
is checked by aggregate_matrix_sites' cross-site validation.

COORDINATOR LOGIC: for each SNP j, sums each site's (A_j, b_j, q_j, n_j)
— exactly additive across disjoint samples for the same reason the 3x3
case is (A = Z'Z and b = Z'y are themselves sums over individuals, and
summing sums over disjoint individual sets is trivially the sum over
the union) — then solves the normal equations A beta = b once per SNP:

    beta    = A^-1 b
    sigma^2 = (q - beta' b) / (n - d)          d = 2 + p parameters
    Cov(beta) = sigma^2 * A^-1
    se_G    = sqrt(Cov(beta)[1, 1])             index 1 = genotype coefficient

Coefficient ordering is fixed by construction:
    beta[0] = intercept
    beta[1] = genotype (the quantity federated MR actually needs)
    beta[2:] = covariates, in covariate_names order

Before summing, cross-site compatibility is checked: same role, same
SNP count/IDs/order, same alleles (as in the 1-covariate aggregator),
PLUS same covariate_names in the same order and same design dimension
d — silently summing A matrices built from different covariate
definitions would produce coefficients that don't mean the same thing
across sites and there would be no way to detect it downstream.
"""

from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np


# ============================== site-level ==============================


@dataclass
class SiteMatrixStats:
    """Per-site, per-SNP matrix sufficient statistics for OLS(y ~ 1 + G + C_1..C_p)."""
    site_name: str
    site_role: str                  # "exposure" or "outcome" (kept for parity with the 1-covariate design)
    n_snps: int
    covariate_names: tuple          # length p, in the same order as columns of A/b
    snp_ids: np.ndarray             # (n_snps,) str
    effect_allele: np.ndarray       # (n_snps,) str
    other_allele: np.ndarray        # (n_snps,) str
    n: np.ndarray                   # (n_snps,) — public, constant across SNPs at a given site
    A: np.ndarray                   # (n_snps, d, d) — d = 2 + p
    b: np.ndarray                   # (n_snps, d)
    q: np.ndarray                   # (n_snps,) — sum(y^2), constant across SNPs
    site_realized_causal_effect: Optional[float] = None

    @property
    def d(self) -> int:
        """Design dimension: intercept + genotype + p covariates."""
        return 2 + len(self.covariate_names)

    def validate(self) -> None:
        """
        Structural checks, mirroring data_simulator.SiteSufficientStats.validate():
        catches SNP-ordering / shape / NaN-Inf bugs before any cross-site sum.
        """
        d = self.d
        if self.A.shape != (self.n_snps, d, d):
            raise ValueError(
                f"{self.site_name}: A has shape {self.A.shape}, expected "
                f"({self.n_snps}, {d}, {d}) for {len(self.covariate_names)} covariate(s)."
            )
        if self.b.shape != (self.n_snps, d):
            raise ValueError(
                f"{self.site_name}: b has shape {self.b.shape}, expected ({self.n_snps}, {d})."
            )
        for name, arr in (("n", self.n), ("q", self.q)):
            if arr.shape != (self.n_snps,):
                raise ValueError(f"{self.site_name}: {name} has shape {arr.shape}, expected ({self.n_snps},)")

        if not np.all(np.isfinite(self.A)) or not np.all(np.isfinite(self.b)) \
           or not np.all(np.isfinite(self.q)) or not np.all(np.isfinite(self.n)):
            raise ValueError(f"{self.site_name}: non-finite value in matrix sufficient statistics.")

        # A_j must be symmetric (it's Z_j'Z_j by construction) — a real
        # asymmetry here means the release itself was corrupted upstream
        # (e.g. mismatched Z columns), not a numerical-noise-scale issue.
        asym = np.abs(self.A - np.transpose(self.A, (0, 2, 1)))
        if np.max(asym) > 1e-6 * max(1.0, np.max(np.abs(self.A))):
            raise ValueError(f"{self.site_name}: A is not symmetric — Z'Z should always be symmetric.")

        if len(self.snp_ids) != self.n_snps or len(set(self.snp_ids)) != self.n_snps:
            raise ValueError(f"{self.site_name}: snp_ids missing, duplicated, or wrong length.")
        for meta_name, meta_arr in (("effect_allele", self.effect_allele), ("other_allele", self.other_allele)):
            if np.asarray(meta_arr).shape != (self.n_snps,):
                raise ValueError(
                    f"{self.site_name}: {meta_name} has shape {np.asarray(meta_arr).shape}, "
                    f"expected exactly ({self.n_snps},) — e.g. a (n_snps, 1) array is rejected "
                    "too, not just a wrong length, since it would silently broadcast wrong "
                    "in later elementwise allele comparisons."
                )
        if len(set(self.covariate_names)) != len(self.covariate_names):
            raise ValueError(
                f"{self.site_name}: covariate_names contains duplicate(s): {self.covariate_names} "
                "— every covariate must have a unique name (a repeated name would make the "
                "corresponding column of A/b ambiguous)."
            )
        if self.site_role not in ("exposure", "outcome"):
            raise ValueError(f"{self.site_name}: invalid site_role {self.site_role!r}.")
        if np.any(self.n <= d):
            raise ValueError(
                f"{self.site_name}: fewer than {d + 1} individuals for a design with "
                f"{d} parameters — regression underdetermined."
            )


def derive_matrix_sufficient_stats(
    site_name: str,
    site_role: str,
    snp_ids: np.ndarray,
    effect_allele: np.ndarray,
    other_allele: np.ndarray,
    G: np.ndarray,
    C: np.ndarray,
    y: np.ndarray,
    covariate_names: Sequence[str],
    site_realized_causal_effect: Optional[float] = None,
) -> SiteMatrixStats:
    """
    G: (n_people, n_snps) genotype dosage matrix.
    C: (n_people, p) covariate matrix (p may be 0 -> pass shape (n_people, 0)).
    y: (n_people,) phenotype/trait vector.
    covariate_names: length-p sequence naming C's columns, in order —
        carried through so downstream code/reports can label coefficients,
        and so aggregate_matrix_sites can verify every site used the same
        covariate definition before summing (see module docstring).

    Loops per SNP (mirrors data_simulator.py's style): only the genotype
    column of Z changes across SNPs, so re-forming Z per SNP is the
    simplest way to guarantee A_j always equals the *actual* Z_j'Z_j
    used, rather than trying to hand-vectorize the cross-products and
    risking a subtle indexing bug that would silently miscompute results
    for the real MAGE run.
    """
    G = np.asarray(G, dtype=float)
    C = np.asarray(C, dtype=float)
    y = np.asarray(y, dtype=float)
    covariate_names = tuple(covariate_names)

    n_people, n_snps = G.shape
    if C.ndim != 2 or C.shape[0] != n_people:
        raise ValueError(f"C must have shape (n_people={n_people}, p), got {C.shape}")
    p = C.shape[1]
    if len(covariate_names) != p:
        raise ValueError(
            f"covariate_names has {len(covariate_names)} entries but C has {p} columns."
        )
    if y.shape != (n_people,):
        raise ValueError(f"y must have shape ({n_people},), got {y.shape}")

    if not np.all(np.isfinite(G)):
        raise ValueError(f"{site_name}: G contains non-finite (NaN or Inf) value(s).")
    if p > 0 and not np.all(np.isfinite(C)):
        raise ValueError(f"{site_name}: C contains non-finite (NaN or Inf) value(s).")
    if not np.all(np.isfinite(y)):
        raise ValueError(f"{site_name}: y contains non-finite (NaN or Inf) value(s).")

    d = 2 + p
    ones = np.ones(n_people)

    A = np.empty((n_snps, d, d))
    b = np.empty((n_snps, d))
    q_scalar = float(y @ y)
    q = np.full(n_snps, q_scalar)
    n_vec = np.full(n_snps, n_people, dtype=float)

    for j in range(n_snps):
        Z = np.column_stack([ones, G[:, j], C]) if p > 0 else np.column_stack([ones, G[:, j]])
        A[j] = Z.T @ Z
        b[j] = Z.T @ y

    stats = SiteMatrixStats(
        site_name=site_name, site_role=site_role, n_snps=n_snps,
        covariate_names=covariate_names, snp_ids=np.asarray(snp_ids),
        effect_allele=np.asarray(effect_allele), other_allele=np.asarray(other_allele),
        n=n_vec, A=A, b=b, q=q,
        site_realized_causal_effect=site_realized_causal_effect,
    )
    stats.validate()
    return stats


# ============================== coordinator ==============================


@dataclass
class GlobalMatrixAssociation:
    """Global per-SNP association from the generalized (multi-covariate) design."""
    role: str
    n_snps: int
    covariate_names: tuple
    snp_ids: np.ndarray
    effect_allele: np.ndarray
    other_allele: np.ndarray
    beta_G: np.ndarray          # (n_snps,) genotype coefficient — what MR consumes
    se_G: np.ndarray            # (n_snps,)
    beta_full: np.ndarray       # (n_snps, d) full coefficient vector, for reference/debugging
    n_total: np.ndarray         # (n_snps,)
    residual_variance: np.ndarray  # (n_snps,) sigma^2, for reference/debugging


def _sum_field(sites: list[SiteMatrixStats], field_name: str) -> np.ndarray:
    return np.sum([getattr(s, field_name) for s in sites], axis=0)


def _validate_matrix_sites_compatible(sites: list[SiteMatrixStats]) -> None:
    if len(sites) == 0:
        raise ValueError("aggregate_matrix_sites: got an empty site list — nothing to aggregate.")

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
            raise ValueError(f"{s.site_name}: n_snps={s.n_snps} != {ref.site_name}'s n_snps={ref.n_snps}.")
        if not np.array_equal(s.snp_ids, ref.snp_ids):
            raise ValueError(f"{s.site_name}: snp_ids do not match {ref.site_name}'s (different SNPs or order).")
        if not np.array_equal(s.effect_allele, ref.effect_allele) or \
           not np.array_equal(s.other_allele, ref.other_allele):
            raise ValueError(
                f"{s.site_name}: allele coding does not match {ref.site_name}'s — "
                "genotype dosages are not comparable without harmonization first."
            )
        if s.covariate_names != ref.covariate_names:
            raise ValueError(
                f"{s.site_name}: covariate_names={s.covariate_names} does not match "
                f"{ref.site_name}'s covariate_names={ref.covariate_names} — every site "
                "must release sufficient statistics built from the identical covariate "
                "definition (Section 7.2: centralized and federated models must use "
                "exactly the same covariate matrix)."
            )


def aggregate_matrix_sites(sites: list[SiteMatrixStats]) -> GlobalMatrixAssociation:
    _validate_matrix_sites_compatible(sites)

    role = sites[0].site_role
    n_snps = sites[0].n_snps
    covariate_names = sites[0].covariate_names
    snp_ids = sites[0].snp_ids
    effect_allele = sites[0].effect_allele
    other_allele = sites[0].other_allele
    d = sites[0].d

    n = _sum_field(sites, "n")
    A = _sum_field(sites, "A")     # (n_snps, d, d)
    b = _sum_field(sites, "b")     # (n_snps, d)
    q = _sum_field(sites, "q")     # (n_snps,)

    beta_full = np.empty((n_snps, d))
    se_G = np.empty(n_snps)
    sigma2_arr = np.empty(n_snps)

    for j in range(n_snps):
        Aj, bj, qj, nj = A[j], b[j], q[j], n[j]

        cond = np.linalg.cond(Aj)
        if not np.isfinite(cond) or cond > 1e12:
            raise np.linalg.LinAlgError(
                f"{role} SNP {snp_ids[j]} (index {j}): A is singular or near-singular "
                f"(cond={cond:.3g}) — likely a monomorphic SNP, collinear covariates, or "
                "a covariate perfectly correlated with genotype."
            )
        A_inv_col1 = np.linalg.solve(Aj, np.eye(d)[:, 1])  # column 1 of A^-1, i.e. Cov(beta) scale factor for genotype
        coef = np.linalg.solve(Aj, bj)

        rss = qj - coef @ bj
        dof = nj - d
        if dof <= 0:
            raise ValueError(
                f"{role} SNP {snp_ids[j]} (index {j}): non-positive degrees of freedom "
                f"(n={nj}, d={d}) — not enough pooled individuals for this many parameters."
            )
        sigma2 = rss / dof
        if sigma2 <= 0:
            raise ValueError(
                f"{role} SNP {snp_ids[j]} (index {j}): residual variance estimate is "
                f"non-positive (sigma^2={sigma2:.4g}). Under DP noise this signals the "
                "privacy budget was too tight to identify this SNP; without DP it "
                "indicates a data or normal-equations problem upstream."
            )
        var_bG = sigma2 * A_inv_col1[1]
        if not np.isfinite(var_bG) or var_bG <= 0:
            raise ValueError(
                f"{role} SNP {snp_ids[j]} (index {j}): genotype coefficient variance is "
                f"non-finite or non-positive (var_bG={var_bG:.4g}) even though sigma^2 "
                f"({sigma2:.4g}) was positive — A^-1's [1,1] entry is behaving abnormally "
                "(DP noise, or a near-collinear covariate)."
            )

        beta_full[j] = coef
        se_G[j] = np.sqrt(var_bG)
        sigma2_arr[j] = sigma2

    return GlobalMatrixAssociation(
        role=role, n_snps=n_snps, covariate_names=covariate_names,
        snp_ids=snp_ids, effect_allele=effect_allele, other_allele=other_allele,
        beta_G=beta_full[:, 1], se_G=se_G, beta_full=beta_full,
        n_total=n, residual_variance=sigma2_arr,
    )