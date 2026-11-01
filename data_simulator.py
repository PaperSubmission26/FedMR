"""
data_simulator.py

Generates fake individual-level people, split into separate exposure-sites
and outcome-sites (Design B: two-sample summary-data MR). Each site then
derives its own local sufficient statistics from its simulated
individuals — this derivation step is what preprocess_real_data.py will
later mirror for real data.

Generative model per SNP j, per individual i:
    G_ij ~ Binomial(2, maf_j)                      genotype dosage (effect-allele count)
    X_i  ~ Normal(COVARIATE_MEAN, COVARIATE_SD)     one covariate

    Exposure-site individuals:
        E_i = sum_j beta_gx_j * G_ij
              + COVARIATE_EFFECT_ON_EXPOSURE * X_i
              + eps_x,   eps_x ~ N(0, EXPOSURE_NOISE_SD)

    Outcome-site individuals (different people, two-sample MR):
        Y_i = beta_causal_site * sum_j beta_gx_j * G_ij      (mediated path)
              + sum_j alpha_j * G_ij                          (pleiotropy)
              + COVARIATE_EFFECT_ON_OUTCOME * X_i
              + eps_y,   eps_y ~ N(0, OUTCOME_NOISE_SD)

    beta_causal_site = TRUE_CAUSAL_EFFECT + site-specific perturbation
                        (SITE_HETEROGENEITY_SD), for outcome sites only —
                        this is what site-level heterogeneity diagnostics
                        will later try to detect. The realized value is
                        stored per site (site_realized_causal_effect) since
                        it is a fact about the simulation, distinct from
                        TRUE_CAUSAL_EFFECT (the population ground truth)
                        and from whatever a centralized/federated MR run
                        later estimates (a computational reference, not
                        a ground truth in its own right).

Sufficient statistics derived per site, per SNP (10 raw sums):
    n, sum_G, sum_X, sum_G2, sum_GX, sum_X2, sum_Y, sum_GY, sum_XY, sum_Y2

These are exactly the entries needed to reconstruct the OLS regression
    trait ~ intercept + G + X
via its normal equations (X'X and X'Y, plus Y'Y for the residual variance/SE),
so summing them across sites reproduces the centralized regression exactly
(Phase 2 correctness proof relies on this — but note run_centralized_baseline.py
deliberately uses a *different* solver, np.linalg.lstsq, rather than the same
normal-equations code path, so a shared implementation bug can't pass unnoticed).
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import config


@dataclass
class SiteSufficientStats:
    """Per-site, per-SNP sufficient statistics for OLS(trait ~ G + X + 1)."""
    site_name: str
    site_role: str            # "exposure" or "outcome"
    n_snps: int
    snp_ids: np.ndarray        # (n_snps,) str — stable SNP identifiers, e.g. "rs1000"
    effect_allele: np.ndarray  # (n_snps,) str — allele whose dosage G counts
    other_allele: np.ndarray   # (n_snps,) str
    # each numeric array below has shape (n_snps,)
    n: np.ndarray
    sum_G: np.ndarray
    sum_X: np.ndarray
    sum_G2: np.ndarray
    sum_GX: np.ndarray
    sum_X2: np.ndarray
    sum_Y: np.ndarray
    sum_GY: np.ndarray
    sum_XY: np.ndarray
    sum_Y2: np.ndarray                        # sum of trait^2 — needed for residual variance / SE
    site_realized_causal_effect: Optional[float] = None  # outcome sites only, when heterogeneity is on

    def validate(self) -> None:
        """Basic structural checks — catches SNP-ordering and NaN/Inf bugs early."""
        arrays = [self.n, self.sum_G, self.sum_X, self.sum_G2, self.sum_GX,
                  self.sum_X2, self.sum_Y, self.sum_GY, self.sum_XY, self.sum_Y2]
        for a in arrays:
            if a.shape != (self.n_snps,):
                raise ValueError(
                    f"{self.site_name}: sufficient-stat array has shape {a.shape}, "
                    f"expected ({self.n_snps},)"
                )
            if not np.all(np.isfinite(a)):
                raise ValueError(f"{self.site_name}: non-finite value in sufficient statistics")
        if len(self.snp_ids) != self.n_snps or len(set(self.snp_ids)) != self.n_snps:
            raise ValueError(f"{self.site_name}: snp_ids missing, duplicated, or wrong length")
        if self.site_role not in ("exposure", "outcome"):
            raise ValueError(f"{self.site_name}: invalid site_role {self.site_role!r}")
        if np.any(self.n <= 3):
            raise ValueError(f"{self.site_name}: fewer than 4 individuals — regression underdetermined")


@dataclass
class SNPPanel:
    """The shared true genetic architecture underlying both sample sets."""
    snp_ids: np.ndarray              # (n_snps,) str
    effect_allele: np.ndarray        # (n_snps,) str, e.g. "A"
    other_allele: np.ndarray         # (n_snps,) str, e.g. "G"
    maf: np.ndarray                  # (n_snps,) effect-allele frequency
    beta_gx: np.ndarray              # (n_snps,) true SNP -> exposure effect
    is_weak_instrument: np.ndarray   # (n_snps,) bool
    alpha_pleiotropy: np.ndarray     # (n_snps,) true direct SNP -> outcome effect
    is_pleiotropic: np.ndarray       # (n_snps,) bool


_BASES = ["A", "C", "G", "T"]


def build_snp_panel() -> SNPPanel:
    rng = config.get_rng("snp_panel")
    n = config.N_SNPS

    snp_ids = np.array([f"rs{1000 + i}" for i in range(n)])

    effect_allele = np.empty(n, dtype=object)
    other_allele = np.empty(n, dtype=object)
    for i in range(n):
        pair = rng.choice(_BASES, size=2, replace=False)
        effect_allele[i], other_allele[i] = pair[0], pair[1]

    maf = rng.uniform(config.MAF_RANGE[0], config.MAF_RANGE[1], size=n)

    beta_gx = rng.normal(config.BETA_GX_MEAN, config.BETA_GX_SD, size=n)

    n_weak = int(round(config.WEAK_INSTRUMENT_FRACTION * n))
    weak_idx = rng.choice(n, size=n_weak, replace=False)
    is_weak = np.zeros(n, dtype=bool)
    is_weak[weak_idx] = True
    beta_gx = np.where(is_weak, beta_gx * config.WEAK_INSTRUMENT_SCALE, beta_gx)

    n_pleio = int(round(config.PLEIOTROPY_FRACTION * n))
    pleio_idx = rng.choice(n, size=n_pleio, replace=False)
    is_pleio = np.zeros(n, dtype=bool)
    is_pleio[pleio_idx] = True
    alpha = np.zeros(n)
    alpha[pleio_idx] = rng.normal(
        config.PLEIOTROPY_MEAN, config.PLEIOTROPY_SD, size=n_pleio
    )

    return SNPPanel(
        snp_ids=snp_ids,
        effect_allele=effect_allele,
        other_allele=other_allele,
        maf=maf,
        beta_gx=beta_gx,
        is_weak_instrument=is_weak,
        alpha_pleiotropy=alpha,
        is_pleiotropic=is_pleio,
    )


def _simulate_genotypes(rng: np.random.Generator, n_people: int, maf: np.ndarray) -> np.ndarray:
    """Returns (n_people, n_snps) dosage matrix, independent SNPs."""
    n_snps = maf.shape[0]
    G = rng.binomial(2, maf, size=(n_people, n_snps)).astype(float)
    return G


def _derive_sufficient_stats(
    site_name: str,
    site_role: str,
    panel: SNPPanel,
    G: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    site_realized_causal_effect: Optional[float] = None,
) -> SiteSufficientStats:
    """
    G: (n_people, n_snps), X: (n_people,), Y: (n_people,)
    Computes the 10 raw sums per SNP, vectorized across SNPs.
    """
    n_people, n_snps = G.shape
    n_vec = np.full(n_snps, n_people, dtype=float)

    sum_G = G.sum(axis=0)
    sum_G2 = (G ** 2).sum(axis=0)
    sum_GX = (G * X[:, None]).sum(axis=0)
    sum_GY = (G * Y[:, None]).sum(axis=0)

    sum_X = np.full(n_snps, X.sum())
    sum_X2 = np.full(n_snps, (X ** 2).sum())
    sum_Y = np.full(n_snps, Y.sum())
    sum_XY = np.full(n_snps, (X * Y).sum())
    sum_Y2 = np.full(n_snps, (Y ** 2).sum())

    stats = SiteSufficientStats(
        site_name=site_name,
        site_role=site_role,
        n_snps=n_snps,
        snp_ids=panel.snp_ids,
        effect_allele=panel.effect_allele,
        other_allele=panel.other_allele,
        n=n_vec,
        sum_G=sum_G,
        sum_X=sum_X,
        sum_G2=sum_G2,
        sum_GX=sum_GX,
        sum_X2=sum_X2,
        sum_Y=sum_Y,
        sum_GY=sum_GY,
        sum_XY=sum_XY,
        sum_Y2=sum_Y2,
        site_realized_causal_effect=site_realized_causal_effect,
    )
    stats.validate()
    return stats


def _simulate_exposure_individuals(site_index: int, n_people: int, panel: SNPPanel) -> dict:
    site_name = f"exposure_site_{site_index}"
    rng = config.get_rng(site_name)

    G = _simulate_genotypes(rng, n_people, panel.maf)
    X = rng.normal(config.COVARIATE_MEAN, config.COVARIATE_SD, size=n_people)
    eps = rng.normal(0.0, config.EXPOSURE_NOISE_SD, size=n_people)

    E = G @ panel.beta_gx + config.COVARIATE_EFFECT_ON_EXPOSURE * X + eps

    return {"site_name": site_name, "G": G, "X": X, "trait": E, "realized_causal_effect": None}


def _simulate_outcome_individuals(site_index: int, n_people: int, panel: SNPPanel) -> dict:
    site_name = f"outcome_site_{site_index}"
    rng = config.get_rng(site_name)

    G = _simulate_genotypes(rng, n_people, panel.maf)
    X = rng.normal(config.COVARIATE_MEAN, config.COVARIATE_SD, size=n_people)
    eps = rng.normal(0.0, config.OUTCOME_NOISE_SD, size=n_people)

    site_perturbation = rng.normal(0.0, config.SITE_HETEROGENEITY_SD)
    beta_causal_site = config.TRUE_CAUSAL_EFFECT + site_perturbation

    mediated = beta_causal_site * (G @ panel.beta_gx)
    pleiotropic = G @ panel.alpha_pleiotropy

    Y = mediated + pleiotropic + config.COVARIATE_EFFECT_ON_OUTCOME * X + eps

    return {
        "site_name": site_name, "G": G, "X": X, "trait": Y,
        "realized_causal_effect": beta_causal_site,
    }


def simulate_exposure_site(site_index: int, n_people: int, panel: SNPPanel) -> SiteSufficientStats:
    d = _simulate_exposure_individuals(site_index, n_people, panel)
    return _derive_sufficient_stats(d["site_name"], "exposure", panel, d["G"], d["X"], d["trait"])


def simulate_outcome_site(site_index: int, n_people: int, panel: SNPPanel) -> SiteSufficientStats:
    d = _simulate_outcome_individuals(site_index, n_people, panel)
    return _derive_sufficient_stats(
        d["site_name"], "outcome", panel, d["G"], d["X"], d["trait"],
        site_realized_causal_effect=d["realized_causal_effect"],
    )


def simulate_all_sites(panel: SNPPanel = None):
    """
    Returns (panel, exposure_site_stats_list, outcome_site_stats_list) —
    only per-site sufficient statistics, never raw individuals. This is
    the entry point the "normal" federated path uses.
    """
    if panel is None:
        panel = build_snp_panel()

    exposure_sites = [
        simulate_exposure_site(i, n, panel)
        for i, n in enumerate(config.EXPOSURE_SITE_SIZES)
    ]
    outcome_sites = [
        simulate_outcome_site(i, n, panel)
        for i, n in enumerate(config.OUTCOME_SITE_SIZES)
    ]

    return panel, exposure_sites, outcome_sites


def simulate_all_sites_with_individuals(panel: SNPPanel = None):
    """
    Same as simulate_all_sites, but also returns the raw individual-level
    arrays per site (G, X, trait, realized_causal_effect) so the
    centralized baseline can pool them directly. Calls the exact same
    per-site simulation helpers as simulate_all_sites() (no duplicated
    generative code), so the two entry points cannot silently drift apart.
    """
    if panel is None:
        panel = build_snp_panel()

    exposure_individuals = [
        _simulate_exposure_individuals(i, n, panel)
        for i, n in enumerate(config.EXPOSURE_SITE_SIZES)
    ]
    outcome_individuals = [
        _simulate_outcome_individuals(i, n, panel)
        for i, n in enumerate(config.OUTCOME_SITE_SIZES)
    ]

    exposure_site_stats = [
        _derive_sufficient_stats(d["site_name"], "exposure", panel, d["G"], d["X"], d["trait"])
        for d in exposure_individuals
    ]
    outcome_site_stats = [
        _derive_sufficient_stats(
            d["site_name"], "outcome", panel, d["G"], d["X"], d["trait"],
            site_realized_causal_effect=d["realized_causal_effect"],
        )
        for d in outcome_individuals
    ]

    return panel, exposure_site_stats, outcome_site_stats, exposure_individuals, outcome_individuals


if __name__ == "__main__":
    panel, exp_sites, out_sites = simulate_all_sites()
    print(f"Simulated {len(exp_sites)} exposure sites, {len(out_sites)} outcome sites, "
          f"{config.N_SNPS} SNPs.")
    print(f"Weak instruments: {panel.is_weak_instrument.sum()}, "
          f"Pleiotropic SNPs: {panel.is_pleiotropic.sum()}")
    for s in exp_sites:
        print(f"  {s.site_name}: n={int(s.n[0])}, sum_G[0]={s.sum_G[0]:.1f}")
    for s in out_sites:
        print(f"  {s.site_name}: n={int(s.n[0])}, realized_causal_effect="
              f"{s.site_realized_causal_effect}")
