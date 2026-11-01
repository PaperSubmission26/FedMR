"""
heterogeneity_check.py

Two DISTINCT heterogeneity diagnostics, per the Phase 1 correction that
they must not be conflated:

1. SNP-LEVEL heterogeneity (Cochran's Q on the IVW ratio estimates):
   "do individual SNPs disagree with each other more than sampling
   variation alone would predict, given the pooled causal-effect
   estimate?" Surfaces mr_estimators.ivw()'s own Q statistic (which is
   already correctly centered on the IVW slope — IVW's own fit) as an
   explicit diagnostic with a p-value.

2. SITE-LEVEL heterogeneity: "do different SITES disagree with each
   other more than sampling variation alone would predict, given a
   properly combined per-SNP estimate?" Computed as a standard
   fixed-effect meta-analysis Cochran's Q across sites' own per-SNP
   associations. CRITICAL: this Q must be centered on the
   INVERSE-VARIANCE-WEIGHTED fixed-effect estimate
       beta_FE = sum_i(w_i * beta_i) / sum_i(w_i),  w_i = 1/se_i^2
   computed from the per-site estimates themselves — NOT on the pooled
   OLS estimate from aggregate_sites_with_se(sites). Those two are
   close but NOT identical (pooled OLS re-derives the estimate from
   summed raw sufficient statistics across individuals, a different
   estimator than combining each site's own already-fitted estimate by
   inverse-variance weighting) — centering on the wrong one breaks the
   identity sum_i(w_i*(beta_i - beta_FE)) = 0 that Cochran's Q relies on
   to be properly chi-square-distributed under the null, inflating Q and
   the false-positive rate. See test_diagnostics.py's
   test_site_heterogeneity_uses_fixed_effect_center for the regression
   test on this.
"""

from dataclasses import dataclass
import numpy as np
from scipy import stats

from data_simulator import SiteSufficientStats
from federated_aggregator import GlobalAssociation, aggregate_sites_with_se, _validate_sites_compatible
from mr_estimators import MRResult, ivw
from sensitivity_analysis import _validate_exposure_outcome_pair as _validate_exposure_outcome_pair_local

SNP_HETEROGENEITY_ALPHA = 0.05
SITE_HETEROGENEITY_ALPHA = 0.05
# NOTE: this fraction is an OPERATIONAL WARNING THRESHOLD, not a formal
# statistical test — it has no claimed type-I error rate of its own.
# The per-SNP p-values (raw and BH-FDR-adjusted) ARE the formal tests;
# this fraction is just a practical "how much of the genome looks
# affected" summary on top of them.
SITE_HETEROGENEITY_WARN_FRAC = 0.20


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Standard BH step-up FDR adjustment. Returns adjusted p-values,
    same shape/order as input, monotone-enforced (each adjusted p is
    >= all smaller-rank adjusted p's, per the standard procedure)."""
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest p-value down
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty(n)
    out[order] = adjusted
    return out


@dataclass
class SNPHeterogeneityReport:
    Q: float
    df: int
    p_value: float
    status: str  # "PASS", "FAIL", or "NOT_APPLICABLE"
    alpha: float


@dataclass
class SiteHeterogeneityReport:
    role: str
    n_sites: int
    n_snps: int
    snp_ids: np.ndarray
    beta_FE: np.ndarray            # (n_snps,) the inverse-variance fixed-effect estimate used as Q's center
    Q_per_snp: np.ndarray
    df_per_snp: int
    p_per_snp: np.ndarray
    p_per_snp_fdr: np.ndarray      # Benjamini-Hochberg adjusted, same order as snp_ids
    n_significant_raw: int
    n_significant_fdr: int
    frac_significant_raw: float
    alpha: float
    warn_frac_threshold: float
    status: str                    # "PASS", "WARN", "FAIL" (any BH-FDR-significant SNP), or "NOT_APPLICABLE"


def _validate_snp_arrays(alpha: float) -> None:
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def check_snp_level_heterogeneity(
    global_exposure: GlobalAssociation, global_outcome: GlobalAssociation,
    alpha: float = SNP_HETEROGENEITY_ALPHA,
    max_iter: int = 50, tol: float = 1e-10,
) -> SNPHeterogeneityReport:
    """
    Cochran's Q on the IVW ratio estimates, using MODIFIED ITERATIVE
    WEIGHTS (NOT the textbook "second-order weights" term, which refers
    to a specific one-shot delta-method correction — this iterates that
    correction to convergence, which is a related but distinct
    calculation, so it gets its own name rather than borrowing a term
    that implies something slightly different):
        w_j = 1 / (se_out_j^2 + beta_ivw^2 * se_exp_j^2)
    reweighted repeatedly using the current beta_ivw estimate, rather
    than the FIRST-ORDER weights (w_j = 1/se_out_j^2) that
    mr_estimators.ivw()'s own MRResult.extra['cochrans_Q'] uses (that Q
    is intentionally left as first-order there, matching the standard/
    default IVW convention documented in mr_estimators.py — this
    function does NOT surface that Q, it computes its own, corrected
    one).

    ITERATED TO CONVERGENCE (not a single plug-in step): starts from the
    first-order beta_ivw, then repeatedly reweights using the current
    beta_ivw estimate and refits, until beta_ivw changes by less than
    `tol` between iterations (or max_iter is reached). This is standard
    iteratively-reweighted least squares.

    WHY THIS MATTERS, CONCRETELY: first-order weights ignore sampling
    uncertainty in beta_exp entirely. Empirically, under a clean null
    simulation (no pleiotropy, no weak instruments, no site
    heterogeneity — Cochran's Q SHOULD reject at ~alpha), first-order
    weights gave a ~23% false-positive rate at alpha=0.05 in this
    project's default config — a 4-5x inflation. These modified
    iterative weights bring the empirical false-positive rate down to
    roughly 5-10% (see test_diagnostics.py's calibration test) — still
    not a perfect 5% (regression dilution in a ratio estimator is a
    known, imperfectly-correctable phenomenon), but a substantial,
    verified improvement.

    Requires an EXPOSURE/OUTCOME pair (validated the same way
    sensitivity_analysis.py's functions are). Returns
    status="NOT_APPLICABLE" (not a misleading FAIL) when:
      - df<=0 (fewer than 2 SNPs), or
      - every beta_exp is exactly 0 (the first-order fit divides by
        sum(w*beta_exp^2), which is 0 in this degenerate case — there is
        no instrument signal at all to test heterogeneity ON, so this is
        correctly "not applicable", not a failed test with a NaN Q that
        would otherwise be misreported as FAIL since NaN >= alpha is
        False in Python).
    """
    _validate_snp_arrays(alpha)
    _validate_exposure_outcome_pair_local(global_exposure, global_outcome)
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError(f"max_iter must be a positive integer, got {max_iter}")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError(f"tol must be finite and > 0, got {tol}")

    beta_exp, se_exp = global_exposure.beta, global_exposure.se
    beta_out, se_out = global_outcome.beta, global_outcome.se
    n_snps = global_exposure.n_snps
    df = n_snps - 1

    if df <= 0:
        return SNPHeterogeneityReport(Q=float("nan"), df=df, p_value=float("nan"),
                                       status="NOT_APPLICABLE", alpha=alpha)

    if np.all(beta_exp == 0):
        return SNPHeterogeneityReport(Q=float("nan"), df=df, p_value=float("nan"),
                                       status="NOT_APPLICABLE", alpha=alpha)

    # start from the first-order fit
    w = 1.0 / (se_out ** 2)
    beta_ivw = np.sum(w * beta_exp * beta_out) / np.sum(w * beta_exp ** 2)

    for _ in range(max_iter):
        w_new = 1.0 / (se_out ** 2 + (beta_ivw ** 2) * (se_exp ** 2))
        beta_ivw_new = np.sum(w_new * beta_exp * beta_out) / np.sum(w_new * beta_exp ** 2)
        if abs(beta_ivw_new - beta_ivw) < tol:
            beta_ivw = beta_ivw_new
            w = w_new
            break
        beta_ivw = beta_ivw_new
        w = w_new

    Q = np.sum(w * (beta_out - beta_ivw * beta_exp) ** 2)

    p = float(stats.chi2.sf(Q, df))
    return SNPHeterogeneityReport(Q=float(Q), df=df, p_value=p,
                                   status=("PASS" if p >= alpha else "FAIL"), alpha=alpha)


def check_site_level_heterogeneity(
    sites: list[SiteSufficientStats],
    alpha: float = SITE_HETEROGENEITY_ALPHA,
    warn_frac_threshold: float = SITE_HETEROGENEITY_WARN_FRAC,
) -> SiteHeterogeneityReport:
    """
    For each SNP: gets each site's own per-SNP association, computes the
    inverse-variance-weighted fixed-effect estimate beta_FE across sites,
    and runs Cochran's Q centered on beta_FE (NOT the pooled OLS estimate
    — see module docstring). Requires at least 2 sites.
    """
    _validate_snp_arrays(alpha)
    if not (0 < warn_frac_threshold <= 1):
        raise ValueError(f"warn_frac_threshold must be in (0, 1], got {warn_frac_threshold}")
    if len(sites) < 2:
        raise ValueError(
            f"check_site_level_heterogeneity requires at least 2 sites to compare "
            f"(got {len(sites)}) — site-level heterogeneity is undefined for one site."
        )
    # Regression guard: the fixed-effect centering fix computes per-site
    # estimates via aggregate_sites_with_se([site]) singleton calls, which
    # each trivially pass compatibility checks (nothing to compare a lone
    # site against) — so the cross-site checks (same role, same SNP
    # count/IDs/order, same alleles) must be run explicitly here, not
    # assumed to happen as a side effect of some other call.
    _validate_sites_compatible(sites)

    role = sites[0].site_role
    n_snps = sites[0].n_snps
    snp_ids = sites[0].snp_ids
    n_sites = len(sites)

    per_site_beta = np.empty((n_sites, n_snps))
    per_site_se = np.empty((n_sites, n_snps))
    for i, site in enumerate(sites):
        local = aggregate_sites_with_se([site])
        if not np.all(np.isfinite(local.beta)) or not np.all(np.isfinite(local.se)):
            raise ValueError(f"{site.site_name}: non-finite local association — cannot compute Q.")
        if np.any(local.se <= 0):
            raise ValueError(f"{site.site_name}: non-positive local SE — cannot compute Q.")
        per_site_beta[i, :] = local.beta
        per_site_se[i, :] = local.se

    weights = 1.0 / (per_site_se ** 2)
    # THE FIX: center on the inverse-variance-weighted fixed-effect
    # estimate derived from the per-site betas themselves, not on
    # aggregate_sites_with_se(sites)'s pooled-OLS beta — see module
    # docstring for why these are not interchangeable.
    beta_FE = np.sum(weights * per_site_beta, axis=0) / np.sum(weights, axis=0)

    Q_per_snp = np.sum(weights * (per_site_beta - beta_FE[np.newaxis, :]) ** 2, axis=0)
    df = n_sites - 1

    if df <= 0:
        return SiteHeterogeneityReport(
            role=role, n_sites=n_sites, n_snps=n_snps, snp_ids=snp_ids, beta_FE=beta_FE,
            Q_per_snp=Q_per_snp, df_per_snp=df,
            p_per_snp=np.full(n_snps, np.nan), p_per_snp_fdr=np.full(n_snps, np.nan),
            n_significant_raw=0, n_significant_fdr=0, frac_significant_raw=0.0,
            alpha=alpha, warn_frac_threshold=warn_frac_threshold, status="NOT_APPLICABLE",
        )

    p_per_snp = stats.chi2.sf(Q_per_snp, df)
    p_per_snp_fdr = _benjamini_hochberg(p_per_snp)

    n_sig_raw = int(np.sum(p_per_snp < alpha))
    n_sig_fdr = int(np.sum(p_per_snp_fdr < alpha))
    frac_sig_raw = n_sig_raw / n_snps

    # THE FIX: any FDR-significant SNP is a FORMAL statistical finding
    # (BH-FDR correction already accounts for multiple testing across
    # SNPs) and must not be reported as a clean PASS, regardless of how
    # small a fraction of SNPs that represents. The raw-p 20% rule
    # remains what it always was — an operational WARN threshold with no
    # claimed type-I error rate of its own — layered UNDER the formal
    # FDR-based FAIL, not instead of it.
    if n_sig_fdr > 0:
        status = "FAIL"
    elif frac_sig_raw >= warn_frac_threshold:
        status = "WARN"
    else:
        status = "PASS"

    return SiteHeterogeneityReport(
        role=role, n_sites=n_sites, n_snps=n_snps, snp_ids=snp_ids, beta_FE=beta_FE,
        Q_per_snp=Q_per_snp, df_per_snp=df, p_per_snp=p_per_snp, p_per_snp_fdr=p_per_snp_fdr,
        n_significant_raw=n_sig_raw, n_significant_fdr=n_sig_fdr, frac_significant_raw=frac_sig_raw,
        alpha=alpha, warn_frac_threshold=warn_frac_threshold, status=status,
    )


def print_snp_report(report: SNPHeterogeneityReport) -> None:
    if report.status == "NOT_APPLICABLE":
        print(f"[N/A] SNP-level heterogeneity: undefined with df={report.df} "
              f"(need at least 2 SNPs)")
        return
    print(f"[{report.status}] SNP-level heterogeneity (Cochran's Q={report.Q:.2f}, "
          f"df={report.df}, p={report.p_value:.4g}, alpha={report.alpha})")
    if report.status == "FAIL":
        print("  Significant SNP-level heterogeneity detected — individual SNPs disagree "
              "more than sampling error predicts (possible pleiotropy).")


@dataclass
class SiteCausalEffectHeterogeneityReport:
    n_sites: int
    site_names: list
    site_causal_estimates: np.ndarray   # (n_sites,) each outcome site's own IVW causal estimate
    site_causal_ses: np.ndarray         # (n_sites,) each site's own SE for that estimate
    beta_FE: float                       # inverse-variance-weighted mean across sites' causal estimates
    Q: float
    df: int
    p_value: float
    status: str  # "PASS", "FAIL", or "NOT_APPLICABLE" (<2 sites, or IVW failed for any site)
    alpha: float
    is_approximate: bool = True  # see check_site_causal_effect_heterogeneity docstring: this
                                  # Q treats per-site estimates as independent, but they all
                                  # share the same global_exposure, so they are NOT truly
                                  # independent — exploratory diagnostic, not a formal test


def check_site_causal_effect_heterogeneity(
    global_exposure: GlobalAssociation, outcome_sites: list[SiteSufficientStats],
    alpha: float = SITE_HETEROGENEITY_ALPHA,
) -> SiteCausalEffectHeterogeneityReport:
    """
    A DIFFERENT diagnostic from check_site_level_heterogeneity, testing a
    DIFFERENT question: check_site_level_heterogeneity asks whether
    sites disagree on the raw per-SNP SNP->exposure (or SNP->outcome)
    ASSOCIATION. This function instead asks whether outcome sites
    disagree on the derived CAUSAL EFFECT ESTIMATE itself — the thing
    the whole pipeline is ultimately trying to estimate, and the thing
    data_simulator.py's SITE_HETEROGENEITY_SD mechanism actually
    perturbs per outcome site (site_realized_causal_effect). A pipeline
    could show no significant per-SNP association heterogeneity while
    still masking real causal-effect heterogeneity across sites (or vice
    versa) — these are genuinely different failure modes, not
    duplicates of each other.

    For each outcome site, computes an IVW causal estimate using the
    SAME shared global_exposure (from all exposure sites pooled — Design
    B's two-sample structure means individual outcome sites never have
    their own exposure data) paired with THAT SITE's own local per-SNP
    outcome association, then runs a standard fixed-effect Cochran's Q
    across the resulting per-site causal estimates.

    IMPORTANT CAVEAT — THIS Q IS APPROXIMATE, NOT A FORMALLY VALID TEST:
    Cochran's Q assumes the per-site estimates are INDEPENDENT. Here,
    every site's causal estimate is computed from the SAME
    global_exposure array — so the sampling error in beta_exp/se_exp is
    SHARED (correlated) across every "independent" site estimate, not
    actually independent at all. This means the chi-square reference
    distribution used for the p-value is not exactly correct (it could
    be either too liberal or too conservative depending on how that
    shared component enters the ratio for each site — not analyzed
    here). Properly accounting for this would need a joint/multivariate
    correction (e.g. treating the shared exposure uncertainty as a
    common random effect and computing a covariance-aware test), which
    is NOT implemented. Treat this diagnostic as EXPLORATORY — useful
    for spotting an obviously large site-specific causal-effect
    difference (as in the regression test for this function), not as a
    calibrated hypothesis test with a trustworthy type-I error rate.
    is_approximate=True on every report is a permanent reminder of this,
    not a placeholder to be later flipped to False without actually
    fixing the underlying independence violation.
    """
    _validate_snp_arrays(alpha)
    if global_exposure.role != "exposure":
        raise ValueError(
            f"check_site_causal_effect_heterogeneity requires an EXPOSURE association "
            f"(got role={global_exposure.role!r})."
        )
    if len(outcome_sites) < 2:
        raise ValueError(
            f"check_site_causal_effect_heterogeneity requires at least 2 outcome sites "
            f"(got {len(outcome_sites)})."
        )
    for s in outcome_sites:
        s.validate()
        if s.site_role != "outcome":
            raise ValueError(f"{s.site_name}: expected an outcome site, got role={s.site_role!r}")
    # Outcome sites must all agree on SNP count/IDs/order/alleles with each
    # OTHER (same check federated_aggregator.aggregate_sites_with_se uses) —
    # otherwise per-site local outcome associations wouldn't even be
    # comparable to each other before we get to comparing them to the
    # exposure side at all.
    _validate_sites_compatible(outcome_sites)

    site_estimates = np.empty(len(outcome_sites))
    site_ses = np.empty(len(outcome_sites))
    site_names = [s.site_name for s in outcome_sites]

    for i, site in enumerate(outcome_sites):
        local_outcome = aggregate_sites_with_se([site])
        # THE FIX: validate this site's local outcome association against
        # the SHARED global_exposure before pairing them elementwise in
        # ivw() — without this, a site with different SNP order or allele
        # coding than global_exposure would silently corrupt every
        # per-site causal estimate (ivw() itself does no such check, it
        # just does elementwise array arithmetic).
        _validate_exposure_outcome_pair_local(global_exposure, local_outcome)
        res = ivw(global_exposure.beta, global_exposure.se, local_outcome.beta, local_outcome.se)
        site_estimates[i] = res.estimate
        site_ses[i] = res.se

    if not np.all(np.isfinite(site_ses)) or np.any(site_ses <= 0):
        return SiteCausalEffectHeterogeneityReport(
            n_sites=len(outcome_sites), site_names=site_names,
            site_causal_estimates=site_estimates, site_causal_ses=site_ses,
            beta_FE=float("nan"), Q=float("nan"), df=0, p_value=float("nan"),
            status="NOT_APPLICABLE", alpha=alpha,
        )

    w = 1.0 / (site_ses ** 2)
    beta_FE = float(np.sum(w * site_estimates) / np.sum(w))
    Q = float(np.sum(w * (site_estimates - beta_FE) ** 2))
    df = len(outcome_sites) - 1
    p = float(stats.chi2.sf(Q, df))

    return SiteCausalEffectHeterogeneityReport(
        n_sites=len(outcome_sites), site_names=site_names,
        site_causal_estimates=site_estimates, site_causal_ses=site_ses,
        beta_FE=beta_FE, Q=Q, df=df, p_value=p,
        status=("PASS" if p >= alpha else "FAIL"), alpha=alpha,
    )


def print_site_causal_report(report: SiteCausalEffectHeterogeneityReport) -> None:
    if report.status == "NOT_APPLICABLE":
        print("[N/A] Site-specific causal-effect heterogeneity: could not compute "
              "(non-finite or non-positive SE for at least one site)")
        return
    approx_tag = " (APPROXIMATE/exploratory -- see docstring)" if report.is_approximate else ""
    print(f"[{report.status}]{approx_tag} Site-specific causal-effect heterogeneity "
          f"({report.n_sites} outcome sites): Q={report.Q:.2f}, df={report.df}, "
          f"p={report.p_value:.4g}, alpha={report.alpha}")
    if report.is_approximate:
        print("  NOTE: per-site estimates share the same global exposure association, so they "
              "are not truly independent -- this Q's p-value is not formally calibrated.")
    if report.status == "FAIL":
        print(f"  Significant heterogeneity in the CAUSAL ESTIMATE itself across sites (not just "
              f"the raw associations) — per-site estimates: "
              f"{dict(zip(report.site_names, np.round(report.site_causal_estimates, 4)))}")



def print_site_report(report: SiteHeterogeneityReport) -> None:
    if report.status == "NOT_APPLICABLE":
        print(f"[N/A] Site-level heterogeneity ({report.role}): undefined with "
              f"df={report.df_per_snp} (need at least 2 sites)")
        return
    print(f"[{report.status}] Site-level heterogeneity ({report.role}, {report.n_sites} sites): "
          f"{report.n_significant_raw}/{report.n_snps} SNPs significant at raw p<{report.alpha} "
          f"({report.frac_significant_raw:.1%}); {report.n_significant_fdr}/{report.n_snps} "
          f"significant after BH-FDR correction")
    if report.status == "FAIL":
        print(f"  FAIL: at least one SNP is significant after BH-FDR correction — a formal, "
              f"multiple-testing-corrected finding of site-level heterogeneity, not just the "
              f"operational raw-p threshold.")
    elif report.status == "WARN":
        print(f"  >= {report.warn_frac_threshold:.0%} of SNPs significant at raw p<alpha — this "
              f"is an OPERATIONAL warning threshold, not a formal statistical test; check the "
              f"FDR-adjusted count above for the more conservative read.")


if __name__ == "__main__":
    from data_simulator import simulate_all_sites

    panel, exposure_sites, outcome_sites = simulate_all_sites()
    global_exposure = aggregate_sites_with_se(exposure_sites)
    global_outcome = aggregate_sites_with_se(outcome_sites)

    snp_report = check_snp_level_heterogeneity(global_exposure, global_outcome)
    print_snp_report(snp_report)

    exp_site_report = check_site_level_heterogeneity(exposure_sites)
    print_site_report(exp_site_report)

    out_site_report = check_site_level_heterogeneity(outcome_sites)
    print_site_report(out_site_report)

    causal_report = check_site_causal_effect_heterogeneity(global_exposure, outcome_sites)
    print_site_causal_report(causal_report)
