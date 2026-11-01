"""
mr_estimators.py

Permanent "textbook" MR summary-data estimators. These are pure functions
of already-aggregated global per-SNP associations (beta_exposure,
se_exposure, beta_outcome, se_outcome) — they know nothing about sites,
federation, or privacy. That separation is deliberate: federated_aggregator.py
produces the global per-SNP inputs; everything below is ordinary,
centralized MR math applied to those inputs.

Each estimator returns an MRResult(estimate, se, p_value, extra).
"""

from dataclasses import dataclass, field
import numpy as np
from scipy import stats


@dataclass
class MRResult:
    method: str
    estimate: float
    se: float
    p_value: float
    n_snps: int
    extra: dict = field(default_factory=dict)


def _ratio_estimates(beta_exp: np.ndarray, beta_out: np.ndarray) -> np.ndarray:
    """Per-SNP Wald ratio: Gamma_j / gamma_j."""
    return beta_out / beta_exp


def _p_value_from_z(z: float) -> float:
    """
    2 * norm.sf(|z|) instead of 2 * (1 - norm.cdf(|z|)): norm.sf computes
    the survival function directly and stays numerically accurate for
    large |z| (tiny p-values), where 1 - cdf(z) loses precision to
    floating-point cancellation and can round to exactly 0.
    """
    return 2 * stats.norm.sf(abs(z))


def _validate_mr_inputs(
    beta_exp: np.ndarray, se_exp: np.ndarray, beta_out: np.ndarray, se_out: np.ndarray,
    method: str, min_snps: int, require_nonzero_beta_exp: bool,
) -> None:
    """
    Shared validation for all MR estimator entry points. Raises a clear,
    specific ValueError rather than letting bad input silently propagate
    into NaNs, infs, or a divide-by-zero RuntimeWarning deep in the math.
    """
    arrays = {"beta_exp": beta_exp, "se_exp": se_exp, "beta_out": beta_out, "se_out": se_out}
    for name, arr in arrays.items():
        arr = np.asarray(arr)
        if arr.ndim != 1:
            raise ValueError(f"{method}: {name} must be a 1-D array, got shape {arr.shape}.")

    n = len(np.asarray(beta_exp))
    for name, arr in arrays.items():
        arr = np.asarray(arr)
        if len(arr) != n:
            raise ValueError(
                f"{method}: array length mismatch — beta_exp has {n} elements but "
                f"{name} has {len(arr)}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{method}: {name} contains NaN or infinite values.")

    se_exp_arr = np.asarray(se_exp)
    se_out_arr = np.asarray(se_out)
    if np.any(se_exp_arr <= 0):
        raise ValueError(f"{method}: se_exp must be strictly positive (found a value <= 0).")
    if np.any(se_out_arr <= 0):
        raise ValueError(f"{method}: se_out must be strictly positive (found a value <= 0).")

    if require_nonzero_beta_exp and np.any(np.asarray(beta_exp) == 0):
        raise ValueError(
            f"{method}: beta_exp contains an exact 0 — ratio-based MR estimators divide "
            "by beta_exp per SNP, so a zero exposure coefficient is invalid input "
            "(exclude that SNP upstream instead)."
        )

    if n < min_snps:
        raise ValueError(
            f"{method}: requires at least {min_snps} SNP(s), got {n}."
        )


def ivw(
    beta_exp: np.ndarray, se_exp: np.ndarray, beta_out: np.ndarray, se_out: np.ndarray
) -> MRResult:
    """
    Inverse-variance-weighted estimator, random-effects (multiplicative
    dispersion correction allowed).

    Weights here are w_j = 1 / se_out_j^2 — these are FIRST-ORDER weights
    under the NOME (NO Measurement Error in the exposure) assumption:
    they ignore uncertainty in beta_exp entirely. This is the standard/
    default IVW weighting in two-sample MR and is what this function
    implements. "Second-order weights" (which additionally incorporate
    se_exp, e.g. via a delta-method correction to the weight itself) are
    a distinct, stricter refinement this function does NOT implement —
    labeling these as second-order would be incorrect.
    """
    _validate_mr_inputs(beta_exp, se_exp, beta_out, se_out, method="IVW",
                        min_snps=1, require_nonzero_beta_exp=False)

    w = 1.0 / (se_out ** 2)
    num = np.sum(w * beta_exp * beta_out)
    den = np.sum(w * beta_exp ** 2)
    beta_ivw = num / den

    resid = beta_out - beta_ivw * beta_exp
    rss = np.sum(w * resid ** 2)
    n_snps = len(beta_exp)
    dof = n_snps - 1
    phi = max(rss / dof, 1.0) if dof > 0 else 1.0  # dispersion factor, floored at 1 (fixed-effect)

    se_ivw = np.sqrt(phi / den)
    z = beta_ivw / se_ivw
    p = _p_value_from_z(z)

    return MRResult(
        method="IVW",
        estimate=beta_ivw,
        se=se_ivw,
        p_value=p,
        n_snps=n_snps,
        extra={"dispersion_phi": phi, "cochrans_Q": rss, "Q_df": dof,
               "weights": "first-order (NOME): w_j = 1/se_out_j^2"},
    )


def mr_egger(
    beta_exp: np.ndarray, se_exp: np.ndarray, beta_out: np.ndarray, se_out: np.ndarray
) -> MRResult:
    """
    MR-Egger regression: beta_out = intercept + slope * beta_exp,
    weighted by 1/se_out^2, WITH an intercept term (allows an average
    directional pleiotropic effect; the intercept itself is the test
    for directional pleiotropy — see extra['intercept']/['intercept_p']).

    Allele orientation matters for MR-Egger specifically (unlike IVW and
    weighted-median, whose ratios/weights are orientation-invariant): the
    intercept test is only valid when every SNP's alleles are coded so
    that beta_exp is positive (see Bowden et al. 2015). Before regressing,
    any SNP with a negative beta_exp has BOTH beta_exp and the
    corresponding beta_out sign-flipped (equivalent to swapping which
    allele is "effect" vs "other" for that SNP) — flipping only one of
    the pair would silently corrupt the ratio.
    """
    _validate_mr_inputs(beta_exp, se_exp, beta_out, se_out, method="MR-Egger",
                        min_snps=3, require_nonzero_beta_exp=False)

    sign = np.where(beta_exp < 0, -1.0, 1.0)
    beta_exp_oriented = beta_exp * sign
    beta_out_oriented = beta_out * sign
    # se_exp/se_out are unaffected by sign flips (SEs are always non-negative)

    w = 1.0 / (se_out ** 2)
    n_snps = len(beta_exp)

    Xd = np.column_stack([np.ones(n_snps), beta_exp_oriented])
    W = np.diag(w)
    XtWX = Xd.T @ W @ Xd
    XtWY = Xd.T @ W @ beta_out_oriented
    XtWX_inv = np.linalg.inv(XtWX)
    coef = XtWX_inv @ XtWY
    intercept, slope = coef

    resid = beta_out_oriented - Xd @ coef
    rss = np.sum(w * resid ** 2)
    dof = n_snps - 2
    phi = max(rss / dof, 1.0) if dof > 0 else 1.0

    cov = phi * XtWX_inv
    se_intercept = np.sqrt(cov[0, 0])
    se_slope = np.sqrt(cov[1, 1])

    z_slope = slope / se_slope
    p_slope = _p_value_from_z(z_slope)

    z_intercept = intercept / se_intercept
    p_intercept = _p_value_from_z(z_intercept)

    return MRResult(
        method="MR-Egger",
        estimate=slope,
        se=se_slope,
        p_value=p_slope,
        n_snps=n_snps,
        extra={
            "intercept": intercept,
            "intercept_se": se_intercept,
            "intercept_p": p_intercept,
            "dispersion_phi": phi,
            "n_snps_flipped": int(np.sum(sign < 0)),
        },
    )


def weighted_median(
    beta_exp: np.ndarray,
    se_exp: np.ndarray,
    beta_out: np.ndarray,
    se_out: np.ndarray,
    n_bootstrap: int = 10000,
    rng: np.random.Generator = None,
) -> MRResult:
    """
    Simple weighted-median estimator over per-SNP Wald ratios.
    As noted in the problem statement: this needs the full set of
    global per-SNP ratio estimates + weights, but since the pipeline
    already produces those as an aggregation intermediate, computing
    the median here is ordinary centralized math on already-aggregated
    numbers, not a federated sorting problem.

    SE obtained via parametric bootstrap: redraw BOTH beta_exp_j ~
    N(beta_exp_j, se_exp_j^2) and beta_out_j ~ N(beta_out_j, se_out_j^2)
    independently per replicate (not just beta_out — the exposure
    associations carry their own sampling uncertainty and holding them
    fixed would understate the true SE), recompute the weighted median,
    and take the SD across replicates. Weights are held fixed at their
    point-estimate values across replicates (standard simplification).
    """
    _validate_mr_inputs(beta_exp, se_exp, beta_out, se_out, method="WeightedMedian",
                        min_snps=3, require_nonzero_beta_exp=True)

    if rng is None:
        rng = np.random.default_rng(0)

    ratios = _ratio_estimates(beta_exp, beta_out)
    weights = (beta_exp ** 2) / (se_out ** 2)  # inverse-variance of the ratio (delta method, NOME)

    point = _weighted_median_value(ratios, weights)

    boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        beta_exp_b = rng.normal(beta_exp, se_exp)
        beta_out_b = rng.normal(beta_out, se_out)
        ratios_b = _ratio_estimates(beta_exp_b, beta_out_b)
        boot[b] = _weighted_median_value(ratios_b, weights)

    se_boot = float(np.std(boot, ddof=1))
    z = point / se_boot if se_boot > 0 else np.nan
    p = _p_value_from_z(z) if se_boot > 0 else np.nan

    return MRResult(
        method="WeightedMedian",
        estimate=point,
        se=se_boot,
        p_value=p,
        n_snps=len(beta_exp),
        extra={"n_bootstrap": n_bootstrap},
    )


def _weighted_median_value(values: np.ndarray, weights: np.ndarray) -> float:
    """Standard weighted-median: sort by value, find where cumulative weight crosses 50%."""
    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    cum_w = np.cumsum(w_sorted) - 0.5 * w_sorted
    cum_w /= np.sum(w_sorted)
    idx = np.searchsorted(cum_w, 0.5)
    if idx == 0:
        return v_sorted[0]
    if idx >= len(v_sorted):
        return v_sorted[-1]
    lo, hi = cum_w[idx - 1], cum_w[idx]
    frac = (0.5 - lo) / (hi - lo) if hi > lo else 0.0
    return v_sorted[idx - 1] + frac * (v_sorted[idx] - v_sorted[idx - 1])


if __name__ == "__main__":
    from data_simulator import simulate_all_sites
    from federated_aggregator import aggregate_sites_with_se

    panel, exp_sites, out_sites = simulate_all_sites()
    ge = aggregate_sites_with_se(exp_sites)
    go = aggregate_sites_with_se(out_sites)

    for fn in (ivw, mr_egger, weighted_median):
        res = fn(ge.beta, ge.se, go.beta, go.se)
        print(f"{res.method:15s} estimate={res.estimate:.4f}  se={res.se:.4f}  p={res.p_value:.4g}")
