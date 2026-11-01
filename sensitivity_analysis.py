"""
sensitivity_analysis.py

Additional validity diagnostics beyond the core instrument-strength and
heterogeneity checks, all federated-safe (operate only on already-
aggregated global per-SNP associations, same as the other Phase 4
modules):

  - MR-Egger intercept test: already computed inside mr_egger()'s
    result.extra, surfaced here as an explicit pass/fail diagnostic
    (a significant intercept signals average directional pleiotropy).
  - Leave-one-SNP-out IVW: re-runs IVW once per SNP with that SNP
    excluded; flags SNPs whose removal shifts the estimate by more than
    a configurable fraction of the full-sample SE ("influential SNPs").
  - IVW after excluding weak instruments: the sensitivity analysis
    instrument_strength_check.py's WARN status points to — re-runs IVW
    using only the SNPs that clear the F-statistic threshold.
  - Cross-estimator comparison: IVW vs MR-Egger vs weighted-median side
    by side, flagging if MR-Egger's slope disagrees with IVW/median by
    more than their combined SE would suggest (a simple, non-formal
    "do the estimators agree" check — this is descriptive, not a formal
    hypothesis test).
  - I^2_GX: Bowden et al.'s measure of how much of the variability in the
    SNP -> exposure estimates is real signal vs. sampling noise, used to
    judge whether MR-Egger's NOME assumption (no measurement error in
    the exposure) is reasonable. Low I^2_GX (<0.9, the commonly used
    rule of thumb) means MR-Egger's own slope/CI may be unreliable
    (regression dilution) regardless of what its p-value says.
"""

from dataclasses import dataclass
import numpy as np
from scipy import stats

from federated_aggregator import GlobalAssociation
from mr_estimators import ivw, mr_egger, weighted_median, MRResult
from instrument_strength_check import (
    check_instrument_strength, WEAK_INSTRUMENT_F_THRESHOLD,
    _validate_exposure_association, _validate_global_association_structure,
)

EGGER_INTERCEPT_ALPHA = 0.05
INFLUENTIAL_SNP_SE_FRACTION = 0.5  # a SNP is "influential" if removing it shifts the
                                    # estimate by more than this fraction of the full SE
I2_GX_LOW_THRESHOLD = 0.90         # Bowden et al.'s commonly used rule of thumb


def _validate_numeric_param(name: str, value: float, low, high, low_inclusive=False, high_inclusive=False) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if low_inclusive:
        if value < low:
            raise ValueError(f"{name} must be >= {low}, got {value}")
    else:
        if value <= low:
            raise ValueError(f"{name} must be > {low}, got {value}")
    if high is not None:
        if high_inclusive:
            if value > high:
                raise ValueError(f"{name} must be <= {high}, got {value}")
        else:
            if value >= high:
                raise ValueError(f"{name} must be < {high}, got {value}")


def _validate_exposure_outcome_pair(global_exposure: GlobalAssociation, global_outcome: GlobalAssociation) -> None:
    """
    Every function in this module that combines an exposure and an
    outcome GlobalAssociation elementwise (beta_exp[j] paired with
    beta_out[j]) requires them to actually correspond to the SAME SNPs
    in the SAME order with the SAME allele coding — otherwise SNP j's
    exposure effect could silently get paired with a different SNP's
    outcome effect, or with flipped allele coding, corrupting every
    downstream ratio/regression without any error being raised.
    """
    if global_exposure.role != "exposure":
        raise ValueError(f"Expected an exposure association, got role={global_exposure.role!r}")
    if global_outcome.role != "outcome":
        raise ValueError(f"Expected an outcome association, got role={global_outcome.role!r}")
    _validate_global_association_structure(global_exposure)
    _validate_global_association_structure(global_outcome)
    if global_exposure.n_snps != global_outcome.n_snps:
        raise ValueError(
            f"Exposure ({global_exposure.n_snps} SNPs) and outcome ({global_outcome.n_snps} SNPs) "
            f"associations have different SNP counts — cannot pair them elementwise."
        )
    if not np.array_equal(global_exposure.snp_ids, global_outcome.snp_ids):
        raise ValueError(
            "Exposure and outcome associations have different SNP IDs or SNP order — "
            "cannot pair them elementwise without risking mismatched SNPs."
        )
    if not np.array_equal(global_exposure.effect_allele, global_outcome.effect_allele) or \
       not np.array_equal(global_exposure.other_allele, global_outcome.other_allele):
        raise ValueError(
            "Exposure and outcome associations have mismatched effect/other allele coding — "
            "genotype dosages are not comparable without allele harmonization first."
        )
    # Data-quality checks — these must raise HERE, not be left for
    # mr_egger()/weighted_median() to discover downstream, because
    # callers that wrap those estimators in a broad try/except for
    # legitimate numerical failures (singular design matrix, etc. — see
    # sensitivity_analysis.safe_egger_intercept_check / compare_estimators)
    # would otherwise silently absorb a genuine data-quality bug (NaN/Inf
    # in the input, non-positive SE) as an innocuous NOT_APPLICABLE
    # "numerical_failure", instead of surfacing it as the loud, distinct
    # error it actually is.
    for name, ge in (("exposure", global_exposure), ("outcome", global_outcome)):
        if not np.all(np.isfinite(ge.beta)) or not np.all(np.isfinite(ge.se)):
            raise ValueError(f"{name} association contains NaN or infinite beta/se values.")
        if np.any(ge.se <= 0):
            raise ValueError(f"{name} association has non-positive se values.")


@dataclass
class EggerInterceptReport:
    intercept: float
    intercept_se: float
    p_value: float
    status: str  # "PASS", "FAIL", or "NOT_APPLICABLE"
    alpha: float
    reason: str = ""              # set when NOT_APPLICABLE: "too_few_snps" or "numerical_failure"
    low_i2gx_caveat: bool = False  # True if I^2_GX is low -- does NOT change status, see below


def check_egger_intercept(egger_result: MRResult, alpha: float = EGGER_INTERCEPT_ALPHA) -> EggerInterceptReport:
    if egger_result.method != "MR-Egger":
        raise ValueError(f"check_egger_intercept requires an MR-Egger MRResult, got {egger_result.method!r}")
    _validate_numeric_param("alpha", alpha, 0, 1)
    p = egger_result.extra["intercept_p"]
    return EggerInterceptReport(
        intercept=egger_result.extra["intercept"],
        intercept_se=egger_result.extra["intercept_se"],
        p_value=p,
        status=("PASS" if p >= alpha else "FAIL"),
        alpha=alpha,
    )


def safe_egger_intercept_check(
    global_exposure: GlobalAssociation, global_outcome: GlobalAssociation,
    alpha: float = EGGER_INTERCEPT_ALPHA, i2_gx_threshold: float = I2_GX_LOW_THRESHOLD,
) -> EggerInterceptReport:
    """
    Wraps check_egger_intercept so callers never need to call mr_egger()
    themselves first and risk it crashing the whole diagnostics run.
    Guards against TWO distinct failure modes, not just one:
      - too few SNPs (<3): checked up front, cheap, explicit reason
      - numerical failure even with >=3 SNPs: MR-Egger's design matrix
        [1, beta_exp] is singular whenever beta_exp has near-zero
        variance across SNPs (e.g. all SNPs happen to have very similar
        effect sizes) — this raises np.linalg.LinAlgError from inside
        mr_estimators.py regardless of how many SNPs there are, so it
        must be caught separately from the <3-SNPs precondition check.

    ALSO flags (via low_i2gx_caveat, NOT by changing status) when I^2_GX
    is below i2_gx_threshold: a low I^2_GX means the underlying
    regression is subject to dilution and the intercept p-value should
    be interpreted with that in mind. This does NOT downgrade a
    significant FAIL to WARN — a formally significant test result stays
    FAIL; the caveat is reported ALONGSIDE it, not instead of it, so a
    real finding is never silently reclassified into something softer.
    (An earlier version of this function did downgrade FAIL->WARN here;
    that was reverted because hiding a statistically significant result
    behind a business-logic softening rule is worse than reporting both
    facts — the formal result AND the reliability caveat — together.)
    """
    _validate_exposure_outcome_pair(global_exposure, global_outcome)
    _validate_numeric_param("alpha", alpha, 0, 1)
    if global_exposure.n_snps < 3:
        return EggerInterceptReport(
            intercept=float("nan"), intercept_se=float("nan"), p_value=float("nan"),
            status="NOT_APPLICABLE", alpha=alpha, reason="too_few_snps",
        )
    try:
        egger_res = mr_egger(global_exposure.beta, global_exposure.se, global_outcome.beta, global_outcome.se)
        report = check_egger_intercept(egger_res, alpha=alpha)
    except (ValueError, np.linalg.LinAlgError):
        return EggerInterceptReport(
            intercept=float("nan"), intercept_se=float("nan"), p_value=float("nan"),
            status="NOT_APPLICABLE", alpha=alpha, reason="numerical_failure",
        )

    i2 = i_squared_gx(global_exposure)
    low_i2gx = bool(np.isfinite(i2) and i2 < i2_gx_threshold)
    if low_i2gx:
        report = EggerInterceptReport(
            intercept=report.intercept, intercept_se=report.intercept_se,
            p_value=report.p_value, status=report.status, alpha=alpha,
            low_i2gx_caveat=True,
        )
    return report


@dataclass
class LeaveOneOutReport:
    snp_ids: np.ndarray
    estimates_excl: np.ndarray     # (n_snps,) IVW estimate with that SNP excluded
    full_estimate: float
    full_se: float
    shift: np.ndarray              # (n_snps,) estimates_excl - full_estimate
    is_influential: np.ndarray     # (n_snps,) bool
    n_influential: int
    influential_snp_ids: np.ndarray
    status: str  # "PASS" or "WARN"


def leave_one_out_ivw(
    global_exposure: GlobalAssociation, global_outcome: GlobalAssociation,
    se_fraction_threshold: float = INFLUENTIAL_SNP_SE_FRACTION,
) -> LeaveOneOutReport:
    """
    Re-fits IVW n_snps times, each time excluding one SNP, to find
    individually influential SNPs — ones whose removal shifts the causal
    estimate by more than se_fraction_threshold * the full-sample SE.
    Requires at least 3 SNPs (need >=2 remaining after exclusion for IVW
    to still be meaningful).
    """
    n = global_exposure.n_snps
    if n < 3:
        raise ValueError(f"leave_one_out_ivw requires at least 3 SNPs, got {n}")
    _validate_numeric_param("se_fraction_threshold", se_fraction_threshold, 0, None)
    _validate_exposure_outcome_pair(global_exposure, global_outcome)

    full_result = ivw(global_exposure.beta, global_exposure.se, global_outcome.beta, global_outcome.se)

    estimates_excl = np.empty(n)
    for j in range(n):
        mask = np.ones(n, dtype=bool)
        mask[j] = False
        res_j = ivw(global_exposure.beta[mask], global_exposure.se[mask],
                    global_outcome.beta[mask], global_outcome.se[mask])
        estimates_excl[j] = res_j.estimate

    shift = estimates_excl - full_result.estimate
    is_influential = np.abs(shift) > se_fraction_threshold * full_result.se
    n_infl = int(np.sum(is_influential))

    return LeaveOneOutReport(
        snp_ids=global_exposure.snp_ids, estimates_excl=estimates_excl,
        full_estimate=full_result.estimate, full_se=full_result.se,
        shift=shift, is_influential=is_influential, n_influential=n_infl,
        influential_snp_ids=global_exposure.snp_ids[is_influential],
        status=("WARN" if n_infl > 0 else "PASS"),
    )


@dataclass
class WeakSnpExcludedReport:
    full_estimate: float
    full_se: float
    excl_estimate: float
    excl_se: float
    n_excluded: int
    n_remaining: int
    status: str    # "PASS", "WARN" (large shift), or "NOT_APPLICABLE"
    reason: str    # "" normally; "no_weak_snps" or "all_snps_weak" when NOT_APPLICABLE --
                   # these are very different situations (nothing needed excluding vs.
                   # excluding weak SNPs would leave nothing to estimate from) and must
                   # not be reported with the same undifferentiated message.


def ivw_excluding_weak_instruments(
    global_exposure: GlobalAssociation, global_outcome: GlobalAssociation,
    f_threshold: float = WEAK_INSTRUMENT_F_THRESHOLD, shift_se_fraction: float = 0.5,
) -> WeakSnpExcludedReport:
    _validate_exposure_outcome_pair(global_exposure, global_outcome)
    _validate_numeric_param("f_threshold", f_threshold, 0, None)
    _validate_numeric_param("shift_se_fraction", shift_se_fraction, 0, None)

    strength_report = check_instrument_strength(global_exposure, threshold=f_threshold)
    full_result = ivw(global_exposure.beta, global_exposure.se, global_outcome.beta, global_outcome.se)

    is_weak = strength_report.f_stats < f_threshold
    n_remaining = int(np.sum(~is_weak))

    if strength_report.n_weak == 0:
        return WeakSnpExcludedReport(
            full_estimate=full_result.estimate, full_se=full_result.se,
            excl_estimate=float("nan"), excl_se=float("nan"),
            n_excluded=0, n_remaining=n_remaining, status="NOT_APPLICABLE", reason="no_weak_snps",
        )
    if n_remaining < 1:
        return WeakSnpExcludedReport(
            full_estimate=full_result.estimate, full_se=full_result.se,
            excl_estimate=float("nan"), excl_se=float("nan"),
            n_excluded=strength_report.n_weak, n_remaining=0, status="NOT_APPLICABLE",
            reason="all_snps_weak",
        )

    excl_result = ivw(global_exposure.beta[~is_weak], global_exposure.se[~is_weak],
                       global_outcome.beta[~is_weak], global_outcome.se[~is_weak])

    shift = abs(excl_result.estimate - full_result.estimate)
    status = "WARN" if shift > shift_se_fraction * full_result.se else "PASS"

    return WeakSnpExcludedReport(
        full_estimate=full_result.estimate, full_se=full_result.se,
        excl_estimate=excl_result.estimate, excl_se=excl_result.se,
        n_excluded=strength_report.n_weak, n_remaining=n_remaining, status=status, reason="",
    )


@dataclass
class CrossEstimatorReport:
    ivw_estimate: float
    egger_estimate: float
    median_estimate: float
    ivw_vs_egger_diff: float
    ivw_vs_median_diff: float
    status: str  # "PASS", "WARN", or "NOT_APPLICABLE"
    reason: str = ""  # set when NOT_APPLICABLE: "too_few_snps" or "numerical_failure"


def compare_estimators(
    global_exposure: GlobalAssociation, global_outcome: GlobalAssociation,
    disagreement_se_fraction: float = 1.0,
) -> CrossEstimatorReport:
    """
    Descriptive comparison, NOT a formal statistical test: flags WARN if
    IVW and MR-Egger (or IVW and weighted-median) differ by more than
    disagreement_se_fraction times their combined SE. Estimator
    disagreement is a common, informal MR sensitivity signal (Bowden et
    al.'s "MR-Egger sensitivity analysis" convention), not a p-value-
    backed claim.

    Returns status="NOT_APPLICABLE" (with nan fields) rather than
    crashing in TWO distinct situations:
      - fewer than 3 SNPs (MR-Egger/weighted-median's own minimum,
        checked up front, cheap)
      - >=3 SNPs but a numerical failure inside one of the estimators —
        MR-Egger's design matrix can be singular (near-zero variance in
        beta_exp across SNPs) even with plenty of SNPs, and
        weighted_median raises if any beta_exp is exactly 0 — both are
        caught here rather than letting them propagate and take down
        the whole diagnostics run.
    """
    _validate_exposure_outcome_pair(global_exposure, global_outcome)
    _validate_numeric_param("disagreement_se_fraction", disagreement_se_fraction, 0, None)

    if global_exposure.n_snps < 3:
        return CrossEstimatorReport(
            ivw_estimate=float("nan"), egger_estimate=float("nan"), median_estimate=float("nan"),
            ivw_vs_egger_diff=float("nan"), ivw_vs_median_diff=float("nan"), status="NOT_APPLICABLE",
            reason="too_few_snps",
        )

    try:
        ivw_res = ivw(global_exposure.beta, global_exposure.se, global_outcome.beta, global_outcome.se)
        egger_res = mr_egger(global_exposure.beta, global_exposure.se, global_outcome.beta, global_outcome.se)
        median_res = weighted_median(global_exposure.beta, global_exposure.se,
                                      global_outcome.beta, global_outcome.se)
    except (ValueError, np.linalg.LinAlgError):
        return CrossEstimatorReport(
            ivw_estimate=float("nan"), egger_estimate=float("nan"), median_estimate=float("nan"),
            ivw_vs_egger_diff=float("nan"), ivw_vs_median_diff=float("nan"), status="NOT_APPLICABLE",
            reason="numerical_failure",
        )

    combined_se_ie = np.sqrt(ivw_res.se ** 2 + egger_res.se ** 2)
    combined_se_im = np.sqrt(ivw_res.se ** 2 + median_res.se ** 2)
    diff_ie = ivw_res.estimate - egger_res.estimate
    diff_im = ivw_res.estimate - median_res.estimate

    disagree = (abs(diff_ie) > disagreement_se_fraction * combined_se_ie or
                abs(diff_im) > disagreement_se_fraction * combined_se_im)

    return CrossEstimatorReport(
        ivw_estimate=ivw_res.estimate, egger_estimate=egger_res.estimate,
        median_estimate=median_res.estimate,
        ivw_vs_egger_diff=diff_ie, ivw_vs_median_diff=diff_im,
        status=("WARN" if disagree else "PASS"), reason="",
    )


@dataclass
class I2GXReport:
    i2_gx: float
    threshold: float
    status: str  # "PASS", "WARN", or "NOT_APPLICABLE" (undefined, e.g. <2 SNPs)


def check_i_squared_gx(global_exposure: GlobalAssociation, threshold: float = I2_GX_LOW_THRESHOLD) -> I2GXReport:
    _validate_exposure_association(global_exposure)
    _validate_numeric_param("threshold", threshold, 0, 1, low_inclusive=True, high_inclusive=True)
    i2 = i_squared_gx(global_exposure)
    if np.isnan(i2):
        status = "NOT_APPLICABLE"
    elif i2 < threshold:
        status = "WARN"
    else:
        status = "PASS"
    return I2GXReport(i2_gx=i2, threshold=threshold, status=status)


def i_squared_gx(global_exposure: GlobalAssociation) -> float:
    """
    Bowden et al.'s I^2_GX: measures what fraction of the observed
    variability in per-SNP exposure associations is real signal rather
    than sampling noise (a weighted analogue of the standard I^2
    heterogeneity statistic, applied to the INSTRUMENT-STRENGTH side
    rather than the outcome side). Low I^2_GX (commonly < 0.90) means
    MR-Egger's slope is subject to regression dilution and its own
    result should be interpreted cautiously regardless of its p-value.

    Orients beta_exp the same way mr_egger() does (flip sign of any SNP
    with negative beta_exp) before computing Q — since I^2_GX is
    specifically meant to characterize the reliability of the SAME
    Egger regression, a sign-flipped SNP that mr_egger() would reorient
    to positive should not be allowed to contribute a spuriously large
    deviation to Q here just because it happened to keep its raw,
    unoriented sign. Confirmed on real simulated data: orienting changes
    I^2_GX measurably (e.g. 0.737 -> 0.711 with just 1/50 SNPs flipped),
    not a hypothetical edge case.
    """
    beta_raw, se = global_exposure.beta, global_exposure.se
    sign = np.where(beta_raw < 0, -1.0, 1.0)
    beta = beta_raw * sign  # same orientation step as mr_estimators.mr_egger()

    w = 1.0 / (se ** 2)
    beta_bar = np.sum(w * beta) / np.sum(w)
    Q = np.sum(w * (beta - beta_bar) ** 2)
    df = len(beta) - 1
    if df <= 0 or Q <= 0:
        return float("nan")
    return float(max(0.0, (Q - df) / Q))


def print_sensitivity_reports(
    egger_report: EggerInterceptReport,
    i2_report: I2GXReport,
    loo_report,  # LeaveOneOutReport or None (None when n_snps < 3)
    weak_excl_report: WeakSnpExcludedReport,
    cross_report: CrossEstimatorReport,
) -> None:
    """
    Pure printing function: takes already-built report objects (the
    SAME ones run_diagnostics.py used to compute the overall status) and
    prints them — never recomputes anything itself. This guarantees the
    printed output always exactly matches whatever the overall
    PASS/WARN/FAIL/NOT_APPLICABLE status was actually based on; a
    version that recomputed each report independently for printing could
    in principle diverge from what was actually used for the status
    (e.g. if any estimator's internals ever become non-deterministic
    without an explicit fixed seed), and would waste every computation
    twice regardless.
    """
    if egger_report.status == "NOT_APPLICABLE":
        reason_text = ("MR-Egger's design matrix was singular" if egger_report.reason == "numerical_failure"
                       else "fewer than 3 SNPs")
        print(f"[N/A] MR-Egger intercept test: {reason_text}, MR-Egger could not run")
    else:
        print(f"[{egger_report.status}] MR-Egger intercept test: intercept="
              f"{egger_report.intercept:.4f} (se={egger_report.intercept_se:.4f}), "
              f"p={egger_report.p_value:.4g}")

    if i2_report.status == "NOT_APPLICABLE":
        print("[N/A] I^2_GX: undefined (fewer than 2 SNPs or zero heterogeneity denominator)")
    else:
        print(f"[{i2_report.status}] I^2_GX = {i2_report.i2_gx:.3f} (threshold {i2_report.threshold}) — "
              f"{'MR-Egger NOME assumption looks reasonable' if i2_report.status == 'PASS' else 'MR-Egger may suffer regression dilution'}")

    if loo_report is None:
        print("[N/A] Leave-one-out IVW: fewer than 3 SNPs, not computed")
    else:
        print(f"[{loo_report.status}] Leave-one-out IVW: {loo_report.n_influential} "
              f"influential SNP(s) out of {loo_report.snp_ids.shape[0]}")
        if loo_report.n_influential > 0:
            print(f"  influential SNP IDs: {', '.join(loo_report.influential_snp_ids[:10])}")

    if weak_excl_report.status == "NOT_APPLICABLE":
        # THE FIX: distinguish the two very different NOT_APPLICABLE
        # reasons instead of always printing the same "no weak SNPs"
        # message even when the real reason was "every SNP was weak, so
        # none remained to run IVW on".
        if weak_excl_report.reason == "all_snps_weak":
            print(f"[N/A] IVW excluding weak instruments: ALL {weak_excl_report.n_excluded} SNPs "
                  f"were weak — excluding them would leave none to estimate from")
        else:
            print("[N/A] IVW excluding weak instruments: no weak SNPs to exclude")
    else:
        print(f"[{weak_excl_report.status}] IVW excluding {weak_excl_report.n_excluded} weak "
              f"SNP(s): full={weak_excl_report.full_estimate:.4f} vs "
              f"excl-weak={weak_excl_report.excl_estimate:.4f}")

    if cross_report.status == "NOT_APPLICABLE":
        reason_text = ("a numerical failure in MR-Egger or weighted-median" if cross_report.reason == "numerical_failure"
                       else "fewer than 3 SNPs")
        print(f"[N/A] Cross-estimator comparison: {reason_text}, MR-Egger/weighted-median could not run")
    else:
        print(f"[{cross_report.status}] Cross-estimator comparison: IVW={cross_report.ivw_estimate:.4f}, "
              f"MR-Egger={cross_report.egger_estimate:.4f}, WeightedMedian={cross_report.median_estimate:.4f}")


if __name__ == "__main__":
    from data_simulator import simulate_all_sites
    from federated_aggregator import aggregate_sites_with_se

    panel, exposure_sites, outcome_sites = simulate_all_sites()
    global_exposure = aggregate_sites_with_se(exposure_sites)
    global_outcome = aggregate_sites_with_se(outcome_sites)

    egger_report = safe_egger_intercept_check(global_exposure, global_outcome)
    i2_report = check_i_squared_gx(global_exposure)
    loo_report = (leave_one_out_ivw(global_exposure, global_outcome)
                  if global_exposure.n_snps >= 3 else None)
    weak_excl_report = ivw_excluding_weak_instruments(global_exposure, global_outcome)
    cross_report = compare_estimators(global_exposure, global_outcome)

    print_sensitivity_reports(egger_report, i2_report, loo_report, weak_excl_report, cross_report)
