"""
instrument_strength_check.py

Federated instrument-strength diagnostic, computed entirely from
already-aggregated global per-SNP exposure associations
(federated_aggregator.GlobalAssociation) — federated-safe by
construction.

Per-SNP F-statistic (standard weak-instrument diagnostic):
    F_j = (beta_gx_j / se_gx_j)^2
Rule of thumb (Staiger & Stock 1997): F < 10 signals a weak instrument.

STATUS is a 3-tier classification, not a single mean-F>=10 boolean —
a mean F comfortably above 10 can still hide a meaningful fraction of
individually weak SNPs, and calling that a clean PASS is misleading:
    FAIL: mean F < threshold
    WARN: mean F >= threshold, but at least one SNP has F < threshold
    PASS: mean F >= threshold AND every SNP has F >= threshold
"""

from dataclasses import dataclass
import numpy as np

from federated_aggregator import GlobalAssociation

WEAK_INSTRUMENT_F_THRESHOLD = 10.0  # Staiger & Stock (1997) rule of thumb


def _validate_global_association_structure(ge: GlobalAssociation) -> None:
    """
    Basic structural sanity every GlobalAssociation should satisfy,
    regardless of role: n_snps must match the actual length of every
    per-SNP array. A GlobalAssociation is a plain dataclass with no
    __post_init__ validation, so nothing enforces this at construction —
    callers (including test fixtures) could otherwise build one with
    mismatched lengths and every downstream elementwise operation would
    silently misalign SNPs instead of raising.
    """
    arrays = {
        "snp_ids": ge.snp_ids, "effect_allele": ge.effect_allele, "other_allele": ge.other_allele,
        "beta": ge.beta, "se": ge.se, "n_total": ge.n_total,
    }
    for name, arr in arrays.items():
        if len(arr) != ge.n_snps:
            raise ValueError(
                f"GlobalAssociation structural mismatch: n_snps={ge.n_snps} but "
                f"{name} has length {len(arr)}."
            )


def _validate_exposure_association(ge: GlobalAssociation) -> None:
    """Reusable check for any function that specifically needs an EXPOSURE
    association with finite beta/se and strictly positive se — used here
    and by sensitivity_analysis.check_i_squared_gx (which needs the exact
    same guarantees, since I^2_GX is defined on the exposure side)."""
    _validate_global_association_structure(ge)
    if ge.role != "exposure":
        raise ValueError(
            f"Requires an EXPOSURE association (got role={ge.role!r}) — this diagnostic is "
            f"about the SNP -> exposure relationship."
        )
    if ge.n_snps < 1:
        raise ValueError("Requires at least 1 SNP.")
    if not np.all(np.isfinite(ge.beta)) or not np.all(np.isfinite(ge.se)):
        raise ValueError("beta/se contain NaN or infinite values.")
    if np.any(ge.se <= 0):
        raise ValueError("se must be strictly positive.")


def _validate_global_association(ge: GlobalAssociation, threshold: float) -> None:
    _validate_exposure_association(ge)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError(f"threshold must be finite and > 0, got {threshold}")


@dataclass
class InstrumentStrengthReport:
    n_snps: int
    f_stats: np.ndarray
    mean_f: float
    median_f: float
    min_f: float
    n_weak: int
    frac_weak: float
    weak_snp_ids: np.ndarray
    threshold: float
    status: str  # "PASS", "WARN", or "FAIL"
    sensitivity_mean_f_excl_weak: float  # mean F among the remaining SNPs, after dropping weak ones
    n_strong_remaining: int


def check_instrument_strength(
    global_exposure: GlobalAssociation, threshold: float = WEAK_INSTRUMENT_F_THRESHOLD
) -> InstrumentStrengthReport:
    _validate_global_association(global_exposure, threshold)

    f_stats = (global_exposure.beta / global_exposure.se) ** 2
    is_weak = f_stats < threshold
    n_weak = int(np.sum(is_weak))
    mean_f = float(np.mean(f_stats))

    if mean_f < threshold:
        status = "FAIL"
    elif n_weak > 0:
        status = "WARN"
    else:
        status = "PASS"

    n_strong_remaining = global_exposure.n_snps - n_weak
    sensitivity_mean_f = float(np.mean(f_stats[~is_weak])) if n_strong_remaining > 0 else float("nan")

    return InstrumentStrengthReport(
        n_snps=global_exposure.n_snps,
        f_stats=f_stats,
        mean_f=mean_f,
        median_f=float(np.median(f_stats)),
        min_f=float(np.min(f_stats)),
        n_weak=n_weak,
        frac_weak=float(np.mean(is_weak)),
        weak_snp_ids=global_exposure.snp_ids[is_weak],
        threshold=threshold,
        status=status,
        sensitivity_mean_f_excl_weak=sensitivity_mean_f,
        n_strong_remaining=n_strong_remaining,
    )


def print_report(report: InstrumentStrengthReport) -> None:
    print(f"[{report.status}] Instrument strength (mean F={report.mean_f:.2f}, "
          f"threshold={report.threshold})")
    print(f"  median F={report.median_f:.2f}  min F={report.min_f:.2f}")
    print(f"  weak SNPs (F<{report.threshold}): {report.n_weak}/{report.n_snps} "
          f"({report.frac_weak:.1%})")
    if report.n_weak > 0:
        shown = report.weak_snp_ids[:10]
        suffix = "..." if report.n_weak > 10 else ""
        print(f"  weak SNP IDs: {', '.join(shown)}{suffix}")
        print(f"  sensitivity: excluding weak SNPs, mean F over remaining "
              f"{report.n_strong_remaining} SNPs = {report.sensitivity_mean_f_excl_weak:.2f}")
    if report.status == "WARN":
        print("  WARN: overall mean F passes, but some individual SNPs are weak — consider "
              "a sensitivity analysis re-running MR with weak SNPs excluded "
              "(see run_diagnostics.py's sensitivity_analysis output).")


if __name__ == "__main__":
    from data_simulator import simulate_all_sites
    from federated_aggregator import aggregate_sites_with_se

    panel, exposure_sites, outcome_sites = simulate_all_sites()
    global_exposure = aggregate_sites_with_se(exposure_sites)

    report = check_instrument_strength(global_exposure)
    print_report(report)
