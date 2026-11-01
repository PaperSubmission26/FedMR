"""
run_centralized_baseline.py

Pools ALL simulated individuals (across every site) directly into one
matrix per role, and runs the same OLS-per-SNP regression + MR estimators
as the federated path. This is the "computational reference" for the
Phase 2 correctness proof — NOT a ground truth (TRUE_CAUSAL_EFFECT, the
simulation's population parameter, is the ground truth; this baseline is
just what centralized analysis of the same data would produce).

Deliberately uses a DIFFERENT numerical solver than federated_aggregator.py:
federated_aggregator.py solves the 3x3 normal equations (X'X)^-1 X'Y directly
via np.linalg.solve/inv on summed sufficient statistics. This module instead
calls np.linalg.lstsq on the actual pooled design matrix (SVD-based least
squares, no explicit normal equations, no matrix inverse). If the two
disagree, that's a genuine finding, not a shared bug reproducing itself.
"""

import numpy as np

from data_simulator import simulate_all_sites_with_individuals
from federated_aggregator import GlobalAssociation
from mr_estimators import ivw, mr_egger, weighted_median
from compute_confidence_intervals import summarize, print_summary
import config


def _pool_individuals(individuals: list[dict]):
    """Stacks every site's raw arrays into one pooled (G, X, trait)."""
    G = np.vstack([d["G"] for d in individuals])
    X = np.concatenate([d["X"] for d in individuals])
    trait = np.concatenate([d["trait"] for d in individuals])
    return G, X, trait


def _pooled_ols_per_snp_lstsq(
    G: np.ndarray, X: np.ndarray, trait: np.ndarray, role: str, snp_ids: np.ndarray,
    effect_allele: np.ndarray = None, other_allele: np.ndarray = None,
) -> GlobalAssociation:
    """
    Runs trait ~ 1 + G_j + X directly on pooled individuals, per SNP,
    using np.linalg.lstsq — an independent numerical path from the
    normal-equations solve in federated_aggregator.py.
    """
    n_people, n_snps = G.shape
    beta = np.empty(n_snps)
    se = np.empty(n_snps)
    n_total = np.full(n_snps, n_people, dtype=float)

    ones = np.ones(n_people)

    for j in range(n_snps):
        Gj = G[:, j]
        design = np.column_stack([ones, Gj, X])   # [intercept, G, X]

        coef, residuals, rank, sv = np.linalg.lstsq(design, trait, rcond=None)
        if rank < design.shape[1]:
            raise np.linalg.LinAlgError(
                f"SNP {snp_ids[j]}: design matrix rank-deficient (rank={rank}, "
                f"expected {design.shape[1]}) — likely a monomorphic or collinear SNP."
            )

        fitted = design @ coef
        resid = trait - fitted
        rss = float(resid @ resid)
        dof = n_people - design.shape[1]
        sigma2 = rss / dof

        # var(coef) = sigma^2 * (X'X)^-1, obtained via the SVD of the design
        # matrix itself (U, S, Vt with design = U @ diag(S) @ Vt), so that
        # (X'X)^-1 = V @ diag(1/S^2) @ V.T — never forming design.T @ design
        # (the normal-equations gram matrix) at all. This keeps the whole
        # centralized path numerically independent of the aggregator's
        # normal-equations solve, not just at the top level.
        _, S, Vt = np.linalg.svd(design, full_matrices=False)
        XtX_inv = (Vt.T * (1.0 / S ** 2)) @ Vt
        var_bG = sigma2 * XtX_inv[1, 1]

        beta[j] = coef[1]
        se[j] = np.sqrt(max(var_bG, 0.0))

    return GlobalAssociation(
        role=role, n_snps=n_snps,
        snp_ids=snp_ids,
        effect_allele=effect_allele if effect_allele is not None else np.array([""] * n_snps),
        other_allele=other_allele if other_allele is not None else np.array([""] * n_snps),
        beta=beta, se=se, n_total=n_total,
    )


def run_centralized_baseline():
    panel, exp_sites, out_sites, exp_indiv, out_indiv = simulate_all_sites_with_individuals(
        panel=None
    )

    G_exp, X_exp, E = _pool_individuals(exp_indiv)
    G_out, X_out, Y = _pool_individuals(out_indiv)

    global_exposure = _pooled_ols_per_snp_lstsq(
        G_exp, X_exp, E, role="exposure", snp_ids=panel.snp_ids,
        effect_allele=panel.effect_allele, other_allele=panel.other_allele,
    )
    global_outcome = _pooled_ols_per_snp_lstsq(
        G_out, X_out, Y, role="outcome", snp_ids=panel.snp_ids,
        effect_allele=panel.effect_allele, other_allele=panel.other_allele,
    )

    results = [
        fn(global_exposure.beta, global_exposure.se, global_outcome.beta, global_outcome.se)
        for fn in (ivw, mr_egger, weighted_median)
    ]
    return panel, global_exposure, global_outcome, results


if __name__ == "__main__":
    panel, global_exposure, global_outcome, results = run_centralized_baseline()
    ci_results = summarize(results)
    print("=== Centralized baseline (pooled individuals, np.linalg.lstsq) ===")
    print_summary(ci_results, true_effect=config.TRUE_CAUSAL_EFFECT)
