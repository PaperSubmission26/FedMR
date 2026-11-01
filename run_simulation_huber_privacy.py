from pathlib import Path
from contextlib import contextmanager
import argparse, math
import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq
from scipy.special import ndtr, log_ndtr

import config
from data_simulator import simulate_all_sites_with_individuals, _derive_sufficient_stats
from federated_aggregator import aggregate_sites_with_se

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/simulations/paper"
REP_OUT = OUT / "Table_simulation_huber_privacy_repetitions.tsv"
NONPRIVATE_OUT = OUT / "Table_simulation_huber_privacy_nonprivate_summary.tsv"
DP_OUT = OUT / "Table_simulation_huber_privacy_summary.tsv"
EXACT_OUT = OUT / "Table_simulation_huber_federated_exactness.tsv"

TAU, LAMBDA, DELTA = 0.25, 0.01, 1e-6
EPS = [0.5, 1.0, 2.0, 5.0, 10.0]
X_BOUND, TRAIT_BOUND = 4.0, 8.0
CX, BETA_SCALE, D = math.sqrt(3.0), 4.0, 3
N_VALUES, M_VALUES, N_SITES = [729, 2500, 7500], [3, 5, 10, 20], 5
TRUE_BETA, GX_MEAN, GX_SD = 0.30, 0.30, 0.03
BASE_SEED = 80000
SMOKE_REPS, FULL_REPS = 2, 100
SMOKE_DRAWS, FULL_DRAWS = 10, 50


@contextmanager
def cfg(overrides):
    x = dict(overrides)
    x["N_EXPOSURE_SITES"] = len(x["EXPOSURE_SITE_SIZES"])
    x["N_OUTCOME_SITES"] = len(x["OUTCOME_SITE_SIZES"])
    saved = {k: getattr(config, k) for k in x}
    try:
        for k, v in x.items():
            setattr(config, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(config, k, v)


def sizes(total):
    q, r = divmod(total, N_SITES)
    return [q + (i < r) for i in range(N_SITES)]


def local(theta, Z, y):
    r = y - Z @ theta
    a = np.abs(r)
    loss = np.where(a <= TAU, 0.5 * r * r, TAU * a - 0.5 * TAU * TAU)
    return float(loss.sum()), -(Z.T @ np.clip(r, -TAU, TAU))


def central_obj(theta, Z, y):
    loss, grad = local(theta, Z, y)
    n = len(y)
    return loss / n + 0.5 * LAMBDA * float(theta @ theta), grad / n + LAMBDA * theta


def fed_obj(theta, sites):
    loss, grad, n = 0.0, np.zeros(len(theta)), 0
    for Z, y in sites:
        l, g = local(theta, Z, y)
        loss, grad, n = loss + l, grad + g, n + len(y)
    return loss / n + 0.5 * LAMBDA * float(theta @ theta), grad / n + LAMBDA * theta


def fit(fun):
    return minimize(fun, np.zeros(D), jac=True, method="L-BFGS-B",
                    options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-10, "maxls": 100})


def analytic_delta(eps, sigma, sens):
    a, b = sens / (2 * sigma), eps * sigma / sens
    z = eps + log_ndtr(-a - b)
    t2 = 0.0 if z < -745 else math.exp(z)
    return float(min(1.0, max(0.0, ndtr(a - b) - t2)))


def sigma_for(eps, sens):
    lo, hi = sens * 1e-12, sens
    while analytic_delta(eps, hi, sens) > DELTA:
        hi *= 2
    return float(brentq(lambda s: analytic_delta(eps, s, sens) - DELTA,
                        lo, hi, xtol=1e-12, rtol=1e-12))


def ivw(bx, by, sey):
    bx, by, sey = map(lambda z: np.asarray(z, float), (bx, by, sey))
    w = 1 / sey**2
    den = float(np.sum(w * bx**2))
    est = float(np.sum(w * bx * by) / den)
    Q = float(np.sum(w * (by - est * bx)**2))
    df = len(bx) - 1
    phi = max(Q / df, 1.0) if df else 1.0
    return est, float(np.sqrt(phi / den)), den


def ivw_draws(B, by, sey):
    w = 1 / np.asarray(sey, float)**2
    by = np.asarray(by, float)
    den = B**2 @ w
    est = (B @ (w * by)) / den
    resid = by[None, :] - est[:, None] * B
    df = B.shape[1] - 1
    phi = np.maximum(np.sum(w[None, :] * resid**2, axis=1) / df, 1.0)
    return est, np.sqrt(phi / den), den


def prepare_exposure(individuals, panel):
    clipped_stats, scaled = [], []
    for d in individuals:
        Xr, yr = np.asarray(d["X"], float), np.asarray(d["trait"], float)
        Xc, yc = np.clip(Xr, -X_BOUND, X_BOUND), np.clip(yr, -TRAIT_BOUND, TRAIT_BOUND)
        clipped_stats.append(_derive_sufficient_stats(d["site_name"], "exposure", panel, d["G"], Xc, yc))
        scaled.append({"H": np.asarray(d["G"], float) / 2.0,
                       "X": Xc / X_BOUND, "y": yc / TRAIT_BOUND,
                       "Xr": Xr, "yr": yr})
    return clipped_stats, scaled


def huber_panel(scaled, m, audit):
    betas, exact = np.empty(m), []
    for j in range(m):
        sites = []
        for s in scaled:
            Z = np.column_stack([np.ones(len(s["y"])), s["H"][:, j], s["X"]])
            if np.max(np.linalg.norm(Z, axis=1)) > CX + 1e-12:
                raise RuntimeError("synthetic design bound violated")
            sites.append((Z, s["y"]))
        ff = fit(lambda th: fed_obj(th, sites))
        if not ff.success:
            raise RuntimeError(f"federated Huber failed: {ff.message}")
        betas[j] = BETA_SCALE * ff.x[1]
        if audit:
            Za, ya = np.vstack([z for z, _ in sites]), np.concatenate([y for _, y in sites])
            fc = fit(lambda th: central_obj(th, Za, ya))
            if not fc.success:
                raise RuntimeError(f"central Huber failed: {fc.message}")
            oc, gc = central_obj(fc.x, Za, ya)
            of, gf = fed_obj(fc.x, sites)
            exact.append({"instrument_index": j + 1,
                          "central_huber_beta": BETA_SCALE * fc.x[1],
                          "federated_huber_beta": betas[j],
                          "abs_beta_difference": abs(BETA_SCALE * fc.x[1] - betas[j]),
                          "objective_difference": abs(oc - of),
                          "max_gradient_difference": float(np.max(np.abs(gc - gf)))})
    return betas, exact


def one_dataset(n, m, rep, draws, audit):
    cell = N_VALUES.index(n) * len(M_VALUES) + M_VALUES.index(m)
    seed = BASE_SEED + 100000 * (cell + 1) + rep
    ss = sizes(n)
    overrides = dict(RANDOM_SEED=seed, N_SNPS=m, TRUE_CAUSAL_EFFECT=TRUE_BETA,
                     MAF_RANGE=(0.05, 0.50), BETA_GX_MEAN=GX_MEAN, BETA_GX_SD=GX_SD,
                     WEAK_INSTRUMENT_FRACTION=0.0, WEAK_INSTRUMENT_SCALE=0.10,
                     PLEIOTROPY_FRACTION=0.0, PLEIOTROPY_MEAN=0.0, PLEIOTROPY_SD=0.08,
                     SITE_HETEROGENEITY_SD=0.0, EXPOSURE_SITE_SIZES=ss, OUTCOME_SITE_SIZES=ss)

    with cfg(overrides):
        panel, exp_raw_sites, out_sites, exp_indiv, _ = simulate_all_sites_with_individuals()
        exp_raw, out = aggregate_sites_with_se(exp_raw_sites), aggregate_sites_with_se(out_sites)
        clipped_sites, scaled = prepare_exposure(exp_indiv, panel)
        exp_clip = aggregate_sites_with_se(clipped_sites)
        bx_huber, exact = huber_panel(scaled, m, audit)
        for r in exact:
            r.update(total_n_per_role=n, panel_size=m)

        raw, clipped, hub = ivw(exp_raw.beta, out.beta, out.se), ivw(exp_clip.beta, out.beta, out.se), ivw(bx_huber, out.beta, out.se)
        F = (exp_raw.beta / exp_raw.se)**2
        npeople = sum(len(s["yr"]) for s in scaled)
        base = {"total_n_per_role": n, "panel_size": m, "repetition": rep, "seed": seed,
                "truth": TRUE_BETA, "median_observed_F": float(np.median(F)),
                "fraction_observed_F_gt_10": float(np.mean(F > 10)),
                "fraction_X_clipped": sum(np.sum(np.abs(s["Xr"]) > X_BOUND) for s in scaled) / npeople,
                "fraction_exposure_trait_clipped": sum(np.sum(np.abs(s["yr"]) > TRAIT_BOUND) for s in scaled) / npeople,
                "raw_OLS_IVW": raw[0], "raw_OLS_IVW_SE": raw[1],
                "clipped_OLS_IVW": clipped[0], "clipped_OLS_IVW_SE": clipped[1],
                "Huber_IVW": hub[0], "Huber_IVW_conditional_SE": hub[1],
                "clipping_MR_shift": clipped[0] - raw[0],
                "Huber_MR_shift_vs_raw_OLS": hub[0] - raw[0],
                "Huber_coefficient_RMSE_vs_true_beta_gx": float(np.sqrt(np.mean((bx_huber - panel.beta_gx)**2)))}

        scalar_sens = 2 * TAU * CX / (n * LAMBDA)
        joint_sens = math.sqrt(m) * scalar_sens
        dp_rows = []
        for eps in EPS:
            sig = BETA_SCALE * sigma_for(eps, joint_sens)
            rng = np.random.default_rng(seed + int(10000 * eps) + 9999999)
            B = bx_huber[None, :] + rng.normal(0, sig, size=(draws, m))
            est, se, den = ivw_draws(B, out.beta, out.se)
            finite = np.isfinite(est) & np.isfinite(se) & np.isfinite(den)
            invalid = 1 - float(np.mean(finite))
            est, se, den, B = est[finite], se[finite], den[finite], B[finite]
            if not len(est):
                raise RuntimeError("all DP MR draws invalid")
            low, high = est - 1.96 * se, est + 1.96 * se
            dp_rows.append({**base, "epsilon_total": eps, "delta_total": DELTA,
                            "n_DP_draws": len(est), "sigma_original": sig,
                            "DP_MR_bias_mean": float(np.mean(est - TRUE_BETA)),
                            "DP_MR_sq_error_mean": float(np.mean((est - TRUE_BETA)**2)),
                            "DP_MR_direction_recovery": float(np.mean(np.sign(est) == np.sign(TRUE_BETA))),
                            "naive_conditional_95_coverage": float(np.mean((low <= TRUE_BETA) & (TRUE_BETA <= high))),
                            "prob_denominator_below_0_10_Huber_reference": float(np.mean(den / hub[2] < 0.10)),
                            "DP_coefficient_sq_error_mean": float(np.mean((B - panel.beta_gx[None, :])**2)),
                            "DP_coefficient_sign_recovery": float(np.mean(np.sign(B) == np.sign(panel.beta_gx[None, :]))),
                            "invalid_DP_MR_rate": invalid})
    return base, dp_rows, exact


def summaries(reps, dp):
    non = []
    for (n, m), g in reps.groupby(["total_n_per_role", "panel_size"]):
        for layer, ec, sc in [("raw_OLS", "raw_OLS_IVW", "raw_OLS_IVW_SE"),
                              ("clipped_OLS", "clipped_OLS_IVW", "clipped_OLS_IVW_SE"),
                              ("Huber_nonprivate", "Huber_IVW", "Huber_IVW_conditional_SE")]:
            e, s = g[ec].to_numpy(float), g[sc].to_numpy(float)
            non.append({"total_n_per_role": n, "panel_size": m, "layer": layer,
                        "n_repetitions": len(g), "mean_estimate": float(e.mean()),
                        "bias": float(np.mean(e - TRUE_BETA)),
                        "RMSE": float(np.sqrt(np.mean((e - TRUE_BETA)**2))),
                        "conditional_95_coverage": float(np.mean((e - 1.96*s <= TRUE_BETA) & (TRUE_BETA <= e + 1.96*s))),
                        "median_observed_F": float(g["median_observed_F"].median()),
                        "mean_fraction_trait_clipped": float(g["fraction_exposure_trait_clipped"].mean())})
    non = pd.DataFrame(non)

    rows = []
    for (n, m, eps), g in dp.groupby(["total_n_per_role", "panel_size", "epsilon_total"]):
        rows.append({"total_n_per_role": n, "panel_size": m, "epsilon_total": eps,
                     "n_datasets": len(g), "DP_draws_per_dataset": int(g["n_DP_draws"].iloc[0]),
                     "sigma_original": float(g["sigma_original"].iloc[0]),
                     "DP_MR_bias": float(g["DP_MR_bias_mean"].mean()),
                     "DP_MR_RMSE": float(np.sqrt(g["DP_MR_sq_error_mean"].mean())),
                     "direction_recovery": float(g["DP_MR_direction_recovery"].mean()),
                     "naive_conditional_95_coverage": float(g["naive_conditional_95_coverage"].mean()),
                     "prob_denominator_below_0_10_Huber_reference": float(g["prob_denominator_below_0_10_Huber_reference"].mean()),
                     "DP_coefficient_RMSE_vs_true_beta_gx": float(np.sqrt(g["DP_coefficient_sq_error_mean"].mean())),
                     "DP_coefficient_sign_recovery": float(g["DP_coefficient_sign_recovery"].mean()),
                     "invalid_DP_MR_rate": float(g["invalid_DP_MR_rate"].mean())})
    return non, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    reps_n = SMOKE_REPS if args.smoke else FULL_REPS
    draws = SMOKE_DRAWS if args.smoke else FULL_DRAWS
    reps, dp_rows, exact = [], [], []

    print("=== SYNTHETIC HUBER JOINT-OUTPUT PRIVACY STUDY ===")
    print("Locked tau/lambda:", TAU, LAMBDA)
    print("Synthetic Cx:", CX)
    print("Repetitions/cell:", reps_n, "DP draws/dataset/epsilon:", draws)

    cells = [(n, m) for n in N_VALUES for m in M_VALUES]
    for k, (n, m) in enumerate(cells, 1):
        print(f"[{k}/{len(cells)}] N={n}, m={m}")
        for rep in range(reps_n):
            b, d, e = one_dataset(n, m, rep, draws, rep == 0)
            reps.append(b); dp_rows.extend(d); exact.extend(e)
        print(f"  repetitions: {reps_n}/{reps_n}")

    reps, dp_rows, exact = pd.DataFrame(reps), pd.DataFrame(dp_rows), pd.DataFrame(exact)
    non, dpsum = summaries(reps, dp_rows)
    reps.to_csv(REP_OUT, sep="\t", index=False)
    non.to_csv(NONPRIVATE_OUT, sep="\t", index=False)
    dpsum.to_csv(DP_OUT, sep="\t", index=False)
    exact.to_csv(EXACT_OUT, sep="\t", index=False)

    print("\n=== HUBER FEDERATED EXACTNESS ===")
    print(exact.groupby(["total_n_per_role", "panel_size"], as_index=False).agg(
        max_abs_beta_difference=("abs_beta_difference", "max"),
        max_objective_difference=("objective_difference", "max"),
        max_gradient_difference=("max_gradient_difference", "max")).to_string(index=False))
    print("\n=== NONPRIVATE LAYERS ===")
    print(non.to_string(index=False))
    print("\n=== JOINT DP SUMMARY ===")
    print(dpsum.to_string(index=False))
    print("\nRESULT: PASS")


if __name__ == "__main__":
    main()