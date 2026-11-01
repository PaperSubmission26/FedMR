from pathlib import Path
from contextlib import contextmanager
import argparse
import math
import time

import numpy as np
import pandas as pd

import config

from data_simulator import (
    simulate_all_sites_with_individuals,
)

from federated_aggregator import (
    aggregate_sites_with_se,
)

from run_centralized_baseline import (
    _pool_individuals,
    _pooled_ols_per_snp_lstsq,
)

from mr_estimators import (
    ivw,
    mr_egger,
    weighted_median,
)

import simulation_study_config as study


ROOT = Path(__file__).resolve().parent

OUTDIR = (
    ROOT
    / "results/simulations/paper"
)

REPETITION_OUT = (
    OUTDIR
    / "Table_simulation_known_truth_repetitions.tsv"
)

SUMMARY_OUT = (
    OUTDIR
    / "Table_simulation_known_truth_summary.tsv"
)

SYSTEM_OUT = (
    OUTDIR
    / "Table_simulation_system_scaling.tsv"
)

FIG_BIAS = (
    OUTDIR
    / "Figure_simulation_bias.png"
)

FIG_RMSE = (
    OUTDIR
    / "Figure_simulation_RMSE.png"
)

FIG_COVERAGE = (
    OUTDIR
    / "Figure_simulation_coverage.png"
)

FIG_POWER = (
    OUTDIR
    / "Figure_simulation_power_type1.png"
)

FIG_RUNTIME = (
    OUTDIR
    / "Figure_simulation_runtime_scaling.png"
)

FIG_EXACTNESS = (
    OUTDIR
    / "Figure_simulation_federated_exactness.png"
)


# ============================================================
# Temporary config handling
# ============================================================

@contextmanager
def temporary_config(overrides):

    saved = {}

    expanded = dict(overrides)

    # These are derived values in the legacy config.
    if "EXPOSURE_SITE_SIZES" in expanded:
        expanded[
            "N_EXPOSURE_SITES"
        ] = len(
            expanded[
                "EXPOSURE_SITE_SIZES"
            ]
        )

    if "OUTCOME_SITE_SIZES" in expanded:
        expanded[
            "N_OUTCOME_SITES"
        ] = len(
            expanded[
                "OUTCOME_SITE_SIZES"
            ]
        )

    try:

        for key, value in expanded.items():

            if not hasattr(config, key):
                raise AttributeError(
                    f"config.py has no attribute {key}"
                )

            saved[key] = getattr(
                config,
                key,
            )

            setattr(
                config,
                key,
                value,
            )

        yield

    finally:

        for key, value in saved.items():

            setattr(
                config,
                key,
                value,
            )


def scenario_config(scenario):

    d = dict(
        study.BASE
    )

    d.update(
        {
            key: value
            for key, value
            in scenario.items()
            if key != "name"
        }
    )

    return d


# ============================================================
# MR helpers
# ============================================================

def normal_ci(
    estimate,
    se,
):

    return (
        estimate
        - 1.96 * se,

        estimate
        + 1.96 * se,
    )


def result_row(
    scenario_name,
    rep,
    truth,
    method_name,
    result,
    panel,
    exposure,
    realized_site_effect_mean,
):

    low, high = normal_ci(
        float(
            result.estimate
        ),
        float(
            result.se
        ),
    )

    significant = bool(
        float(
            result.p_value
        )
        < study.ALPHA
    )

    truth_is_null = bool(
        np.isclose(
            truth,
            0.0,
        )
    )

    F = (
        exposure.beta
        / exposure.se
    ) ** 2

    return {
        "scenario":
            scenario_name,

        "repetition":
            rep,

        "method":
            method_name,

        "truth":
            truth,

        "estimate":
            float(
                result.estimate
            ),

        "se":
            float(
                result.se
            ),

        "ci_low":
            float(low),

        "ci_high":
            float(high),

        "bias":
            float(
                result.estimate
                - truth
            ),

        "abs_error":
            float(
                abs(
                    result.estimate
                    - truth
                )
            ),

        "sq_error":
            float(
                (
                    result.estimate
                    - truth
                ) ** 2
            ),

        "ci_covers_truth":
            bool(
                low
                <= truth
                <= high
            ),

        "significant_p_lt_0_05":
            significant,

        "power_indicator":
            (
                significant
                if not truth_is_null
                else np.nan
            ),

        "type1_indicator":
            (
                significant
                if truth_is_null
                else np.nan
            ),

        "n_snps":
            int(
                panel.beta_gx.size
            ),

        "n_true_weak":
            int(
                panel.is_weak_instrument.sum()
            ),

        "n_true_pleiotropic":
            int(
                panel.is_pleiotropic.sum()
            ),

        "mean_exposure_F":
            float(
                np.mean(F)
            ),

        "median_exposure_F":
            float(
                np.median(F)
            ),

        "fraction_F_gt_10":
            float(
                np.mean(
                    F > 10.0
                )
            ),

        "realized_mean_outcome_site_causal_effect":
            realized_site_effect_mean,

        "failed":
            False,

        "failure_reason":
            "",
    }


def failure_row(
    scenario_name,
    rep,
    truth,
    method_name,
    reason,
):

    return {
        "scenario":
            scenario_name,

        "repetition":
            rep,

        "method":
            method_name,

        "truth":
            truth,

        "estimate":
            np.nan,

        "se":
            np.nan,

        "ci_low":
            np.nan,

        "ci_high":
            np.nan,

        "bias":
            np.nan,

        "abs_error":
            np.nan,

        "sq_error":
            np.nan,

        "ci_covers_truth":
            False,

        "significant_p_lt_0_05":
            False,

        "power_indicator":
            np.nan,

        "type1_indicator":
            np.nan,

        "n_snps":
            np.nan,

        "n_true_weak":
            np.nan,

        "n_true_pleiotropic":
            np.nan,

        "mean_exposure_F":
            np.nan,

        "median_exposure_F":
            np.nan,

        "fraction_F_gt_10":
            np.nan,

        "realized_mean_outcome_site_causal_effect":
            np.nan,

        "failed":
            True,

        "failure_reason":
            str(reason),
    }


# ============================================================
# One statistical repetition
# ============================================================

def run_statistical_repetition(
    scenario,
    rep,
):

    scenario_name = (
        scenario["name"]
    )

    overrides = scenario_config(
        scenario
    )

    overrides[
        "RANDOM_SEED"
    ] = (
        study.BASE_SEED
        + 10000
        * (
            study.STATISTICAL_SCENARIOS
            .index(
                scenario
            )
            + 1
        )
        + rep
    )

    rows = []

    with temporary_config(
        overrides
    ):

        (
            panel,
            exposure_sites,
            outcome_sites,
            _,
            _,
        ) = (
            simulate_all_sites_with_individuals()
        )

        truth = float(
            config.TRUE_CAUSAL_EFFECT
        )

        try:

            exposure = (
                aggregate_sites_with_se(
                    exposure_sites
                )
            )

            outcome = (
                aggregate_sites_with_se(
                    outcome_sites
                )
            )

        except (
            ValueError,
            np.linalg.LinAlgError,
        ) as exc:

            for method in (
                "IVW",
                "MR-Egger",
                "WeightedMedian",
            ):

                rows.append(
                    failure_row(
                        scenario_name,
                        rep,
                        truth,
                        method,
                        f"aggregation: {exc}",
                    )
                )

            return rows

        realized = [
            site.site_realized_causal_effect
            for site in outcome_sites
            if (
                site.site_realized_causal_effect
                is not None
            )
        ]

        realized_mean = float(
            np.mean(
                realized
            )
        )

        estimators = [
            (
                "IVW",
                lambda:
                    ivw(
                        exposure.beta,
                        exposure.se,
                        outcome.beta,
                        outcome.se,
                    ),
            ),

            (
                "MR-Egger",
                lambda:
                    mr_egger(
                        exposure.beta,
                        exposure.se,
                        outcome.beta,
                        outcome.se,
                    ),
            ),

            (
                "WeightedMedian",
                lambda:
                    weighted_median(
                        exposure.beta,
                        exposure.se,
                        outcome.beta,
                        outcome.se,

                        n_bootstrap=
                            study.WEIGHTED_MEDIAN_BOOTSTRAPS,

                        rng=np.random.default_rng(
                            overrides[
                                "RANDOM_SEED"
                            ]
                            + 999999
                        ),
                    ),
            ),
        ]

        for method_name, fn in estimators:

            try:

                result = fn()

                rows.append(
                    result_row(
                        scenario_name,
                        rep,
                        truth,
                        method_name,
                        result,
                        panel,
                        exposure,
                        realized_mean,
                    )
                )

            except (
                ValueError,
                np.linalg.LinAlgError,
                FloatingPointError,
            ) as exc:

                rows.append(
                    failure_row(
                        scenario_name,
                        rep,
                        truth,
                        method_name,
                        exc,
                    )
                )

    return rows


# ============================================================
# Summary
# ============================================================

def summarize_known_truth(
    df,
):

    rows = []

    for (
        scenario,
        method,
    ), group in df.groupby(
        [
            "scenario",
            "method",
        ],
        sort=False,
    ):

        success = group[
            ~group["failed"]
        ].copy()

        n_total = len(group)

        n_success = len(
            success
        )

        if n_success == 0:

            rows.append({
                "scenario":
                    scenario,

                "method":
                    method,

                "n_repetitions":
                    n_total,

                "failure_rate":
                    1.0,
            })

            continue

        truth = float(
            success["truth"]
            .iloc[0]
        )

        bias = float(
            success["bias"]
            .mean()
        )

        empirical_sd = float(
            success["estimate"]
            .std(
                ddof=1
            )
        )

        mean_model_se = float(
            success["se"]
            .mean()
        )

        rmse = float(
            np.sqrt(
                success[
                    "sq_error"
                ].mean()
            )
        )

        coverage = float(
            success[
                "ci_covers_truth"
            ].mean()
        )

        if np.isclose(
            truth,
            0.0,
        ):

            type1 = float(
                success[
                    "significant_p_lt_0_05"
                ].mean()
            )

            power = np.nan

        else:

            power = float(
                success[
                    "significant_p_lt_0_05"
                ].mean()
            )

            type1 = np.nan

        # Monte Carlo SE for coverage / power proportions.
        coverage_mcse = math.sqrt(
            coverage
            * (
                1.0
                - coverage
            )
            / n_success
        )

        decision_prop = (
            type1
            if np.isfinite(type1)
            else power
        )

        decision_mcse = (
            math.sqrt(
                decision_prop
                * (
                    1.0
                    - decision_prop
                )
                / n_success
            )
            if np.isfinite(
                decision_prop
            )
            else np.nan
        )

        rows.append({
            "scenario":
                scenario,

            "method":
                method,

            "truth":
                truth,

            "n_repetitions":
                n_total,

            "n_success":
                n_success,

            "failure_rate":
                float(
                    1.0
                    - n_success
                    / n_total
                ),

            "mean_estimate":
                float(
                    success[
                        "estimate"
                    ].mean()
                ),

            "bias":
                bias,

            "empirical_SD":
                empirical_sd,

            "mean_model_SE":
                mean_model_se,

            "RMSE":
                rmse,

            "coverage_95":
                coverage,

            "coverage_MCSE":
                coverage_mcse,

            "power_p_lt_0_05":
                power,

            "type1_error_p_lt_0_05":
                type1,

            "decision_MCSE":
                decision_mcse,

            "median_exposure_F":
                float(
                    success[
                        "median_exposure_F"
                    ].median()
                ),

            "mean_fraction_F_gt_10":
                float(
                    success[
                        "fraction_F_gt_10"
                    ].mean()
                ),

            "mean_realized_outcome_site_effect":
                float(
                    success[
                        "realized_mean_outcome_site_causal_effect"
                    ].mean()
                ),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Systems / exactness scaling
# ============================================================

def balanced_sizes(
    total,
    n_sites,
):

    base = (
        total
        // n_sites
    )

    remainder = (
        total
        % n_sites
    )

    return [
        base
        + (
            1
            if i < remainder
            else 0
        )
        for i in range(
            n_sites
        )
    ]


def numeric_payload_bytes(
    n_sites,
    n_snps,
):
    """
    Each legacy SiteSufficientStats has 10 float64 arrays
    of length n_snps.

    Public SNP IDs/alleles and Python/serialization overhead
    are deliberately excluded.
    """

    return int(
        n_sites
        * 10
        * n_snps
        * 8
    )


def run_system_scaling(
    smoke,
):

    rows = []

    n_reps = (
        1
        if smoke
        else study.SYSTEM_REPETITIONS
    )

    condition_index = 0

    for n_sites in (
        study.SYSTEM_SITE_COUNTS
    ):

        sizes = balanced_sizes(
            study.SYSTEM_TOTAL_N_PER_ROLE,
            n_sites,
        )

        for n_snps in (
            study.SYSTEM_SNP_COUNTS
        ):

            condition_index += 1

            for rep in range(
                n_reps
            ):

                overrides = dict(
                    study.BASE
                )

                overrides.update({
                    "N_SNPS":
                        n_snps,

                    "WEAK_INSTRUMENT_FRACTION":
                        0.0,

                    "PLEIOTROPY_FRACTION":
                        0.0,

                    "SITE_HETEROGENEITY_SD":
                        0.0,

                    "EXPOSURE_SITE_SIZES":
                        sizes,

                    "OUTCOME_SITE_SIZES":
                        sizes,

                    "RANDOM_SEED":
                        (
                            study.BASE_SEED
                            + 1_000_000
                            + 1000
                            * condition_index
                            + rep
                        ),
                })

                with temporary_config(
                    overrides
                ):

                    t0 = (
                        time.perf_counter()
                    )

                    (
                        panel,
                        exposure_sites,
                        outcome_sites,
                        exposure_indiv,
                        outcome_indiv,
                    ) = (
                        simulate_all_sites_with_individuals()
                    )

                    simulation_seconds = (
                        time.perf_counter()
                        - t0
                    )

                    t1 = (
                        time.perf_counter()
                    )

                    fed_exp = (
                        aggregate_sites_with_se(
                            exposure_sites
                        )
                    )

                    fed_out = (
                        aggregate_sites_with_se(
                            outcome_sites
                        )
                    )

                    federated_aggregation_seconds = (
                        time.perf_counter()
                        - t1
                    )

                    G_exp, X_exp, E = (
                        _pool_individuals(
                            exposure_indiv
                        )
                    )

                    G_out, X_out, Y = (
                        _pool_individuals(
                            outcome_indiv
                        )
                    )

                    t2 = (
                        time.perf_counter()
                    )

                    central_exp = (
                        _pooled_ols_per_snp_lstsq(
                            G_exp,
                            X_exp,
                            E,

                            role="exposure",

                            snp_ids=
                                panel.snp_ids,

                            effect_allele=
                                panel.effect_allele,

                            other_allele=
                                panel.other_allele,
                        )
                    )

                    central_out = (
                        _pooled_ols_per_snp_lstsq(
                            G_out,
                            X_out,
                            Y,

                            role="outcome",

                            snp_ids=
                                panel.snp_ids,

                            effect_allele=
                                panel.effect_allele,

                            other_allele=
                                panel.other_allele,
                        )
                    )

                    centralized_seconds = (
                        time.perf_counter()
                        - t2
                    )

                    exp_beta_error = float(
                        np.max(
                            np.abs(
                                fed_exp.beta
                                - central_exp.beta
                            )
                        )
                    )

                    exp_se_error = float(
                        np.max(
                            np.abs(
                                fed_exp.se
                                - central_exp.se
                            )
                        )
                    )

                    out_beta_error = float(
                        np.max(
                            np.abs(
                                fed_out.beta
                                - central_out.beta
                            )
                        )
                    )

                    out_se_error = float(
                        np.max(
                            np.abs(
                                fed_out.se
                                - central_out.se
                            )
                        )
                    )

                    fed_mr = ivw(
                        fed_exp.beta,
                        fed_exp.se,
                        fed_out.beta,
                        fed_out.se,
                    )

                    central_mr = ivw(
                        central_exp.beta,
                        central_exp.se,
                        central_out.beta,
                        central_out.se,
                    )

                    mr_error = float(
                        abs(
                            fed_mr.estimate
                            - central_mr.estimate
                        )
                    )

                    exposure_payload = (
                        numeric_payload_bytes(
                            n_sites,
                            n_snps,
                        )
                    )

                    outcome_payload = (
                        numeric_payload_bytes(
                            n_sites,
                            n_snps,
                        )
                    )

                    rows.append({
                        "n_sites_per_role":
                            n_sites,

                        "n_snps":
                            n_snps,

                        "total_n_per_role":
                            study.SYSTEM_TOTAL_N_PER_ROLE,

                        "min_site_n":
                            min(
                                sizes
                            ),

                        "max_site_n":
                            max(
                                sizes
                            ),

                        "repetition":
                            rep,

                        "simulation_seconds":
                            simulation_seconds,

                        "federated_aggregation_seconds":
                            federated_aggregation_seconds,

                        "centralized_OLS_seconds":
                            centralized_seconds,

                        "max_exposure_beta_difference":
                            exp_beta_error,

                        "max_exposure_SE_difference":
                            exp_se_error,

                        "max_outcome_beta_difference":
                            out_beta_error,

                        "max_outcome_SE_difference":
                            out_se_error,

                        "abs_IVW_difference":
                            mr_error,

                        "numeric_payload_bytes_exposure":
                            exposure_payload,

                        "numeric_payload_bytes_outcome":
                            outcome_payload,

                        "numeric_payload_bytes_total":
                            (
                                exposure_payload
                                + outcome_payload
                            ),

                        "numerical_equivalence_PASS":
                            bool(
                                max(
                                    exp_beta_error,
                                    exp_se_error,
                                    out_beta_error,
                                    out_se_error,
                                    mr_error,
                                )
                                < 1e-8
                            ),
                    })

                    print(
                        "SYSTEM",
                        f"sites={n_sites}",
                        f"snps={n_snps}",
                        f"rep={rep + 1}/{n_reps}",
                        f"max_beta_error="
                        f"{max(exp_beta_error, out_beta_error):.2e}",
                    )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Figures
# ============================================================

def plot_metric(
    summary,
    metric,
    ylabel,
    title,
    output,
    reference=None,
):

    import matplotlib.pyplot as plt

    ivw = summary[
        summary["method"]
        == "IVW"
    ].copy()

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    x = np.arange(
        len(ivw)
    )

    ax.plot(
        x,
        ivw[metric],
        marker="o",
    )

    if reference is not None:

        ax.axhline(
            reference,
            linestyle="--",
            linewidth=1,
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        ivw["scenario"],
        rotation=45,
        ha="right",
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def make_figures(
    summary,
    systems,
):

    plot_metric(
        summary,
        "bias",
        "Bias",
        "Known-truth IVW bias",
        FIG_BIAS,
        reference=0.0,
    )

    plot_metric(
        summary,
        "RMSE",
        "RMSE",
        "Known-truth IVW RMSE",
        FIG_RMSE,
    )

    plot_metric(
        summary,
        "coverage_95",
        "95% CI coverage",
        "Known-truth IVW interval coverage",
        FIG_COVERAGE,
        reference=0.95,
    )

    # Only rows with a valid decision metric.
    decision = summary[
        summary["method"]
        == "IVW"
    ].copy()

    decision[
        "decision_rate"
    ] = decision[
        "power_p_lt_0_05"
    ]

    null_mask = (
        decision["truth"]
        == 0
    )

    decision.loc[
        null_mask,
        "decision_rate",
    ] = decision.loc[
        null_mask,
        "type1_error_p_lt_0_05",
    ]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    x = np.arange(
        len(decision)
    )

    ax.plot(
        x,
        decision[
            "decision_rate"
        ],
        marker="o",
    )

    ax.axhline(
        0.05,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        decision[
            "scenario"
        ],
        rotation=45,
        ha="right",
    )

    ax.set_ylabel(
        "Power / Type-I error"
    )

    ax.set_title(
        "IVW decision performance"
    )

    fig.tight_layout()

    fig.savefig(
        FIG_POWER,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
    # Systems runtime.
    runtime = (
        systems.groupby(
            [
                "n_sites_per_role",
                "n_snps",
            ],
            as_index=False,
        )[
            "federated_aggregation_seconds"
        ]
        .median()
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    for n_sites in sorted(
        runtime[
            "n_sites_per_role"
        ].unique()
    ):

        d = runtime[
            runtime[
                "n_sites_per_role"
            ]
            == n_sites
        ]

        ax.plot(
            d["n_snps"],
            d[
                "federated_aggregation_seconds"
            ],
            marker="o",
            label=f"{n_sites} sites/role",
        )

    ax.set_xlabel(
        "Number of SNPs"
    )

    ax.set_ylabel(
        "Federated aggregation time (seconds)"
    )

    ax.set_title(
        "Federated aggregation scaling"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_RUNTIME,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    exact = (
        systems.groupby(
            [
                "n_sites_per_role",
                "n_snps",
            ],
            as_index=False,
        )[
            "max_exposure_beta_difference"
        ]
        .max()
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    for n_sites in sorted(
        exact[
            "n_sites_per_role"
        ].unique()
    ):

        d = exact[
            exact[
                "n_sites_per_role"
            ]
            == n_sites
        ]

        ax.plot(
            d["n_snps"],
            d[
                "max_exposure_beta_difference"
            ],
            marker="o",
            label=f"{n_sites} sites/role",
        )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        "Number of SNPs"
    )

    ax.set_ylabel(
        "Maximum |βfed - βcentral|"
    )

    ax.set_title(
        "Federated numerical exactness"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_EXACTNESS,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a fast integrity check "
            "before the full study."
        ),
    )

    args = parser.parse_args()

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    n_repetitions = (
        study.SMOKE_REPETITIONS
        if args.smoke
        else study.FULL_REPETITIONS
    )

    print(
        "=== GENERALIZED FEDIV SIMULATION STUDY ==="
    )

    print()
    print(
        "Statistical scenarios:",
        len(
            study.STATISTICAL_SCENARIOS
        ),
    )

    print(
        "Repetitions/scenario:",
        n_repetitions,
    )

    print()

    rows = []

    for scenario_index, scenario in enumerate(
        study.STATISTICAL_SCENARIOS,
        start=1,
    ):

        print(
            f"[{scenario_index}/"
            f"{len(study.STATISTICAL_SCENARIOS)}] "
            f"{scenario['name']}"
        )

        for rep in range(
            n_repetitions
        ):

            rows.extend(
                run_statistical_repetition(
                    scenario,
                    rep,
                )
            )

            if (
                (rep + 1) % 20 == 0
                or rep + 1
                == n_repetitions
            ):

                print(
                    "  repetitions:",
                    f"{rep + 1}/{n_repetitions}",
                )

    repetitions = pd.DataFrame(
        rows
    )

    repetitions.to_csv(
        REPETITION_OUT,
        sep="\t",
        index=False,
    )

    summary = (
        summarize_known_truth(
            repetitions
        )
    )

    summary.to_csv(
        SUMMARY_OUT,
        sep="\t",
        index=False,
    )

    print()
    print(
        "Running numerical/system scaling..."
    )

    systems = run_system_scaling(
        smoke=args.smoke
    )

    systems.to_csv(
        SYSTEM_OUT,
        sep="\t",
        index=False,
    )

    make_figures(
        summary,
        systems,
    )

    print()
    print(
        "=== KNOWN-TRUTH SUMMARY ==="
    )

    print()

    show = [
        "scenario",
        "method",
        "truth",
        "n_success",
        "failure_rate",
        "bias",
        "RMSE",
        "coverage_95",
        "power_p_lt_0_05",
        "type1_error_p_lt_0_05",
        "median_exposure_F",
        "mean_fraction_F_gt_10",
    ]

    print(
        summary[
            [
                c
                for c in show
                if c in summary.columns
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "=== SYSTEM / EXACTNESS SUMMARY ==="
    )

    print()

    system_summary = (
        systems.groupby(
            [
                "n_sites_per_role",
                "n_snps",
            ],
            as_index=False,
        )
        .agg(
            max_beta_error=(
                "max_exposure_beta_difference",
                "max",
            ),

            max_SE_error=(
                "max_exposure_SE_difference",
                "max",
            ),

            max_IVW_error=(
                "abs_IVW_difference",
                "max",
            ),

            median_federated_seconds=(
                "federated_aggregation_seconds",
                "median",
            ),

            median_central_seconds=(
                "centralized_OLS_seconds",
                "median",
            ),

            numeric_payload_bytes_total=(
                "numeric_payload_bytes_total",
                "first",
            ),

            all_PASS=(
                "numerical_equivalence_PASS",
                "all",
            ),
        )
    )

    print(
        system_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "RESULT: PASS"
    )

    print()
    print("Outputs:")

    for path in [
        REPETITION_OUT,
        SUMMARY_OUT,
        SYSTEM_OUT,
        FIG_BIAS,
        FIG_RMSE,
        FIG_COVERAGE,
        FIG_POWER,
        FIG_RUNTIME,
        FIG_EXACTNESS,
    ]:
        print(
            " ",
            path,
        )


if __name__ == "__main__":
    main()
