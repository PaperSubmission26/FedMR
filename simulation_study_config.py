"""
Generalized simulation-study configuration for FedIV.

This file DOES NOT replace config.py.

The legacy config.py remains the historical/default configuration.
run_simulation_study.py temporarily overrides selected config attributes
for each scenario and restores them after every run.
"""


# ============================================================
# Repetition settings
# ============================================================

BASE_SEED = 50000

# Use --smoke first.
SMOKE_REPETITIONS = 5

# First full publication-quality pass.
# Can later be increased mechanically if Monte Carlo SEs require it.
FULL_REPETITIONS = 200


# ============================================================
# Common baseline
# ============================================================

BASE = {
    "N_SNPS": 50,

    "TRUE_CAUSAL_EFFECT": 0.30,

    "MAF_RANGE": (0.05, 0.50),

    "BETA_GX_MEAN": 0.20,
    "BETA_GX_SD": 0.05,

    "WEAK_INSTRUMENT_FRACTION": 0.0,
    "WEAK_INSTRUMENT_SCALE": 0.10,

    "PLEIOTROPY_FRACTION": 0.0,
    "PLEIOTROPY_MEAN": 0.0,
    "PLEIOTROPY_SD": 0.08,

    "SITE_HETEROGENEITY_SD": 0.0,

    "COVARIATE_MEAN": 0.0,
    "COVARIATE_SD": 1.0,
    "COVARIATE_EFFECT_ON_EXPOSURE": 0.15,
    "COVARIATE_EFFECT_ON_OUTCOME": 0.15,

    "EXPOSURE_NOISE_SD": 1.0,
    "OUTCOME_NOISE_SD": 1.0,

    # Balanced baseline: 5 exposure and 5 outcome sites,
    # total N=2500 on each side.
    "EXPOSURE_SITE_SIZES": [500] * 5,
    "OUTCOME_SITE_SIZES": [500] * 5,
}


# ============================================================
# Known-truth statistical scenarios
#
# Targeted rather than a huge Cartesian product.
# ============================================================

STATISTICAL_SCENARIOS = [

    {
        "name": "clean_baseline",
    },

    # --------------------------------------------------------
    # Sample-size axis
    # --------------------------------------------------------

    {
        "name": "sample_small",
        "EXPOSURE_SITE_SIZES": [150] * 5,
        "OUTCOME_SITE_SIZES": [150] * 5,
    },

    {
        "name": "sample_large",
        "EXPOSURE_SITE_SIZES": [1500] * 5,
        "OUTCOME_SITE_SIZES": [1500] * 5,
    },

    # --------------------------------------------------------
    # Weak-instrument axis
    # --------------------------------------------------------

    {
        "name": "weak_moderate",
        "WEAK_INSTRUMENT_FRACTION": 0.25,
    },

    {
        "name": "weak_severe",
        "WEAK_INSTRUMENT_FRACTION": 0.50,
    },

    # --------------------------------------------------------
    # Pleiotropy axis
    # --------------------------------------------------------

    {
        "name": "pleiotropy_balanced",
        "PLEIOTROPY_FRACTION": 0.25,
        "PLEIOTROPY_MEAN": 0.0,
        "PLEIOTROPY_SD": 0.08,
    },

    {
        "name": "pleiotropy_directional",
        "PLEIOTROPY_FRACTION": 0.25,
        "PLEIOTROPY_MEAN": 0.06,
        "PLEIOTROPY_SD": 0.04,
    },

    # --------------------------------------------------------
    # Causal heterogeneity across outcome sites
    # --------------------------------------------------------

    {
        "name": "heterogeneity_moderate",
        "SITE_HETEROGENEITY_SD": 0.10,
    },

    {
        "name": "heterogeneity_severe",
        "SITE_HETEROGENEITY_SD": 0.30,
    },

    # --------------------------------------------------------
    # Unequal federation, same total sample size
    # --------------------------------------------------------

    {
        "name": "unbalanced_sites",
        "EXPOSURE_SITE_SIZES": [
            900,
            600,
            450,
            350,
            200,
        ],
        "OUTCOME_SITE_SIZES": [
            800,
            650,
            500,
            300,
            250,
        ],
    },

    # --------------------------------------------------------
    # Combined stress condition
    # --------------------------------------------------------

    {
        "name": "combined_stress",
        "WEAK_INSTRUMENT_FRACTION": 0.25,
        "PLEIOTROPY_FRACTION": 0.25,
        "PLEIOTROPY_MEAN": 0.06,
        "PLEIOTROPY_SD": 0.04,
        "SITE_HETEROGENEITY_SD": 0.10,
    },

    # --------------------------------------------------------
    # Null calibration
    # --------------------------------------------------------

    {
        "name": "null_clean",
        "TRUE_CAUSAL_EFFECT": 0.0,
    },

    {
        "name": "null_directional_pleiotropy",
        "TRUE_CAUSAL_EFFECT": 0.0,
        "PLEIOTROPY_FRACTION": 0.25,
        "PLEIOTROPY_MEAN": 0.06,
        "PLEIOTROPY_SD": 0.04,
    },
]


# ============================================================
# Numerical / systems scaling
#
# Total sample size is held fixed at N=4000 per role while site
# count changes, so site scaling is not confounded with total N.
# ============================================================

SYSTEM_SITE_COUNTS = [
    2,
    5,
    10,
    20,
]

SYSTEM_SNP_COUNTS = [
    10,
    50,
    100,
    500,
]

SYSTEM_TOTAL_N_PER_ROLE = 4000

SYSTEM_REPETITIONS = 3


# ============================================================
# Estimator settings
# ============================================================

ALPHA = 0.05

# Weighted median uses an internal parametric bootstrap.
# 1000 is sufficient for this first large Monte Carlo grid.
# We can increase only for the final selected summary rows if needed.
WEIGHTED_MEDIAN_BOOTSTRAPS = 1000
