"""
Central config for the federated MR pipeline. Hardcoded on purpose —
no argparse. Edit values directly and rerun.
"""

import numpy as np

# ---- reproducibility ----
RANDOM_SEED = 42

# ---- instrument panel ----
N_SNPS = 50
MAF_RANGE = (0.05, 0.50)          # minor allele frequency, drawn per SNP

# ---- true causal model (ground truth the pipeline must recover) ----
TRUE_CAUSAL_EFFECT = 0.30         # beta: exposure -> outcome

# SNP -> exposure effect sizes are drawn once and shared by both
# exposure- and outcome-side generative models (two-sample MR).
BETA_GX_MEAN = 0.20
BETA_GX_SD = 0.05

# ---- weak instruments ----
WEAK_INSTRUMENT_FRACTION = 0.10   # fraction of SNPs made "weak"
WEAK_INSTRUMENT_SCALE = 0.10      # multiply beta_gx by this for weak SNPs

# ---- pleiotropy (direct SNP -> outcome effect, bypassing exposure) ----
PLEIOTROPY_FRACTION = 0.10        # fraction of SNPs with a direct effect
PLEIOTROPY_MEAN = 0.00            # balanced pleiotropy by default
PLEIOTROPY_SD = 0.08              # set nonzero mean for directional (Egger) case

# ---- site-level heterogeneity ----
# Small per-site perturbation of the true causal effect, so that
# site-level heterogeneity diagnostics have something real to detect.
# Set to 0.0 for a homogeneous baseline run.
SITE_HETEROGENEITY_SD = 0.0

# ---- covariate (single confounder-like covariate, e.g. age, standardized) ----
COVARIATE_MEAN = 0.0
COVARIATE_SD = 1.0
COVARIATE_EFFECT_ON_EXPOSURE = 0.15
COVARIATE_EFFECT_ON_OUTCOME = 0.15

# ---- noise ----
EXPOSURE_NOISE_SD = 1.0
OUTCOME_NOISE_SD = 1.0

# ---- sites (Design B: separate exposure-sites and outcome-sites) ----
# Unequal site sizes on purpose, to stress-test the aggregator later.
EXPOSURE_SITE_SIZES = [800, 500, 1200, 300]
OUTCOME_SITE_SIZES = [900, 400, 700, 1000, 250]

N_EXPOSURE_SITES = len(EXPOSURE_SITE_SIZES)
N_OUTCOME_SITES = len(OUTCOME_SITE_SIZES)

# ---- numerical tolerance for the Phase 2 correctness proof ----
CORRECTNESS_ATOL = 1e-8
CORRECTNESS_RTOL = 1e-6

# ---- Phase 3: differential privacy ----
# Record-level adjacency: neighboring datasets differ by one individual's
# (G, X, trait) row. Every quantity a site releases must be a function of
# CLIPPED inputs — DP noise calibrated to an unbounded statistic's
# sensitivity is not a real guarantee (see dp_mechanism.py).
#
# Genotype dosage G is already naturally bounded in [0, 2] (0/1/2 copies
# of the effect allele) — no clipping needed there. X and the trait
# (exposure or outcome) are clipped to the ranges below before any
# sufficient statistic is computed from them.
CLIP_G = (0.0, 2.0)
CLIP_X = (-4.0, 4.0)          # covariate is ~N(0,1) in simulation; +-4 SD covers it, clip if real data differs
CLIP_TRAIT = (-8.0, 8.0)      # exposure/outcome noise SD ~1, causal paths add a few units; generous margin

# default privacy budget for a single non-swept DP release (used by
# run_dp_pipeline.py's non-sweep entry point; the epsilon sweep for
# plotting the privacy-utility curve is configured separately there).
# default privacy budget for a single non-swept DP release (used by
# dp_mechanism.py's __main__ demo; the epsilon sweep for plotting the
# full privacy-utility curve, including much smaller/stricter epsilon
# default privacy budget for a single non-swept DP release (used by
# dp_mechanism.py's __main__ demo; the epsilon sweep for plotting the
# full privacy-utility curve, including much smaller/stricter epsilon
# values, is a separate later script).
#
# WHY THIS DEFAULT IS THIS LARGE: at N_SNPS=50 with 9 privatized fields
# under basic composition, using the CORRECTLY calibrated analytic
# Gaussian mechanism (valid for all epsilon > 0 — see
# dp_mechanism.analytic_gaussian_sigma), per-SNP identifiability needs
# roughly epsilon >= 300 for this config to reliably recover the true
# effect. This number went UP from an earlier, incorrect version of this
# file that used the classical closed-form Gaussian calibration
# (sigma = sensitivity*sqrt(2ln(1.25/delta))/epsilon) outside its proven
# valid range (epsilon <= 1) — that formula silently UNDERSTATES the
# noise needed once per-field epsilon exceeds ~1 (which happens for
# every one of the sweep's total-epsilon values except the smallest),
# so results computed under it were not actually private at the claimed
# epsilon. This default is corrected, not just re-tuned to "look good".
# This IS a weak privacy guarantee — real epsilon this large gives very
# little protection. Tightening clip bounds to the data's actual range
# and/or using a tighter composition accountant (advanced
# composition/zCDP instead of basic composition) would both lower the
# epsilon needed for the same identifiability — both flagged as future
# refinements in dp_mechanism.py, not yet implemented.
DP_EPSILON = 500.0
DP_DELTA = 1e-6

# Study-level total budget ceiling — distinct from DP_EPSILON/DP_DELTA
# above (which are PER-SITE, PER-RUN values). Per the problem statement's
# "Composition" commitment, real deployments must cap the TOTAL privacy
# spent across every release of the SAME protected dataset over the
# whole study (repeated analyses, diagnostics, re-runs). This does NOT
# apply to privacy_budget_sweep.py's synthetic-data sweep (explicitly
# exempt — see that file's module docstring) but DOES apply to any
# future real-data case study (Phase 6): every re-privatization of the
# same real dataset must be charged against this ceiling, not just
# against a fresh per-run budget as if each run were an independent
# dataset.
TOTAL_STUDY_EPSILON_BUDGET = 2000.0
TOTAL_STUDY_DELTA_BUDGET = 1e-4

# analytic Gaussian mechanism, so DELTA must be > 0 (pure epsilon-DP via
# Laplace is also supported per-scalar-query in dp_mechanism.py, but the
# vector-valued per-SNP releases here use the Gaussian mechanism, which
# needs a delta slack term).
assert 0 < DP_DELTA < 1, "DP_DELTA must be in (0, 1) for the analytic Gaussian mechanism."


def get_rng(stream_name: str) -> np.random.Generator:
    """
    Deterministic, independent RNG stream per named use (e.g. 'snp_panel',
    'exposure_site_0', 'outcome_site_2'). Keeps site simulations
    reproducible and independent of call order.

    Uses SHA-256 of the stream name (not Python's builtin hash()) because
    hash() is randomized per-process via PYTHONHASHSEED — two separate
    Python processes would derive different seeds for the same
    stream_name and silently break reproducibility. SHA-256 is stable
    across processes, machines, and Python versions.

    Scope of the reproducibility guarantee: this gives IDENTICAL draws
    across processes/machines for a PINNED numpy version, since the
    guarantee also depends on np.random.Generator/SeedSequence's PCG64
    bit-generator algorithm, which NumPy documents as stable but has not
    committed to as a permanent contract across all future major
    versions. Pin numpy in requirements.txt/environment lockfiles for any
    run whose exact numeric output must be reproduced later.
    """
    import hashlib
    digest = hashlib.sha256(stream_name.encode("utf-8")).digest()
    stream_hash = int.from_bytes(digest[:4], byteorder="big") % (2 ** 31)
    seed_seq = np.random.SeedSequence([RANDOM_SEED, stream_hash])
    return np.random.default_rng(seed_seq)