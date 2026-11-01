# FedIV

FedIV is a methods-oriented federated Mendelian randomization (MR) research implementation for estimating exposure–genetic-variant associations across distributed individual-level cohorts without centralizing raw records, followed by downstream MR using public outcome summary statistics.

The project has two main methodological components:

1. **Exact non-private federation** using site-level regression sufficient statistics that reproduce pooled individual-level ordinary least-squares (OLS) association estimates to numerical precision.
2. **A bounded-influence private exposure-association release** using a frozen Huber-loss plus ridge estimator followed by one analytically calibrated Gaussian output perturbation of the joint coefficient vector.

The primary real-data case study uses MAGE/1000 Genomes individual-level exposure data and a public GLGC 2021 transformed triglyceride outcome. The real gene/instrument selection was performed using exposure information only and was frozen before outcome lookup.

## Repository status

The publication-facing codebase has completed reproducibility cleanup.

Current validation status:

- Python test suite: **140 passed**
- ACOT1 9,557-SNP federated-vs-centralized exactness rerun: **PASS**
- ACOT1 primary MR rerun: **PASS**
- ACOT1 Huber/federated equivalence and joint-DP utility rerun: **PASS**
- Huber/privacy publication outputs reproduced byte-for-byte in the cleanup check
- Independent `MendelianRandomization` R-package cross-check reproduces the primary IVW result to numerical precision

Historical method-development branches are not part of the publication-facing execution map.

## Scientific scope

FedIV is a computational/methodological MR project. It should not be interpreted as introducing federated regression, Huber regression, Gaussian output perturbation, or secure aggregation individually.

The intended contribution is the MR-specific integration of:

- exact federated exposure-association estimation,
- exposure-only instrument selection and public-outcome harmonization,
- transparent record-level privacy accounting for the released exposure coefficients,
- joint privacy accounting across an instrument panel, and
- empirical characterization of how privacy perturbation propagates into downstream MR estimates.

The ACOT1 → triglyceride analysis is a **methodological case study**, not a definitive biological causal discovery.

## Real-data cohort and final case study

Exposure cohort:

- MAGE/1000 Genomes individual-level data
- 729 individuals after the frozen exclusions
- covariates: sex and PC1–PC5
- five continental proxy sites:
  - AFR: 196
  - AMR: 113
  - EAS: 141
  - EUR: 140
  - SAS: 139

Final real gene:

- ACOT1
- Ensembl: `ENSG00000184227.8`

Final instruments:

- `14:73563814:T:A`
- `14:73583635:C:G`
- `14:73585292:C:A`

Each retained instrument satisfies the frozen strength and LD-independence criteria.

Primary public outcome:

- GLGC 2021 transformed triglyceride outcome
- phenotype: `logTG_INV`
- effect scale is the transformed outcome scale, not mg/dL

## Primary real MR result

Using exact federated OLS exposure associations and first-order IVW:

- estimate: approximately `0.0062370766`
- SE: approximately `0.0022259073`
- 95% CI: approximately `0.0018743` to `0.0105999`
- p-value: approximately `0.005078`

The result is heavily dominated by the first instrument, which contributes approximately 93.57% of first-order IVW information. Removing that instrument preserves a positive point estimate but substantially reduces precision.

## Exact federated association

For the selected ACOT1 cis region:

- 9,557 QC SNPs
- 729 individuals
- five federated sites

The cleanup rerun reproduced:

- max absolute beta difference between federated and centralized OLS: approximately `7.02e-14`
- max absolute SE difference: approximately `5.16e-15`

This confirms that the non-private sufficient-statistic federation reproduces the centralized pooled-regression target to numerical precision.

## Frozen privacy mechanism

The publication-facing real-data privacy method is fixed and should not be retuned.

Configuration:

- fixed-size replace-one record-level adjacency
- Huber loss parameter `tau = 0.25`
- ridge parameter `lambda = 0.01`
- `delta = 1e-6`
- epsilon grid: `{0.5, 1, 2, 5, 10}`
- expression bound: `[-3, 3]`
- genotype bound: `[0, 2]`
- sex bound: `[0, 1]`
- PC1–PC5 bounds: `[-0.20, 0.20]`
- normalized design: `[1, H, sex, PC1, ..., PC5]` with `H = G/2`
- normalized outcome: `y* = y/3`
- one joint Gaussian coefficient-vector release
- joint sensitivity grows with the square root of the panel size

The public outcome GWAS is not part of the private release.

### Trust boundary

The iterative federated Huber optimization uses site-local objective/gradient contributions.

The output-DP claim assumes that the optimization transcript is visible only to a trusted coordinator or is otherwise protected using secure aggregation/secure computation.

**This repository does not implement cryptographic secure aggregation.**

Once the private exposure coefficient vector is released, downstream MR calculations are post-processing and do not consume additional privacy budget.

Monte Carlo draws used to characterize mechanism utility are simulations of the one-release mechanism distribution. They are not repeated releases of the same protected dataset and should not be described as DP-aware confidence intervals.

## EIF2S1 correctness benchmark

EIF2S1 is retained as a real individual-level validation benchmark for:

- centralized vs federated association correctness,
- five-site and 26-population federation checks, and
- supporting Huber-method validation.

EIF2S1 is not the main multi-instrument biological MR case study.

Retained outputs are under:

```text
results/real_federated/
```

The primary ACOT1 publication outputs are under:

```text
results/real_federated/paper_real_mr/
```

## Simulation studies

Two publication-facing synthetic studies are retained.

### Known-truth MR study

Runner:

```bash
python run_simulation_study.py
```

The completed publication experiment used 200 repetitions per scenario and evaluates statistical failure modes including sample size, weak instruments, pleiotropy, and heterogeneity.

Publication outputs are stored under:

```text
results/simulations/paper/
```

### Synthetic Huber privacy operating-boundary study

Runner:

```bash
python run_simulation_huber_privacy.py
```

Frozen full grid:

- N per role: `{729, 2500, 7500}`
- panel size: `{3, 5, 10, 20}`
- epsilon: `{0.5, 1, 2, 5, 10}`
- delta: `1e-6`
- 100 datasets per N × panel-size cell
- 50 internal DP draws per dataset/epsilon

These full experiments are already completed. They do not need to be rerun simply to verify the repository unless an integrity problem is discovered.

## Quick validation

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
```

Expected cleanup validation:

```text
140 passed
```

Real ACOT1 federated exactness:

```bash
python real_individual_data/run_selected_gene_federated_validation.py
```

Primary real MR analysis:

```bash
python real_individual_data/run_real_mr_analysis.py
```

Frozen Huber/privacy analysis:

```bash
python real_individual_data/run_selected_instrument_huber_privacy.py
```

Independent MR implementation cross-check:

```bash
Rscript external_mr_crosscheck.R
```

For the full publication execution map, see:

```text
reproducibility/execution_map.md
```

## Environment

The cleanup environment used:

- Python 3.10.14
- R 4.4.1
- `MendelianRandomization` 0.10.0

Frozen environment records are stored in:

```text
reproducibility/environment.txt
reproducibility/requirements-lock.txt
reproducibility/R-session-info.txt
reproducibility/source_files.txt
```

## Data and provenance

The local research workspace contains source datasets and frozen provenance needed for reproducibility testing.

MAGE provenance is stored under:

```text
data/mage/provenance/
```

The upstream MAGE repository retained inside the provenance directory is preserved as upstream provenance material and is not treated as FedIV-owned source code.

Before public Git release, each large or human-genomic data asset should be reviewed for:

- redistribution permission,
- file size,
- repository-hosting suitability, and
- whether a download/provenance instruction should replace redistribution.

Do not delete the local research datasets merely to prepare the codebase. Public packaging should be handled separately from the local reproducible research workspace.

## Reproducibility files

Publication-facing reproducibility materials:

```text
reproducibility/environment.txt
reproducibility/requirements-lock.txt
reproducibility/R-session-info.txt
reproducibility/source_files.txt
reproducibility/execution_map.md
```

The final manuscript checksum/manifest inventory should be regenerated only after the documentation and repository layout are finalized.

## Historical analyses

Historical exploratory privacy estimators and other superseded development branches have been moved out of the publication-facing project into a private archive.

The historical LDL → Alzheimer’s disease summary-data analysis is intentionally excluded from the final FedIV manuscript and publication-facing repository because it is not part of the final federated/private MR story.

## Citation

Citation information will be added when the manuscript is finalized.

## License

A repository license has not yet been finalized. Add the chosen license before public release.
