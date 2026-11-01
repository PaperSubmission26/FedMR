# FedIV canonical execution map

This file lists the active publication-facing execution paths for FedIV.
Historical or rejected method-development branches are not part of this map.

## 1. Core exact federated association engine

Core modules:
- `multicov_federated.py` — multi-covariate sufficient-statistic federation.
- `federated_aggregator.py` — synthetic/simple federated aggregation utilities.
- `mr_estimators.py` — IVW, MR-Egger, and weighted-median estimators.
- `heterogeneity_check.py` — heterogeneity diagnostics.
- `instrument_strength_check.py` — instrument-strength diagnostics.
- `sensitivity_analysis.py` — MR sensitivity analyses.

Validation test:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
```

Expected repository-cleanup validation: `140 passed`.

## 2. Real individual-level correctness benchmark: EIF2S1

Five-site correctness:

```bash
python real_individual_data/run_real_federated_validation.py
```

Twenty-six-population correctness:

```bash
python real_individual_data/run_real_26pop_validation.py
```

Key retained outputs:
- `results/real_federated/EIF2S1_centralized_associations.tsv`
- `results/real_federated/EIF2S1_federated_associations.tsv`
- `results/real_federated/EIF2S1_federated_vs_centralized_summary.txt`
- `results/real_federated/EIF2S1_26pop_federated_associations.tsv`
- `results/real_federated/EIF2S1_26pop_federated_vs_centralized_summary.txt`

Purpose: real individual-level validation that federated sufficient-statistic regression reproduces centralized regression. EIF2S1 is not the primary multi-instrument biological MR case study.

## 3. Exposure-only real gene/instrument selection

Relevant scripts:
- `real_individual_data/build_real_mr_candidate_annotation.py`
- `real_individual_data/rank_real_mr_candidate_genes.py`
- `real_individual_data/run_real_mr_candidate_screen_sequence.py`
- `real_individual_data/screen_real_mr_candidate.py`
- `real_individual_data/audit_selected_real_mr_exposure.py`

Frozen selection: ACOT1 (`ENSG00000184227.8`) with three strong LD-independent cis instruments.

Selection was exposure-only and outcome-blind.

## 4. ACOT1 full federated exactness

Canonical command:

```bash
python real_individual_data/run_selected_gene_federated_validation.py
```

Expected key values:
- n = 729
- 9,557 QC cis SNPs
- max |beta_federated - beta_centralized| ≈ 7.02494e-14
- max |SE_federated - SE_centralized| ≈ 5.16254e-15
- three frozen instruments reproduce centralized values to numerical precision

Primary outputs:
- `results/real_federated/paper_real_mr/Table_real_mr_selected_gene_centralized_associations.tsv`
- `results/real_federated/paper_real_mr/Table_real_mr_selected_gene_federated_exactness.tsv`
- `results/real_federated/paper_real_mr/Table_real_mr_selected_gene_federated_exactness_summary.tsv`
- `results/real_federated/paper_real_mr/Table_real_mr_selected_instrument_federated_exactness.tsv`
- `results/real_federated/paper_real_mr/Figure_real_mr_selected_gene_federated_exactness.png`

## 5. Frozen real Huber/output-DP analysis

Canonical command:

```bash
python real_individual_data/run_selected_instrument_huber_privacy.py
```

Frozen configuration:
- Huber tau = 0.25
- ridge lambda = 0.01
- delta = 1e-6
- epsilon = {0.5, 1, 2, 5, 10}
- one joint coefficient-vector release
- joint sensitivity scales with sqrt(m)

Primary outputs:
- `results/real_federated/paper_real_mr/Table_real_mr_selected_instrument_huber_equivalence.tsv`
- `results/real_federated/paper_real_mr/Table_real_mr_selected_instrument_huber_equivalence_summary.tsv`
- `results/real_federated/paper_real_mr/Table_real_mr_selected_instrument_joint_DP_detail.tsv`
- `results/real_federated/paper_real_mr/Table_real_mr_selected_instrument_joint_DP_utility.tsv`
- `results/real_federated/paper_real_mr/Figure_real_mr_selected_instrument_joint_DP_utility.png`

The iterative Huber transcript is assumed to be visible only to a trusted coordinator or protected by secure aggregation/secure computation. FedIV does not implement cryptographic secure aggregation.

## 6. Public triglyceride outcome extraction/harmonization

Relevant script:

```bash
python real_individual_data/extract_frozen_outcome_associations.py
```

This operates on the frozen public GLGC 2021 transformed triglyceride outcome (`logTG_INV`) after the exposure gene/instruments have already been frozen.

Key output:
- `results/real_federated/paper_real_mr/Table_real_mr_harmonized_outcome_associations.tsv`

## 7. Primary real ACOT1 -> triglyceride MR analysis

Canonical command:

```bash
python real_individual_data/run_real_mr_analysis.py
```

Expected primary result:
- IVW estimate ≈ 0.0062370766
- SE ≈ 0.0022259073
- 95% CI ≈ 0.0018743 to 0.0105999
- p ≈ 0.005078
- three instruments
- approximately 93.57% of first-order IVW information from instrument 1

Primary outputs:
- `results/real_federated/paper_real_mr/Table_real_mr_analysis_input.tsv`
- `results/real_federated/paper_real_mr/Table_real_mr_results.tsv`
- `results/real_federated/paper_real_mr/Table_real_mr_Wald_ratios.tsv`
- `results/real_federated/paper_real_mr/Table_real_mr_diagnostics.tsv`
- `results/real_federated/paper_real_mr/Table_real_mr_leave_one_out.tsv`
- `results/real_federated/paper_real_mr/Table_real_mr_DP_distortion.tsv`
- `results/real_federated/paper_real_mr/Figure_real_mr_forest.png`
- `results/real_federated/paper_real_mr/Figure_real_mr_DP_distortion.png`

Interpretation: this is a methodological case study, not a definitive biological causal discovery. The outcome is inverse-normalized log triglyceride, not mg/dL.

## 8. Real DP tail-stability audit

Canonical script:

```bash
python real_individual_data/audit_real_mr_dp_tail_stability.py
```

Primary outputs:
- `results/real_federated/paper_real_mr/Table_real_mr_DP_tail_stability.tsv`
- `results/real_federated/paper_real_mr/Table_real_mr_DP_tail_stability_batches.tsv`
- `results/real_federated/paper_real_mr/Figure_real_mr_DP_tail_RMSE.png`
- `results/real_federated/paper_real_mr/Figure_real_mr_DP_denominator_stability.png`

Mechanism-distribution quantiles are utility summaries, not DP-aware confidence intervals.

## 9. Independent external MR implementation check

Canonical command:

```bash
Rscript external_mr_crosscheck.R
```

External software:
- R 4.4.1
- `MendelianRandomization` 0.10.0

The external implementation reproduces the frozen primary IVW estimate, standard error, confidence interval, p-value, and Cochran Q to numerical precision.

Output:
- `results/real_federated/paper_real_mr/Table_real_mr_external_MR_crosscheck.tsv`

## 10. Known-truth synthetic MR study

Canonical runner:

```bash
python run_simulation_study.py
```

Publication outputs:
- `results/simulations/paper/Table_simulation_known_truth_repetitions.tsv`
- `results/simulations/paper/Table_simulation_known_truth_summary.tsv`
- `results/simulations/paper/Table_simulation_system_scaling.tsv`
- associated publication figures in `results/simulations/paper/`

The completed publication run used 200 repetitions per scenario.

The full experiment does not need to be rerun merely for repository verification; frozen outputs and checksums are the publication record.

## 11. Synthetic Huber privacy operating-boundary study

Canonical runner:

```bash
python run_simulation_huber_privacy.py
```

Frozen grid:
- N per role = {729, 2500, 7500}
- panel size m = {3, 5, 10, 20}
- epsilon = {0.5, 1, 2, 5, 10}
- delta = 1e-6
- 100 synthetic datasets per N x m cell
- 50 DP draws per dataset/epsilon

Publication outputs:
- `results/simulations/paper/Table_simulation_huber_privacy_repetitions.tsv`
- `results/simulations/paper/Table_simulation_huber_privacy_nonprivate_summary.tsv`
- `results/simulations/paper/Table_simulation_huber_privacy_summary.tsv`
- `results/simulations/paper/Table_simulation_huber_federated_exactness.tsv`

The completed full grid should not be rerun unless an integrity problem is identified.

## 12. Reproducibility files

- `reproducibility/environment.txt`
- `reproducibility/requirements-lock.txt`
- `reproducibility/source_files.txt`
- `reproducibility/R-session-info.txt`
- `reproducibility/execution_map.md`

The upstream MAGE repository retained under `data/mage/provenance/MAGE_repository/` is provenance material and is not listed as FedIV-owned source code.

## Scope note

The historical LDL -> Alzheimer’s disease analysis is intentionally excluded from the publication-facing FedIV repository and manuscript because it is not part of the final federated/private MR story.
