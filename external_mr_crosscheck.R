#!/usr/bin/env Rscript

# Independent external MR-software QA check for the frozen ACOT1 3-IV analysis.
#
# This script deliberately uses the CRAN MendelianRandomization package rather
# than any FedIV MR implementation. The input table is already harmonized.
#
# IMPORTANT:
#   - model = "random" is explicit because MendelianRandomization defaults to
#     fixed effects when there are <= 3 variants.
#   - weights = "simple" matches first-order/NOME IVW weights.
#   - distribution = "normal" matches the frozen FedIV confidence interval.

input_path <- "results/real_federated/paper_real_mr/Table_real_mr_analysis_input.tsv"
output_path <- "results/real_federated/paper_real_mr/Table_real_mr_external_MR_crosscheck.tsv"

if (!requireNamespace("MendelianRandomization", quietly = TRUE)) {
  stop(
    "R package 'MendelianRandomization' is not installed.\n",
    "Install it first with:\n",
    "install.packages('MendelianRandomization', repos='https://cloud.r-project.org')"
  )
}

dat <- read.delim(
  input_path,
  header = TRUE,
  sep = "\t",
  stringsAsFactors = FALSE,
  check.names = FALSE
)

required <- c(
  "ID",
  "beta_exposure_OLS",
  "se_exposure_OLS",
  "beta_outcome",
  "se_outcome"
)

missing_cols <- setdiff(required, names(dat))
if (length(missing_cols) > 0) {
  stop("Missing required column(s): ", paste(missing_cols, collapse = ", "))
}

if (nrow(dat) != 3) {
  stop("Expected exactly 3 frozen instruments, found ", nrow(dat))
}

mr_in <- MendelianRandomization::mr_input(
  bx = dat$beta_exposure_OLS,
  bxse = dat$se_exposure_OLS,
  by = dat$beta_outcome,
  byse = dat$se_outcome,
  snps = dat$ID,
  exposure = "ACOT1 expression",
  outcome = "transformed triglycerides"
)

fit <- MendelianRandomization::mr_ivw(
  mr_in,
  model = "random",
  robust = FALSE,
  penalized = FALSE,
  weights = "simple",
  distribution = "normal",
  alpha = 0.05
)

external <- c(
  estimate = as.numeric(fit@Estimate),
  se = as.numeric(fit@StdError),
  ci_low = as.numeric(fit@CILower),
  ci_high = as.numeric(fit@CIUpper),
  p_value = as.numeric(fit@Pvalue),
  Q = as.numeric(fit@Heter.Stat[1])
)

# Frozen FedIV results from the canonical real analysis.
fediv <- c(
  estimate = 0.006237076589676015,
  se = 0.0022259073212311815,
  ci_low = 0.0018743784071388707,
  ci_high = 0.010599774772213159,
  p_value = 0.005078091365840435,
  Q = 3.00023087776235
)

difference <- external - fediv
tolerance <- 1e-10
passed <- abs(difference) <= tolerance

result <- data.frame(
  metric = names(external),
  FedIV = unname(fediv),
  external_MendelianRandomization = unname(external),
  difference = unname(difference),
  abs_difference = abs(unname(difference)),
  tolerance = tolerance,
  pass = unname(passed),
  stringsAsFactors = FALSE
)

write.table(
  result,
  file = output_path,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

cat("============================================================\n")
cat("FedIV external MR-software QA cross-check\n")
cat("============================================================\n")
cat("Package: MendelianRandomization ",
    as.character(utils::packageVersion("MendelianRandomization")), "\n", sep = "")
cat("Input:   ", input_path, "\n", sep = "")
cat("Model:   multiplicative random-effects IVW\n")
cat("Weights: simple / first-order\n")
cat("CI:      normal approximation, 95%\n\n")

print(result, row.names = FALSE, digits = 12)

cat("\nExternal heterogeneity output:\n")
cat("  Q       =", format(external["Q"], digits = 15), "\n")
cat("  Q p-val =", format(as.numeric(fit@Heter.Stat[2]), digits = 15), "\n")
cat("  RSE     =", format(as.numeric(fit@RSE), digits = 15), "\n")

cat("\nOutput written to:\n  ", output_path, "\n", sep = "")

if (!all(passed)) {
  stop("\nFAIL: external MR result does not match frozen FedIV result within tolerance.")
}

cat("\nPASS: external MR-software result matches the frozen FedIV IVW result.\n")