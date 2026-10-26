# Independent cross-check of PharmAgent's 2x2 crossover model and Welch parallel estimator.
#
# Crossover, per case (subject,sequence,period,treatment,Cmax; two rows per subject):
#   lm : log(Cmax) ~ sequence + sequence:subject + period + treatment   (all effects fixed; treatment levels R, T,
#        so the treatmentT coefficient is T - R on the log scale)
#        - estimate, SE, residual df and the 90% CI of treatmentT (confint, then 100*exp)
#        - sequential SS for sequence and subject(sequence) (anova), adjusted SS for period and treatment (drop1)
#   lme: nlme::lme(log(Cmax) ~ sequence + period + treatment, random = ~1 | subject, REML) - the mixed model;
#        for complete 2x2 data its treatment estimate and SE are expected to equal the fixed-effects ones.
# Parallel, per case (subject,treatment,Cmax): Welch t.test on log(Cmax) at 90% (df is Satterthwaite).
#
# Usage:  Rscript crosscheck_anova.R <dir>      reads cases.csv and <case>.csv; writes r_anova_results.csv
#         and r_anova_session.txt. Nothing outside <dir> is read or written; no packages are installed.

args <- commandArgs(trailingOnly = TRUE)
dir <- if (length(args)) args[1] else "."
cases <- read.csv(file.path(dir, "cases.csv"), stringsAsFactors = FALSE)
have_nlme <- requireNamespace("nlme", quietly = TRUE)
fmt <- function(x) sprintf("%.17g", x)

crossover_row <- function(case, d) {
  d$treatment <- factor(d$treatment, levels = c("R", "T"))
  d$period <- factor(d$period)
  d$sequence <- factor(d$sequence)
  d$subject <- factor(d$subject)
  d$y <- log(d$Cmax)
  fit <- lm(y ~ sequence + sequence:subject + period + treatment, data = d)
  s <- summary(fit)$coefficients["treatmentT", ]
  ci <- confint(fit, "treatmentT", level = 0.90)
  a <- anova(fit)
  dr <- drop1(fit, scope = ~ period + treatment)
  mse <- sum(residuals(fit)^2) / fit$df.residual
  ss_res <- sum(residuals(fit)^2)
  f_of <- function(ss, df) (ss / df) / mse
  ss_per <- dr["period", "Sum of Sq"]
  ss_form <- dr["treatment", "Sum of Sq"]
  ms_subj <- a["sequence:subject", "Mean Sq"]
  lme_est <- lme_se <- lme_df <- lme_bsd <- NA_real_
  if (have_nlme) {
    m <- nlme::lme(y ~ sequence + period + treatment, random = ~ 1 | subject, data = d, method = "REML")
    tt <- summary(m)$tTable["treatmentT", ]
    lme_est <- tt["Value"]; lme_se <- tt["Std.Error"]; lme_df <- tt["DF"]
    # a REML between-subject SD of 0 is the boundary case where the mixed model is not comparable (see be_model.py)
    lme_bsd <- as.numeric(nlme::VarCorr(m)["(Intercept)", "StdDev"])
  }
  data.frame(
    case = case, design = "crossover", n_tr = length(unique(d$subject[d$sequence == "TR"])),
    n_rt = length(unique(d$subject[d$sequence == "RT"])), df = fit$df.residual,
    estimate = fmt(s["Estimate"]), se = fmt(s["Std. Error"]), mse = fmt(mse),
    ci_lower_pct = fmt(100 * exp(ci[1])), ci_upper_pct = fmt(100 * exp(ci[2])),
    ss_sequence = fmt(a["sequence", "Sum Sq"]), ss_subject = fmt(a["sequence:subject", "Sum Sq"]),
    ss_period = fmt(ss_per), ss_formulation = fmt(ss_form), ss_residual = fmt(ss_res),
    f_sequence = fmt(a["sequence", "Mean Sq"] / ms_subj),
    p_sequence = fmt(pf(a["sequence", "Mean Sq"] / ms_subj, 1, a["sequence:subject", "Df"], lower.tail = FALSE)),
    f_period = fmt(f_of(ss_per, 1)), p_period = fmt(pf(f_of(ss_per, 1), 1, fit$df.residual, lower.tail = FALSE)),
    f_formulation = fmt(f_of(ss_form, 1)),
    p_formulation = fmt(pf(f_of(ss_form, 1), 1, fit$df.residual, lower.tail = FALSE)),
    lme_estimate = fmt(lme_est), lme_se = fmt(lme_se), lme_df = fmt(lme_df), lme_between_sd = fmt(lme_bsd),
    welch_df = NA_character_, stringsAsFactors = FALSE)
}

parallel_row <- function(case, d) {
  t <- log(d$Cmax[d$treatment == "T"]); r <- log(d$Cmax[d$treatment == "R"])
  w <- t.test(t, r, var.equal = FALSE, conf.level = 0.90)
  data.frame(
    case = case, design = "parallel", n_tr = length(t), n_rt = length(r), df = NA_real_,
    estimate = fmt(unname(w$estimate[1] - w$estimate[2])), se = fmt(unname(w$stderr)), mse = NA_character_,
    ci_lower_pct = fmt(100 * exp(w$conf.int[1])), ci_upper_pct = fmt(100 * exp(w$conf.int[2])),
    ss_sequence = NA_character_, ss_subject = NA_character_, ss_period = NA_character_,
    ss_formulation = NA_character_, ss_residual = NA_character_, f_sequence = NA_character_,
    p_sequence = NA_character_, f_period = NA_character_, p_period = NA_character_, f_formulation = NA_character_,
    p_formulation = NA_character_, lme_estimate = NA_character_, lme_se = NA_character_, lme_df = NA_character_,
    lme_between_sd = NA_character_, welch_df = fmt(unname(w$parameter)), stringsAsFactors = FALSE)
}

rows <- lapply(seq_len(nrow(cases)), function(i) {
  d <- read.csv(file.path(dir, paste0(cases$case[i], ".csv")), stringsAsFactors = FALSE)
  if (cases$design[i] == "crossover") crossover_row(cases$case[i], d) else parallel_row(cases$case[i], d)
})
write.csv(do.call(rbind, rows), file.path(dir, "r_anova_results.csv"), row.names = FALSE)
writeLines(c(R.version.string, paste("stats", as.character(packageVersion("stats"))),
             if (have_nlme) paste("nlme", as.character(packageVersion("nlme"))) else "nlme not available"),
           file.path(dir, "r_anova_session.txt"))
