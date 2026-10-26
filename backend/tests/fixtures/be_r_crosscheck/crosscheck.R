# Independent cross-check of PharmAgent's INTERIM average-BE estimator with base R (stats::t.test).
#   crossover: paired t.test on log(Cmax), test minus reference, matched by subject
#   parallel : two-sample t.test(var.equal = TRUE) on log(Cmax)
# 90% two-sided CI on the log scale, back-transformed to percent. Validates the interim estimator only; it does not
# establish regulatory correctness (no period/sequence/formulation model).
# Usage: Rscript crosscheck.R <dir>   (reads cases.csv and <case>.csv, writes r_results.csv)
args <- commandArgs(trailingOnly = TRUE)
dir <- if (length(args)) args[1] else "."
cases <- read.csv(file.path(dir, "cases.csv"), stringsAsFactors = FALSE)
out <- do.call(rbind, lapply(seq_len(nrow(cases)), function(i) {
  cs <- cases[i, ]
  d <- read.csv(file.path(dir, paste0(cs$case, ".csv")), stringsAsFactors = FALSE)
  t <- d[d$treatment == "T", ]; r <- d[d$treatment == "R", ]
  if (cs$design == "crossover") {
    r <- r[match(t$subject, r$subject), ]
    tt <- t.test(log(t$Cmax), log(r$Cmax), paired = TRUE, conf.level = 0.90)
  } else {
    tt <- t.test(log(t$Cmax), log(r$Cmax), var.equal = TRUE, conf.level = 0.90)
  }
  data.frame(case = cs$case, design = cs$design, n_test = nrow(t), n_reference = nrow(r),
             gmr_pct = sprintf("%.17g", 100 * exp(unname(tt$estimate[1]) - if (cs$design == "parallel") unname(tt$estimate[2]) else 0)),
             ci_lower_pct = sprintf("%.17g", 100 * exp(tt$conf.int[1])),
             ci_upper_pct = sprintf("%.17g", 100 * exp(tt$conf.int[2])),
             df = unname(tt$parameter))
}))
write.csv(out, file.path(dir, "r_results.csv"), row.names = FALSE)
writeLines(c(R.version.string, paste("stats", as.character(packageVersion("stats")))), file.path(dir, "r_session.txt"))
