#!/usr/bin/env Rscript
# PKNCA reference run for the PharmAgent cross-check.   YOU run this (it is not run automatically):
#
#   Rscript scripts/crosscheck_pknca/run_pknca.R scripts/crosscheck_pknca/nca_input.csv scripts/crosscheck_pknca/out
#
# Reads the table made by make_nca_input.py (ID, TIME, DV, AMT, EVID, BLQ) and writes one CSV per configuration
# plus pknca_versions.txt. It installs nothing and uses no network.
#
# Configurations (the point of the exercise is to see how much the BLQ / lambda_z conventions matter):
#   pknca_default : PKNCA's own defaults (conc.blq first=keep middle=drop last=keep, adj.r.squared.factor=1e-4)
#   pknca_matched : set to mimic PharmAgent's documented rules: linear-up/log-down, zeros kept as zeros,
#                   strict maximum adjusted R^2 (PKNCA needs a factor in (0,1), so 1e-12 stands for 0),
#                   >= 3 points, Tmax not in the terminal fit

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: run_pknca.R <nca_input.csv> <outdir>")
input <- args[[1]]; outdir <- args[[2]]
if (!requireNamespace("PKNCA", quietly = TRUE)) stop("PKNCA is not installed (this script does not install packages)")
suppressPackageStartupMessages(library(PKNCA))
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

d <- read.csv(input, stringsAsFactors = FALSE)
stopifnot(all(c("ID", "TIME", "DV", "AMT", "EVID") %in% names(d)))
conc <- d[d$EVID == 0, c("ID", "TIME", "DV")]
dose <- d[d$EVID == 1, c("ID", "TIME", "AMT")]
conc <- conc[!is.na(conc$DV), ]                       # a blank concentration is a missing sample, never a zero
conc <- conc[order(conc$ID, conc$TIME), ]

intervals <- data.frame(start = 0, end = Inf,
                        cmax = TRUE, tmax = TRUE, tlast = TRUE, clast.obs = TRUE,
                        auclast = TRUE, aucinf.obs = TRUE,
                        lambda.z = TRUE, half.life = TRUE,
                        lambda.z.n.points = TRUE, r.squared = TRUE, adj.r.squared = TRUE)

configs <- list(
  pknca_default = list(),
  pknca_matched = list(auc.method = "lin up/log down",
                       conc.blq = list(first = "keep", middle = "keep", last = "keep"),
                       adj.r.squared.factor = 1e-12, min.hl.points = 3, allow.tmax.in.half.life = FALSE)
)

for (tag in names(configs)) {
  o_conc <- PKNCAconc(conc, DV ~ TIME | ID)
  o_dose <- PKNCAdose(dose, AMT ~ TIME | ID, route = "extravascular")
  o_data <- PKNCAdata(o_conc, o_dose, intervals = intervals, options = configs[[tag]])
  # pk.nca() results come back as a tibble, which base reshape() mishandles: make it a plain data.frame first
  res <- data.frame(as.data.frame(pk.nca(o_data)), stringsAsFactors = FALSE)
  wide <- reshape(res[, c("ID", "PPTESTCD", "PPORRES")], idvar = "ID", timevar = "PPTESTCD",
                  v.names = "PPORRES", direction = "wide")
  names(wide) <- sub("^PPORRES\\.", "", names(wide))
  # PKNCA's lower-case parameter names -> the column names compare.py expects
  rename <- c(cmax = "Cmax", tmax = "Tmax", auclast = "AUClast", aucinf.obs = "AUCinf.obs")
  hit <- names(wide) %in% names(rename)
  names(wide)[hit] <- rename[names(wide)[hit]]
  rownames(wide) <- NULL
  write.csv(wide, file.path(outdir, paste0(tag, ".csv")), row.names = FALSE)
  cat(sprintf("%s: %d subjects -> %s\n", tag, nrow(wide), file.path(outdir, paste0(tag, ".csv"))))
}
writeLines(c(paste("PKNCA", as.character(packageVersion("PKNCA"))), R.version.string,
             paste("run", format(Sys.time(), tz = "UTC", usetz = TRUE))),
           file.path(outdir, "pknca_versions.txt"))
