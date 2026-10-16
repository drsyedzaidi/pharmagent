#!/usr/bin/env Rscript
# mrgsolve reference simulation for the PharmAgent cross-check.   YOU run this (it is not run automatically):
#
#   Rscript scripts/crosscheck_mrgsolve/run_mrgsolve.R scripts/crosscheck_mrgsolve scripts/crosscheck_mrgsolve/out
#
# Reads scenarios.csv / doses.csv / obs.csv (made by make_scenarios.py) and writes mrgsolve_sim.csv
# (id, time, cp) plus mrgsolve_versions.txt. Installs nothing, uses no network. It compiles ten small
# mrgsolve models (needs the C++ toolchain mrgsolve already needs).
#
# Every model below is written FROM THE EQUATIONS of the PharmAgent registry entry named in its comment
# (app/compute/pk_models.py), not generated from it. Weight scaling is implemented here independently from the
# stated convention (allometric, centred at 70 kg: clearances, inter-compartmental clearances and VMAX ^0.75,
# volumes ^1, everything else unscaled), so a disagreement shows where the registry departs from that convention.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: run_mrgsolve.R <scenario dir> <outdir>")
indir <- args[[1]]; outdir <- args[[2]]
if (!requireNamespace("mrgsolve", quietly = TRUE)) stop("mrgsolve is not installed (this script does not install packages)")
suppressPackageStartupMessages(library(mrgsolve))
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

scen  <- read.csv(file.path(indir, "scenarios.csv"), stringsAsFactors = FALSE)
doses <- read.csv(file.path(indir, "doses.csv"),     stringsAsFactors = FALSE)
obs   <- read.csv(file.path(indir, "obs.csv"),       stringsAsFactors = FALSE)

code <- list(
  # iv_1cmt
  iv_1cmt = '
$PARAM CL = 5, V = 50
$CMT CENT
$ODE dxdt_CENT = -(CL/V)*CENT;
$TABLE double CP = CENT/V;
$CAPTURE CP',
  # iv_2cmt
  iv_2cmt = '
$PARAM CL = 5, VC = 30, Q = 10, VP = 50
$CMT CENT PER
$ODE
dxdt_CENT = -(CL/VC)*CENT - (Q/VC)*CENT + (Q/VP)*PER;
dxdt_PER  =  (Q/VC)*CENT - (Q/VP)*PER;
$TABLE double CP = CENT/VC;
$CAPTURE CP',
  # iv_3cmt
  iv_3cmt = '
$PARAM CL = 5, VC = 30, Q2 = 10, VP2 = 50, Q3 = 2, VP3 = 100
$CMT CENT P2 P3
$ODE
dxdt_CENT = -(CL/VC)*CENT - (Q2/VC)*CENT + (Q2/VP2)*P2 - (Q3/VC)*CENT + (Q3/VP3)*P3;
dxdt_P2   =  (Q2/VC)*CENT - (Q2/VP2)*P2;
dxdt_P3   =  (Q3/VC)*CENT - (Q3/VP3)*P3;
$TABLE double CP = CENT/VC;
$CAPTURE CP',
  # oral_1cmt
  oral_1cmt = '
$PARAM CL = 5, V = 50, KA = 1
$CMT DEPOT CENT
$ODE
dxdt_DEPOT = -KA*DEPOT;
dxdt_CENT  =  KA*DEPOT - (CL/V)*CENT;
$TABLE double CP = CENT/V;
$CAPTURE CP',
  # oral_1cmt_lag (lag applies to the dose into the depot)
  oral_1cmt_lag = '
$PARAM CL = 5, V = 50, KA = 1, ALAG = 0.3
$CMT DEPOT CENT
$PK ALAG_DEPOT = ALAG;
$ODE
dxdt_DEPOT = -KA*DEPOT;
dxdt_CENT  =  KA*DEPOT - (CL/V)*CENT;
$TABLE double CP = CENT/V;
$CAPTURE CP',
  # oral_2cmt
  oral_2cmt = '
$PARAM CL = 5, VC = 50, Q = 10, VP = 100, KA = 1
$CMT DEPOT CENT PER
$ODE
dxdt_DEPOT = -KA*DEPOT;
dxdt_CENT  =  KA*DEPOT - (CL/VC)*CENT - (Q/VC)*CENT + (Q/VP)*PER;
dxdt_PER   =  (Q/VC)*CENT - (Q/VP)*PER;
$TABLE double CP = CENT/VC;
$CAPTURE CP',
  # oral_1cmt_transit (three transit compartments, KTR = 4/MTT)
  oral_1cmt_transit = '
$PARAM CL = 5, V = 50, MTT = 1
$CMT TR1 TR2 TR3 CENT
$ODE
double KTR = 4.0/MTT;
dxdt_TR1  = -KTR*TR1;
dxdt_TR2  =  KTR*(TR1 - TR2);
dxdt_TR3  =  KTR*(TR2 - TR3);
dxdt_CENT =  KTR*TR3 - (CL/V)*CENT;
$TABLE double CP = CENT/V;
$CAPTURE CP',
  # iv_1cmt_mm
  iv_1cmt_mm = '
$PARAM VMAX = 100, KM = 5, V = 50
$CMT CENT
$ODE
double C = CENT/V;
dxdt_CENT = -VMAX*C/(KM + C);
$TABLE double CP = CENT/V;
$CAPTURE CP',
  # oral_1cmt_mm
  oral_1cmt_mm = '
$PARAM VMAX = 100, KM = 5, V = 50, KA = 1
$CMT DEPOT CENT
$ODE
double C = CENT/V;
dxdt_DEPOT = -KA*DEPOT;
dxdt_CENT  =  KA*DEPOT - VMAX*C/(KM + C);
$TABLE double CP = CENT/V;
$CAPTURE CP',
  # iv_1cmt_mixed (linear plus Michaelis-Menten elimination)
  iv_1cmt_mixed = '
$PARAM CL = 5, VMAX = 100, KM = 5, V = 50
$CMT CENT
$ODE
double C = CENT/V;
dxdt_CENT = -(CL/V)*CENT - VMAX*C/(KM + C);
$TABLE double CP = CENT/V;
$CAPTURE CP'
)

# allometric exponent from the PARAMETER NAME, implemented here and not read from PharmAgent
allo_exponent <- function(name) {
  if (name == "VMAX" || grepl("^(CL|Q)", name)) return(0.75)
  if (grepl("^V", name)) return(1.0)
  0
}
parse_params <- function(s) {
  kv <- strsplit(strsplit(s, ";", fixed = TRUE)[[1]], "=", fixed = TRUE)
  setNames(vapply(kv, function(x) as.numeric(x[2]), numeric(1)), vapply(kv, `[`, character(1), 1))
}

models <- list()
get_model <- function(key) {
  if (is.null(models[[key]])) {
    m <- mcode(paste0("xc_", key), code[[key]], quiet = TRUE)
    models[[key]] <<- update(m, atol = 1e-10, rtol = 1e-10, maxsteps = 100000)
  }
  models[[key]]
}

rows <- list()
for (i in seq_len(nrow(scen))) {
  sid <- scen$id[i]
  mod <- get_model(scen$model[i])
  p <- parse_params(scen$params[i])
  f <- scen$wt[i] / 70
  p <- p * vapply(names(p), function(n) f^allo_exponent(n), numeric(1))     # realised (scaled) values
  mod <- param(mod, as.list(p))

  dd <- doses[doses$id == sid, ]
  oo <- obs[obs$id == sid, ]
  dose_rows <- data.frame(ID = 1, time = dd$time, evid = 1, amt = dd$amt, cmt = 1, rate = dd$rate)
  obs_rows  <- data.frame(ID = 1, time = oo$time, evid = 0, amt = 0,      cmt = 0, rate = 0)
  d <- rbind(dose_rows, obs_rows)
  d <- d[order(d$time, -d$evid), ]            # a dose and an observation at the same time: dose first

  res <- as.data.frame(mrgsim_d(mod, d, obsonly = TRUE))
  stopifnot(nrow(res) == nrow(oo), all(abs(res$time - oo$time) < 1e-9))
  rows[[i]] <- data.frame(id = sid, time = sprintf("%.17g", res$time), cp = sprintf("%.17g", res$CP),
                          stringsAsFactors = FALSE)
  cat(sprintf("%-28s %d points\n", sid, nrow(res)))
}
write.csv(do.call(rbind, rows), file.path(outdir, "mrgsolve_sim.csv"), row.names = FALSE)
writeLines(c(paste("mrgsolve", as.character(packageVersion("mrgsolve"))), R.version.string,
             paste("run", format(Sys.time(), tz = "UTC", usetz = TRUE))),
           file.path(outdir, "mrgsolve_versions.txt"))
