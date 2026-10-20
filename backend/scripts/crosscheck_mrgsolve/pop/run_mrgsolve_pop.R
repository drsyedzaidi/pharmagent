#!/usr/bin/env Rscript
# mrgsolve population cross-check.   YOU run this (it is not run automatically):
#
#   Rscript scripts/crosscheck_mrgsolve/pop/run_mrgsolve_pop.R scripts/crosscheck_mrgsolve/pop scripts/crosscheck_mrgsolve/pop/out
#
# Reads design.csv / doses.csv / obs.csv / draws.csv (made by make_pop.py) and writes mrgsolve_pop.csv
# (id, subject, time, ipred, dv, blq), mrgsolve_eta.csv and mrgsolve_versions.txt. Installs nothing, uses no network.
#
# NO random numbers are generated here: every standard normal comes from draws.csv, the same file PharmAgent consumes.
# Everything downstream of the draw is implemented here independently, from the stated model:
#   eta_k      = sqrt(omega2_k) * z_k                       (one z per random effect, file order)
#   P_k        = theta_k * exp(eta_k)  for the IIV parameters, theta_k otherwise
#   realised   = P_k * (WT/70)^0.75 (CL, Q*, VMAX) | ^1 (V*) | ^0 (everything else)   -- scaling AFTER the eta
#   ipred      = mrgsolve solution of the structural model at the realised parameters
#   sd         = sqrt( sigma_add^2 [additive, combined]  +  (sigma_prop * ipred)^2 [proportional, combined] )
#   dv         = ipred + sd * z_eps                          (one z per observation)
#   blq        = dv < lloq   (only when an lloq is given)
# The models below are copies of the five used by run_mrgsolve.R, written from the registry equations
# (app/compute/pk_models.py); that script is validated and deliberately left untouched.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: run_mrgsolve_pop.R <pop dir> <outdir>")
indir <- args[[1]]; outdir <- args[[2]]
if (!requireNamespace("mrgsolve", quietly = TRUE)) stop("mrgsolve is not installed (this script does not install packages)")
suppressPackageStartupMessages(library(mrgsolve))
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

design <- read.csv(file.path(indir, "design.csv"), stringsAsFactors = FALSE)
doses  <- read.csv(file.path(indir, "doses.csv"),  stringsAsFactors = FALSE)
obs    <- read.csv(file.path(indir, "obs.csv"),    stringsAsFactors = FALSE)
draws  <- read.csv(file.path(indir, "draws.csv"),  stringsAsFactors = FALSE)

code <- list(
  iv_1cmt = '
$PARAM CL = 5, V = 50
$CMT CENT
$ODE dxdt_CENT = -(CL/V)*CENT;
$TABLE double CP = CENT/V;
$CAPTURE CP',
  iv_2cmt = '
$PARAM CL = 5, VC = 30, Q = 10, VP = 50
$CMT CENT PER
$ODE
dxdt_CENT = -(CL/VC)*CENT - (Q/VC)*CENT + (Q/VP)*PER;
dxdt_PER  =  (Q/VC)*CENT - (Q/VP)*PER;
$TABLE double CP = CENT/VC;
$CAPTURE CP',
  oral_1cmt = '
$PARAM CL = 5, V = 50, KA = 1
$CMT DEPOT CENT
$ODE
dxdt_DEPOT = -KA*DEPOT;
dxdt_CENT  =  KA*DEPOT - (CL/V)*CENT;
$TABLE double CP = CENT/V;
$CAPTURE CP',
  oral_1cmt_lag = '
$PARAM CL = 5, V = 50, KA = 1, ALAG = 0.3
$CMT DEPOT CENT
$PK ALAG_DEPOT = ALAG;
$ODE
dxdt_DEPOT = -KA*DEPOT;
dxdt_CENT  =  KA*DEPOT - (CL/V)*CENT;
$TABLE double CP = CENT/V;
$CAPTURE CP',
  iv_1cmt_mm = '
$PARAM VMAX = 100, KM = 5, V = 50
$CMT CENT
$ODE
double C = CENT/V;
dxdt_CENT = -VMAX*C/(KM + C);
$TABLE double CP = CENT/V;
$CAPTURE CP'
)

allo_exponent <- function(name) {
  if (name == "VMAX" || grepl("^(CL|Q)", name)) return(0.75)
  if (grepl("^V", name)) return(1.0)
  0
}
kv <- function(s) {
  if (!nzchar(s)) return(numeric(0))
  p <- strsplit(strsplit(s, ";", fixed = TRUE)[[1]], "=", fixed = TRUE)
  setNames(vapply(p, function(x) as.numeric(x[2]), numeric(1)), vapply(p, `[`, character(1), 1))
}

models <- list()
get_model <- function(key) {
  if (is.null(models[[key]])) {
    m <- mcode(paste0("xcpop_", key), code[[key]], quiet = TRUE)
    models[[key]] <<- update(m, atol = 1e-10, rtol = 1e-10, maxsteps = 100000)
  }
  models[[key]]
}

pop_rows <- list(); eta_rows <- list(); k <- 0
for (i in seq_len(nrow(design))) {
  sid <- design$id[i]
  mod <- get_model(design$model[i])
  theta <- kv(design$theta[i])
  iiv   <- strsplit(design$iiv[i], ",", fixed = TRUE)[[1]]
  om2   <- kv(design$omega2[i])
  err   <- design$error[i]
  s_prop <- if (err %in% c("proportional", "combined")) design$sigma_prop[i] else 0
  s_add  <- if (err %in% c("additive", "combined")) design$sigma_add[i] else 0
  lloq   <- design$lloq[i]
  wt     <- design$wt[i]
  dd <- doses[doses$id == sid, ]
  oo <- obs[obs$id == sid, ]
  for (s in seq_len(design$n_subjects[i])) {
    z <- draws[draws$id == sid & draws$subject == s, ]
    z_eta <- z$z[z$kind == "eta"][order(z$idx[z$kind == "eta"])]
    z_eps <- z$z[z$kind == "eps"][order(z$idx[z$kind == "eps"])]
    stopifnot(length(z_eta) == length(iiv), length(z_eps) == nrow(oo))
    eta <- setNames(sqrt(om2[iiv]) * z_eta, iiv)
    p <- theta
    p[iiv] <- p[iiv] * exp(eta)
    p <- p * vapply(names(p), function(n) (wt / 70)^allo_exponent(n), numeric(1))
    m <- param(mod, as.list(p))
    dose_rows <- data.frame(ID = 1, time = dd$time, evid = 1, amt = dd$amt, cmt = 1, rate = dd$rate)
    obs_rows  <- data.frame(ID = 1, time = oo$time, evid = 0, amt = 0,      cmt = 0, rate = 0)
    d <- rbind(dose_rows, obs_rows)
    d <- d[order(d$time, -d$evid), ]
    res <- as.data.frame(mrgsim_d(m, d, obsonly = TRUE))
    stopifnot(nrow(res) == nrow(oo), all(abs(res$time - oo$time) < 1e-9))
    ipred <- res$CP
    sd <- sqrt(s_add^2 + (s_prop * ipred)^2)
    dv <- ipred + sd * z_eps
    blq <- if (is.na(lloq)) rep(0L, length(dv)) else as.integer(dv < lloq)
    k <- k + 1
    pop_rows[[k]] <- data.frame(id = sid, subject = s, time = sprintf("%.17g", res$time),
                                ipred = sprintf("%.17g", ipred), dv = sprintf("%.17g", dv), blq = blq,
                                stringsAsFactors = FALSE)
    eta_rows[[k]] <- data.frame(id = sid, subject = s, param = iiv, eta = sprintf("%.17g", eta),
                                stringsAsFactors = FALSE)
  }
  cat(sprintf("%-28s %d subjects\n", sid, design$n_subjects[i]))
}
write.csv(do.call(rbind, pop_rows), file.path(outdir, "mrgsolve_pop.csv"), row.names = FALSE)
write.csv(do.call(rbind, eta_rows), file.path(outdir, "mrgsolve_eta.csv"), row.names = FALSE)
writeLines(c(paste("mrgsolve", as.character(packageVersion("mrgsolve"))), R.version.string,
             paste("run", format(Sys.time(), tz = "UTC", usetz = TRUE))),
           file.path(outdir, "mrgsolve_versions.txt"))
