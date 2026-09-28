# Parametrized Monolix fit (lixoftConnectors), run inside the isolated x86_64 R:
#   Rscript monolix_fit.R <config.json>
# Ported from poppk-shiny/monolix/monolix_fit.R.
#
# config.json keys:
#   suite        path to monolixSuite
#   dataFile     NONMEM-style CSV (ID, TIME, DV, AMT, EVID, CMT, WT)
#   headerTypes  Monolix header types aligned to the CSV columns
#   modelFile    mlxtran structural model (pkmodel macro)
#   errorModel   constant | proportional | combined1 | combined2
#   distribution normal | lognormal | logitnormal
#   outDir       where results.csv, loglik.json and status.json are written
#
# Writes status.json {ok, message} always, results.csv (parameter, estimate,
# RSE_pct) and loglik.json on success. Any engine error lands in status.json.

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) >= 1)
cfg <- RJSONIO::fromJSON(args[1])
dir.create(cfg$outDir, showWarnings = FALSE, recursive = TRUE)

status_path <- file.path(cfg$outDir, "status.json")
write_status <- function(ok, message) {
  writeLines(RJSONIO::toJSON(list(ok = ok, message = message)), status_path)
}

result <- tryCatch({
  suppressMessages(library(lixoftConnectors))
  ok <- initializeLixoftConnectors(software = "monolix", path = cfg$suite,
                                   force = TRUE)
  if (!isTRUE(ok)) stop("initializeLixoftConnectors returned FALSE")

  newProject(
    modelFile = cfg$modelFile,
    data = list(dataFile = cfg$dataFile, headerTypes = cfg$headerTypes))

  oi <- tryCatch(getObservationInformation(), error = function(e) NULL)
  obsName <- if (is.list(oi) && !is.null(oi$name)) oi$name[1]
             else if (is.character(oi)) oi[1]
             else "Cc"

  em <- stats::setNames(list(cfg$errorModel), obsName)
  do.call(setErrorModel, em)
  di <- stats::setNames(list(cfg$distribution), obsName)
  do.call(setObservationDistribution, di)

  sc <- getScenario()
  sc$tasks <- c(populationParameterEstimation = TRUE,
                conditionalModeEstimation     = TRUE,
                conditionalDistributionSampling = TRUE,
                standardErrorEstimation       = TRUE,
                logLikelihoodEstimation       = TRUE)
  setScenario(sc)
  runScenario()

  pop <- getEstimatedPopulationParameters()
  se  <- tryCatch(getEstimatedStandardErrors(), error = function(e) NULL)
  ll  <- tryCatch(getEstimatedLogLikelihood(), error = function(e) NULL)

  rse <- rep(NA_real_, length(pop))
  if (is.list(se) && !is.null(se$stochasticApproximation)) {
    sa <- se$stochasticApproximation
    m <- match(names(pop), sa$parameter)
    rse <- 100 * sa$se[m] / abs(pop)
  }
  out <- data.frame(parameter = names(pop), estimate = signif(pop, 6),
                    RSE_pct = signif(rse, 4))
  utils::write.csv(out, file.path(cfg$outDir, "results.csv"), row.names = FALSE)

  if (!is.null(ll)) {
    lst <- if (is.list(ll)) ll else as.list(ll)
    # flatten one level (importanceSampling / linearization sub-lists)
    flat <- list()
    for (nm in names(lst)) {
      v <- lst[[nm]]
      if (is.list(v) || (is.numeric(v) && !is.null(names(v)))) {
        for (k in names(v)) flat[[k]] <- unname(v[[k]])
      } else flat[[nm]] <- unname(v)
    }
    writeLines(RJSONIO::toJSON(flat), file.path(cfg$outDir, "loglik.json"))
  }
  "ok"
}, error = function(e) paste("ERROR:", conditionMessage(e)))

write_status(identical(result, "ok"), result)
cat(result, "\n")
