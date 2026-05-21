#!/usr/bin/env Rscript

# Phase 8 optional MSGARCH robustness appendix.
#
# Scope:
# - Read CSV exported by scripts/export_msgarch_inputs.py
# - Preflight R / MSGARCH / input availability
# - AR(1)-prefilter return_for_msgarch
# - Try sGARCH/std/K=2
# - Fallback to sGARCH/norm/K=2
# - Export filtered probabilities if fit succeeds
# - Export clean skip report if anything required is unavailable
#
# Deliberately not included:
# - HAR residual inputs
# - GJR-GARCH first-pass model
# - K=3 regimes
# - strategy signals
# - backtest outputs
# - VaR/ES
# - cross-market lead-lag

options(warn = 1)

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0 || is.na(x)) y else x
}

args_full <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_full, value = TRUE)
script_path <- if (length(file_arg) > 0) sub("^--file=", "", file_arg[[1]]) else "scripts/run_msgarch.R"
PROJECT_ROOT <- normalizePath(file.path(dirname(script_path), ".."), mustWork = FALSE)

now_utc <- function() {
  format(as.POSIXct(Sys.time(), tz = "UTC"), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
}

ensure_dir <- function(path) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
}

json_escape <- function(x) {
  x <- as.character(x %||% "")
  x <- gsub("\\\\", "\\\\\\\\", x)
  x <- gsub('"', '\\"', x)
  x <- gsub("\n", "\\\\n", x)
  x <- gsub("\r", "\\\\r", x)
  x <- gsub("\t", "\\\\t", x)
  x
}

json_value <- function(x) {
  if (is.null(x) || length(x) == 0 || is.na(x)) {
    return("null")
  }
  if (is.logical(x)) {
    return(ifelse(isTRUE(x), "true", "false"))
  }
  if (is.numeric(x)) {
    if (!is.finite(x)) return("null")
    return(as.character(x))
  }
  paste0('"', json_escape(x), '"')
}

write_json_object <- function(path, values) {
  ensure_dir(dirname(path))
  keys <- names(values)
  lines <- character(length(keys) + 2)
  lines[1] <- "{"
  for (i in seq_along(keys)) {
    comma <- ifelse(i < length(keys), ",", "")
    lines[i + 1] <- paste0('  "', json_escape(keys[[i]]), '": ', json_value(values[[i]]), comma)
  }
  lines[length(lines)] <- "}"
  writeLines(lines, path, useBytes = TRUE)
}

parse_args <- function(args) {
  out <- list(
    market = NULL,
    config = "configs/model_msgarch.yaml"
  )

  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key == "--market") {
      if (i + 1 > length(args)) stop("--market requires a value")
      out$market <- toupper(args[[i + 1]])
      i <- i + 2
    } else if (key == "--config") {
      if (i + 1 > length(args)) stop("--config requires a value")
      out$config <- args[[i + 1]]
      i <- i + 2
    } else {
      stop(paste("Unknown argument:", key))
    }
  }

  if (is.null(out$market)) {
    stop("Missing required argument: --market US / INDIA / ALL")
  }
  if (!(out$market %in% c("US", "INDIA", "ALL"))) {
    stop("--market must be one of: US, INDIA, ALL")
  }

  out
}

project_path <- function(rel_path) {
  if (grepl("^([A-Za-z]:)?[/\\\\]", rel_path)) {
    normalizePath(rel_path, mustWork = FALSE)
  } else {
    normalizePath(file.path(PROJECT_ROOT, rel_path), mustWork = FALSE)
  }
}

market_slug <- function(market) {
  tolower(market)
}

default_input_csv <- function(market) {
  project_path(file.path("data", "interim", "msgarch", paste0(market_slug(market), "_msgarch_input.csv")))
}

raw_output_csv <- function(market) {
  project_path(file.path("data", "interim", "msgarch", paste0(market_slug(market), "_msgarch_raw_output.csv")))
}

model_summary_txt <- function(market) {
  project_path(file.path("data", "interim", "msgarch", paste0(market_slug(market), "_msgarch_model_summary.txt")))
}

skip_report_json <- function(market) {
  project_path(file.path("data", "interim", "msgarch", paste0(market_slug(market), "_msgarch_skip_report.json")))
}

preflight_json <- function(market) {
  project_path(file.path("data", "interim", "msgarch", paste0(market_slug(market), "_msgarch_preflight.json")))
}

read_config_market_input_csv <- function(config_path, market) {
  # Keep yaml optional. If yaml is unavailable, use the standard Phase 8 path.
  config_abs <- project_path(config_path)

  if (!file.exists(config_abs)) {
    return(default_input_csv(market))
  }

  if (!requireNamespace("yaml", quietly = TRUE)) {
    return(default_input_csv(market))
  }

  cfg <- tryCatch(
    yaml::read_yaml(config_abs),
    error = function(e) NULL
  )

  if (is.null(cfg)) {
    return(default_input_csv(market))
  }

  rel <- NULL
  try({
    rel <- cfg$markets[[market]]$output_input_csv
  }, silent = TRUE)

  if (is.null(rel) || is.na(rel) || length(rel) == 0) {
    return(default_input_csv(market))
  }

  project_path(rel)
}

write_preflight <- function(
  market,
  input_csv,
  msgarch_available,
  input_exists,
  n_observations,
  selected_spec,
  status,
  skip_reason
) {
  path <- preflight_json(market)
  write_json_object(
    path,
    list(
      market = market,
      r_available = TRUE,
      msgarch_package_available = isTRUE(msgarch_available),
      input_csv = input_csv,
      input_csv_exists = isTRUE(input_exists),
      n_observations = as.numeric(n_observations %||% 0),
      selected_spec = selected_spec %||% "",
      status = status %||% "",
      skip_reason = skip_reason %||% "",
      created_at_utc = now_utc()
    )
  )
  path
}

write_skip_report <- function(market, input_csv, skip_reason, selected_spec = "") {
  path <- skip_report_json(market)
  write_json_object(
    path,
    list(
      market = market,
      skipped = TRUE,
      skip_reason = skip_reason,
      input_csv = input_csv,
      selected_spec = selected_spec,
      fit_status = "skipped",
      created_at_utc = now_utc()
    )
  )
  path
}

read_msgarch_input <- function(input_csv) {
  df <- read.csv(input_csv, stringsAsFactors = FALSE)

  required <- c(
    "date",
    "market",
    "log_return",
    "return_for_msgarch",
    "source_return_column",
    "input_available"
  )

  missing_cols <- setdiff(required, names(df))
  if (length(missing_cols) > 0) {
    stop(paste("Input CSV missing required column(s):", paste(missing_cols, collapse = ", ")))
  }

  df$date <- as.Date(df$date)
  if (any(is.na(df$date))) {
    stop("Input CSV contains invalid date values.")
  }

  df$return_for_msgarch <- suppressWarnings(as.numeric(df$return_for_msgarch))
  bad_return <- is.na(df$return_for_msgarch) | !is.finite(df$return_for_msgarch)
  if (any(bad_return)) {
    stop(paste("Input CSV contains invalid return_for_msgarch values:", sum(bad_return)))
  }

  df <- df[order(df$date), , drop = FALSE]
  rownames(df) <- NULL
  df
}

prefilter_ar1_residuals <- function(x) {
  x <- as.numeric(x)

  if (length(x) < 10) {
    stop("Need at least 10 observations for AR(1) prefiltering.")
  }

  fit <- tryCatch(
    stats::arima(x, order = c(1, 0, 0), include.mean = TRUE, method = "ML"),
    error = function(e) NULL
  )

  if (is.null(fit)) {
    centered <- x - mean(x, na.rm = TRUE)
    return(
      list(
        residuals = centered,
        method = "demean_fallback",
        ar1_fit_status = "failed_arima_used_demean",
        ar1_phi = NA_real_,
        ar1_intercept = NA_real_
      )
    )
  }

  residuals <- as.numeric(residuals(fit))
  residuals[is.na(residuals)] <- 0.0

  coef_names <- names(fit$coef)
  phi <- if ("ar1" %in% coef_names) as.numeric(fit$coef[["ar1"]]) else NA_real_
  intercept <- if ("intercept" %in% coef_names) as.numeric(fit$coef[["intercept"]]) else NA_real_

  list(
    residuals = residuals,
    method = "AR1",
    ar1_fit_status = "ok",
    ar1_phi = phi,
    ar1_intercept = intercept
  )
}

create_spec <- function(distribution) {
  MSGARCH::CreateSpec(
    variance.spec = list(model = c("sGARCH", "sGARCH")),
    distribution.spec = list(distribution = c(distribution, distribution)),
    switch.spec = list(do.mix = FALSE)
  )
}

fit_msgarch_spec <- function(residuals, distribution) {
  spec <- create_spec(distribution)
  fit <- MSGARCH::FitML(spec = spec, data = residuals)
  list(spec = spec, fit = fit)
}

try_fit_primary_then_fallback <- function(residuals) {
  primary <- tryCatch(
    fit_msgarch_spec(residuals, "std"),
    error = function(e) e
  )

  if (!inherits(primary, "error")) {
    return(
      list(
        fit_object = primary$fit,
        spec_object = primary$spec,
        selected_spec = "sGARCH/std/K=2",
        primary_spec_attempted = "sGARCH/std/K=2",
        fallback_spec_used = "",
        fit_status = "ok"
      )
    )
  }

  fallback <- tryCatch(
    fit_msgarch_spec(residuals, "norm"),
    error = function(e) e
  )

  if (!inherits(fallback, "error")) {
    return(
      list(
        fit_object = fallback$fit,
        spec_object = fallback$spec,
        selected_spec = "sGARCH/norm/K=2",
        primary_spec_attempted = "sGARCH/std/K=2",
        fallback_spec_used = "sGARCH/norm/K=2",
        fit_status = "ok_after_fallback",
        primary_error = conditionMessage(primary)
      )
    )
  }

  stop(
    paste0(
      "Primary and fallback MSGARCH fits failed. ",
      "Primary error: ", conditionMessage(primary), " | ",
      "Fallback error: ", conditionMessage(fallback)
    )
  )
}

as_probability_matrix <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }

  if (is.data.frame(x)) {
    x <- as.matrix(x)
  }

  if (is.vector(x) && is.numeric(x)) {
    x <- matrix(x, ncol = 1)
  }

  if (!is.matrix(x)) {
    return(NULL)
  }

  mode(x) <- "numeric"

  if (ncol(x) == 2) {
    return(x)
  }

  if (nrow(x) == 2) {
    return(t(x))
  }

  NULL
}

extract_from_nested_names <- function(obj, wanted_patterns) {
  if (is.null(obj)) {
    return(NULL)
  }

  direct_names <- names(obj)
  if (!is.null(direct_names)) {
    for (nm in direct_names) {
      lower <- tolower(nm)
      for (pat in wanted_patterns) {
        if (grepl(pat, lower, fixed = TRUE)) {
          mat <- as_probability_matrix(obj[[nm]])
          if (!is.null(mat)) return(mat)
        }
      }
    }
  }

  if (is.list(obj)) {
    for (item in obj) {
      found <- extract_from_nested_names(item, wanted_patterns)
      if (!is.null(found)) return(found)
    }
  }

  NULL
}

extract_state_probabilities <- function(fit_object, residuals) {
  # MSGARCH versions differ in returned object layout.
  # Try common APIs and nested-object names, then fail cleanly.

  state_obj <- tryCatch(
    MSGARCH::State(object = fit_object, data = residuals),
    error = function(e) NULL
  )

  filtered <- extract_from_nested_names(
    state_obj,
    c("filt", "filtered", "filter", "pred")
  )

  smoothed <- extract_from_nested_names(
    state_obj,
    c("smooth", "smoothed")
  )

  if (is.null(filtered)) {
    state_obj2 <- tryCatch(
      MSGARCH::State(fit_object),
      error = function(e) NULL
    )

    filtered <- extract_from_nested_names(
      state_obj2,
      c("filt", "filtered", "filter", "pred")
    )

    smoothed <- extract_from_nested_names(
      state_obj2,
      c("smooth", "smoothed")
    )
  }

  if (is.null(filtered)) {
    # Last-resort scan of fit object.
    filtered <- extract_from_nested_names(
      fit_object,
      c("filt", "filtered", "filter", "state")
    )
  }

  if (is.null(filtered)) {
    stop("Could not extract filtered state probabilities from MSGARCH fit.")
  }

  list(
    filtered = filtered,
    smoothed = smoothed,
    extraction_method = "State_or_nested_object_scan"
  )
}

safe_volatility <- function(fit_object, residuals) {
  vol <- tryCatch(
    MSGARCH::Volatility(object = fit_object, data = residuals),
    error = function(e) NULL
  )

  if (is.null(vol)) {
    vol <- tryCatch(
      MSGARCH::Volatility(fit_object),
      error = function(e) NULL
    )
  }

  if (is.null(vol)) {
    return(rep(NA_real_, length(residuals)))
  }

  vol <- as.numeric(vol)

  if (length(vol) < length(residuals)) {
    vol <- c(rep(NA_real_, length(residuals) - length(vol)), vol)
  }

  if (length(vol) > length(residuals)) {
    vol <- tail(vol, length(residuals))
  }

  vol
}

align_probability_matrix <- function(mat, n) {
  mat <- as_probability_matrix(mat)

  if (is.null(mat)) {
    stop("Probability object cannot be converted to a 2-column matrix.")
  }

  if (nrow(mat) < n) {
    pad <- matrix(NA_real_, nrow = n - nrow(mat), ncol = 2)
    colnames(pad) <- colnames(mat)
    mat <- rbind(pad, mat)
  }

  if (nrow(mat) > n) {
    mat <- tail(mat, n)
  }

  mat
}

validate_probability_matrix <- function(mat) {
  if (is.null(mat)) {
    stop("Probability matrix is NULL.")
  }

  if (!is.matrix(mat)) {
    stop("Probability matrix is not a matrix.")
  }

  if (ncol(mat) != 2) {
    stop(paste("Expected 2 state probability columns, got", ncol(mat)))
  }

  bad <- !is.finite(mat) | mat < -1e-8 | mat > 1 + 1e-8
  if (any(bad, na.rm = TRUE)) {
    stop("Probability matrix contains values outside [0, 1].")
  }

  row_sum <- rowSums(mat)
  valid <- is.finite(row_sum)
  bad_sum <- abs(row_sum[valid] - 1.0) > 1e-4
  if (any(bad_sum)) {
    stop(paste("Probability matrix has", sum(bad_sum), "row(s) whose probabilities do not sum to 1."))
  }

  TRUE
}

build_raw_output <- function(input_df, probs, smooth_probs, cond_vol, fit_status, skip_reason) {
  n <- nrow(input_df)

  probs <- align_probability_matrix(probs, n)
  validate_probability_matrix(probs)

  if (!is.null(smooth_probs)) {
    smooth_probs <- align_probability_matrix(smooth_probs, n)
  }

  out <- data.frame(
    date = as.character(input_df$date),
    market = as.character(input_df$market),
    msgarch_raw_state_0_prob_filtered = as.numeric(probs[, 1]),
    msgarch_raw_state_1_prob_filtered = as.numeric(probs[, 2]),
    msgarch_conditional_volatility = as.numeric(cond_vol),
    msgarch_model_valid = TRUE,
    msgarch_fit_status = fit_status,
    msgarch_skip_reason = skip_reason,
    stringsAsFactors = FALSE
  )

  if (!is.null(smooth_probs)) {
    out$msgarch_raw_state_0_prob_smoothed_diagnostic <- as.numeric(smooth_probs[, 1])
    out$msgarch_raw_state_1_prob_smoothed_diagnostic <- as.numeric(smooth_probs[, 2])
  }

  out
}

write_model_summary <- function(path, values) {
  ensure_dir(dirname(path))

  lines <- c(
    "Phase 8 MSGARCH model summary",
    "================================",
    paste("created_at_utc:", now_utc()),
    paste("market:", values$market),
    paste("selected_spec:", values$selected_spec),
    paste("primary_spec_attempted:", values$primary_spec_attempted),
    paste("fallback_spec_used:", values$fallback_spec_used %||% ""),
    paste("fit_status:", values$fit_status),
    paste("convergence_status:", values$convergence_status %||% ""),
    paste("probability_extraction_method:", values$probability_extraction_method %||% ""),
    paste("prefilter_method:", values$prefilter_method %||% ""),
    paste("ar1_fit_status:", values$ar1_fit_status %||% ""),
    paste("ar1_phi:", values$ar1_phi %||% NA_real_),
    paste("ar1_intercept:", values$ar1_intercept %||% NA_real_),
    "",
    "Note:",
    "MSGARCH is an optional robustness appendix model only. It is not used for strategy construction."
  )

  writeLines(lines, path, useBytes = TRUE)
}

process_market <- function(market, config_path) {
  market <- toupper(market)
  input_csv <- read_config_market_input_csv(config_path, market)

  msgarch_available <- requireNamespace("MSGARCH", quietly = TRUE)
  input_exists <- file.exists(input_csv)

  if (!msgarch_available) {
    reason <- "R package MSGARCH is not installed."
    write_preflight(
      market = market,
      input_csv = input_csv,
      msgarch_available = FALSE,
      input_exists = input_exists,
      n_observations = 0,
      selected_spec = "",
      status = "skipped",
      skip_reason = reason
    )
    write_skip_report(market, input_csv, reason)
    message("[", market, "] skipped: ", reason)
    return(invisible(FALSE))
  }

  if (!input_exists) {
    reason <- paste("MSGARCH input CSV not found:", input_csv)
    write_preflight(
      market = market,
      input_csv = input_csv,
      msgarch_available = TRUE,
      input_exists = FALSE,
      n_observations = 0,
      selected_spec = "",
      status = "skipped",
      skip_reason = reason
    )
    write_skip_report(market, input_csv, reason)
    message("[", market, "] skipped: ", reason)
    return(invisible(FALSE))
  }

  input_df <- tryCatch(
    read_msgarch_input(input_csv),
    error = function(e) e
  )

  if (inherits(input_df, "error")) {
    reason <- paste("Failed to read MSGARCH input:", conditionMessage(input_df))
    write_preflight(
      market = market,
      input_csv = input_csv,
      msgarch_available = TRUE,
      input_exists = TRUE,
      n_observations = 0,
      selected_spec = "",
      status = "skipped",
      skip_reason = reason
    )
    write_skip_report(market, input_csv, reason)
    message("[", market, "] skipped: ", reason)
    return(invisible(FALSE))
  }

  n_obs <- nrow(input_df)

  write_preflight(
    market = market,
    input_csv = input_csv,
    msgarch_available = TRUE,
    input_exists = TRUE,
    n_observations = n_obs,
    selected_spec = "sGARCH/std/K=2 primary; sGARCH/norm/K=2 fallback",
    status = "ready",
    skip_reason = ""
  )

  fit_result <- tryCatch({
    pre <- prefilter_ar1_residuals(input_df$return_for_msgarch)
    residuals <- pre$residuals

    fit_info <- try_fit_primary_then_fallback(residuals)

    prob_info <- extract_state_probabilities(fit_info$fit_object, residuals)
    cond_vol <- safe_volatility(fit_info$fit_object, residuals)

    out <- build_raw_output(
      input_df = input_df,
      probs = prob_info$filtered,
      smooth_probs = prob_info$smoothed,
      cond_vol = cond_vol,
      fit_status = fit_info$fit_status,
      skip_reason = ""
    )

    output_path <- raw_output_csv(market)
    ensure_dir(dirname(output_path))
    write.csv(out, output_path, row.names = FALSE)

    convergence_status <- "unknown"
    try({
      if (!is.null(fit_info$fit_object$optim$message)) {
        convergence_status <- as.character(fit_info$fit_object$optim$message)
      } else if (!is.null(fit_info$fit_object$Optim$message)) {
        convergence_status <- as.character(fit_info$fit_object$Optim$message)
      }
    }, silent = TRUE)

    write_model_summary(
      model_summary_txt(market),
      list(
        market = market,
        selected_spec = fit_info$selected_spec,
        primary_spec_attempted = fit_info$primary_spec_attempted,
        fallback_spec_used = fit_info$fallback_spec_used %||% "",
        fit_status = fit_info$fit_status,
        convergence_status = convergence_status,
        probability_extraction_method = prob_info$extraction_method,
        prefilter_method = pre$method,
        ar1_fit_status = pre$ar1_fit_status,
        ar1_phi = pre$ar1_phi,
        ar1_intercept = pre$ar1_intercept
      )
    )

    TRUE
  }, error = function(e) e)

  if (inherits(fit_result, "error")) {
    reason <- paste("MSGARCH fitting or extraction failed:", conditionMessage(fit_result))
    write_preflight(
      market = market,
      input_csv = input_csv,
      msgarch_available = TRUE,
      input_exists = TRUE,
      n_observations = n_obs,
      selected_spec = "sGARCH/std/K=2 primary; sGARCH/norm/K=2 fallback",
      status = "skipped",
      skip_reason = reason
    )
    write_skip_report(
      market = market,
      input_csv = input_csv,
      skip_reason = reason,
      selected_spec = "sGARCH/std/K=2 primary; sGARCH/norm/K=2 fallback"
    )
    message("[", market, "] skipped after fit attempt: ", reason)
    return(invisible(FALSE))
  }

  # Remove old skip report after success to avoid stale downstream confusion.
  old_skip <- skip_report_json(market)
  if (file.exists(old_skip)) {
    unlink(old_skip)
  }

  message("[", market, "] MSGARCH raw output written: ", raw_output_csv(market))
  invisible(TRUE)
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))

  markets <- if (args$market == "ALL") c("US", "INDIA") else c(args$market)

  results <- lapply(markets, function(m) {
    process_market(market = m, config_path = args$config)
  })

  # Exit code remains 0 even if model skipped, because Phase 8 is optional.
  # Fatal CLI/config errors are handled before this point.
  invisible(results)
}

tryCatch(
  main(),
  error = function(e) {
    message("[MSGARCH R RUNNER ERROR] ", conditionMessage(e))
    quit(status = 1)
  }
)