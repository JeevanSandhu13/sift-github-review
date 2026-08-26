# Sift runtime library for R.
#
# Sourced at the top of every R script Sift runs. Provides the single
# sanctioned I/O surface for emitting structured results:
#   sift$result(type = "linear_regression", ...)
#   sift$from_lm(model, ...)
#   sift$from_t_test(res, ...)
#   sift$from_summarize(var_name, n, mean, sd, missing_count, distinct_count)
#   sift$from_table(var_name, counts, n, missing_count)
#
# The script writes structured payloads to the path in $SIFT_RESULT_PATH.
# Raw stdout / stderr are captured by the executor as the "raw log" the
# researcher sees in the TUI; only the structured JSON reaches the sanitizer
# (and from there, Claude).
#
# Ships a pure-R JSON serializer so researchers don't need to install
# jsonlite. Handles the subset of types Sift needs: null, TRUE/FALSE,
# numbers, strings, unnamed lists/vectors (→ arrays), named lists (→
# objects). That's enough for every v0 analysis payload.
#
# NOTE: floats are emitted at full precision. The Python sanitizer clamps
# precision per-type using sigfigs_for_n; the runtime library does NOT
# pre-round, so the clamp is applied consistently regardless of which
# language produced the result.

sift <- new.env(parent = emptyenv())


# ---------------------------------------------------------------------------
# Per-run authenticity token
# ---------------------------------------------------------------------------
# Read the token once at source-time, stash it in the library's private
# env, and clear the env var so user code loaded after this file can't
# read it via Sys.getenv. Claude's script still CAN find it via
# environment introspection (`ls(sift)`, `get("token", ..., envir =
# sift)`) — R closures are open — but doing so requires code that
# clearly shows up in the executed script the researcher reviews. That
# raises attacker cost without pretending to be a structural
# guarantee. See `docs/architecture.md` "runtime-library contract" for
# the deliberate limits of this measure.

sift$.run_token <- Sys.getenv("SIFT_RUN_TOKEN")
if (!nzchar(sift$.run_token)) {
  stop(
    "SIFT_RUN_TOKEN not set. This script must be run through the ",
    "Sift executor; direct `Rscript` invocation of user code that ",
    "emits result payloads isn't supported."
  )
}
Sys.unsetenv("SIFT_RUN_TOKEN")


# ---------------------------------------------------------------------------
# Pure-R JSON serializer
# ---------------------------------------------------------------------------

sift$.json_escape_str <- function(s) {
  s <- as.character(s)
  # Order matters: backslash first so we don't double-escape ours.
  s <- gsub("\\", "\\\\", s, fixed = TRUE)
  s <- gsub('"', '\\"', s, fixed = TRUE)
  s <- gsub("\n", "\\n", s, fixed = TRUE)
  s <- gsub("\r", "\\r", s, fixed = TRUE)
  s <- gsub("\t", "\\t", s, fixed = TRUE)
  # RFC 8259 §7 requires that ALL U+0000..U+001F appear as escape
  # sequences inside JSON strings. Without this pass, a value-label or
  # variable-label byte like \x01 (real automated-export datasets do
  # ship these) reaches the wire as a raw control character — which
  # Python's `json.loads` rejects with "Invalid control character",
  # and the executor's JSONL parser drops every line of the payload
  # silently. Stata's helpers have the same gap; matching changes
  # land in the .ado files.
  # Codepoints 9 (\t), 10 (\n), 13 (\r) are already escaped via the
  # named-escape gsubs above. NUL (codepoint 0) cannot occur in an R
  # character vector at all -- R rejects strings with embedded nulls
  # at every ingestion boundary -- so 1..31 minus the three handled
  # is the full range we need to walk here.
  for (cp in setdiff(1:31, c(9L, 10L, 13L))) {
    s <- gsub(intToUtf8(cp), sprintf("\\u%04x", cp), s, fixed = TRUE)
  }
  paste0('"', s, '"')
}

sift$.to_json <- function(x) {
  if (is.null(x)) return("null")

  # Coerce factors to their character labels BEFORE any of the scalar
  # / vector branches. A factor is `is.atomic` and `length(x) == 1`
  # for length-1 cases, but is neither `is.numeric` nor
  # `is.character`, so it falls through every scalar guard into the
  # atomic-vector branch. That branch recurses on `x[[1]]`, and for a
  # factor `x[[1]]` is *another factor* equal to `x` itself —
  # infinite recursion → node stack overflow. Concretely: any
  # `factor(...)` value (very common with `read.csv` defaults pre-
  # R 4.0, and with `stringsAsFactors = TRUE`) crashes the script
  # before any payload reaches disk. Convert to character early so
  # the rest of the serializer treats labels like ordinary strings.
  if (is.factor(x)) x <- as.character(x)

  # NA-of-any-flavor is JSON null. Catch this BEFORE the atomic-vector
  # branch below: a length-1 NA passes `is.atomic`, and `x[[1]]` for an
  # NA is identical to NA itself, so without this guard the recursion
  # never terminates and the script crashes with a node stack overflow.
  # Concretely: any helper field that ends up NA upstream (e.g.
  # `attr(x, "label")` returning NA, an upstream `mean(x)` over an
  # all-NA vector) would crash the whole script before any payload
  # reached disk.
  if (length(x) == 1 && is.atomic(x) && is.na(x)) return("null")

  # Scalars first — auto-unbox for length-1 atomics.
  if (is.logical(x) && length(x) == 1 && !is.na(x)) {
    return(if (x) "true" else "false")
  }
  if (is.numeric(x) && length(x) == 1) {
    if (is.na(x) || !is.finite(x)) return("null")
    # `digits = 17` preserves full IEEE-754 precision — the Python
    # sanitizer is responsible for N-appropriate clamping, so the
    # runtime emits as-is. Allow scientific notation (the default) so
    # extremely small / large numbers get compact valid-JSON literals
    # like `1.7858e-41` instead of ugly long decimals.
    # `decimal.mark = "."` is required: `format()` honors the
    # locale-dependent OutDec option, so a researcher script that ran
    # `options(OutDec = ",")` (German/French/Spanish locales) would
    # otherwise emit `3,14...` and break JSON parsing for every line
    # that follows. Stata's helper uses `strofreal(..., "%21.17e")`
    # which is locale-independent — match that guarantee here.
    return(format(x, digits = 17, trim = TRUE, decimal.mark = "."))
  }
  if (is.character(x) && length(x) == 1 && !is.na(x)) {
    return(sift$.json_escape_str(x))
  }

  # Vectors → JSON array, element-wise.
  if (is.atomic(x)) {
    parts <- vapply(seq_along(x), function(i) sift$.to_json(x[[i]]),
                    character(1))
    return(paste0("[", paste(parts, collapse = ","), "]"))
  }

  # Lists.
  if (is.list(x)) {
    nms <- names(x)
    # Mixed naming (some named, some positional) — `list(a = 1, 2)` —
    # used to silently fall back to array-mode and DROP the named
    # entries. That's a quiet data loss: a researcher passing
    # mixed args (e.g. via `do.call`) gets a payload whose shape no
    # longer matches schema validation, but no error is raised.
    # Keep array-mode only for the *fully-positional* case (no names
    # at all). Any partial naming becomes object-mode, with empty
    # names auto-numbered so the JSON is still well-formed.
    if (is.null(nms)) {
      parts <- vapply(x, sift$.to_json, character(1))
      return(paste0("[", paste(parts, collapse = ","), "]"))
    }
    if (any(!nzchar(nms))) {
      idx <- which(!nzchar(nms))
      nms[idx] <- paste0("_", idx)
    }
    parts <- vapply(seq_along(x), function(i) {
      paste0(sift$.json_escape_str(nms[i]), ":",
             sift$.to_json(x[[i]]))
    }, character(1))
    return(paste0("{", paste(parts, collapse = ","), "}"))
  }

  stop("sift.R: unsupported type for JSON: ", class(x)[1])
}


# ---------------------------------------------------------------------------
# Core emit
# ---------------------------------------------------------------------------

sift$.write_result <- function(payload) {
  result_path <- Sys.getenv("SIFT_RESULT_PATH")
  if (!nzchar(result_path)) {
    stop(
      "SIFT_RESULT_PATH not set. This script must be run through ",
      "Sift — direct `Rscript` invocation isn't supported."
    )
  }
  # Embed the per-run authenticity token, then APPEND a single JSONL
  # line so multiple emit calls in one script all reach the executor.
  # Single-helper scripts produce one line; multi-helper scripts
  # produce N lines in emission order.
  payload[["_token"]] <- sift$.run_token
  con <- file(result_path, open = "a", encoding = "UTF-8")
  on.exit(close(con), add = TRUE)
  writeLines(sift$.to_json(payload), con)
  invisible(NULL)
}

sift$result <- function(type, ...) {
  # Strip sanitizer-side markers. Typed helpers write directly via
  # .write_result only after computing disclosure metrics or deriving
  # an exact registry method from a fitted object. Allowing either
  # marker through here would let a generic payload forge that proof.
  payload <- c(list(type = type), list(...))
  payload[["_via_helper"]] <- NULL
  payload[["_registry_method_id"]] <- NULL
  sift$.write_result(payload)
}

# Registry-backed aggregate method result. The central sanitizer supplies the
# method's claim rule and rejects missing mandatory diagnostics; this helper
# deliberately accepts no observation-level vectors.
sift$from_method <- function(method_id, n, diagnostics,
                             estimates = NULL, standard_errors = NULL,
                             p_values = NULL, ci_lower = NULL,
                             ci_upper = NULL, metrics = NULL, ...) {
  payload <- list(
    method_id = method_id,
    n = as.integer(n),
    diagnostics = diagnostics
  )
  optional <- list(
    estimates = estimates,
    standard_errors = standard_errors,
    p_values = p_values,
    ci_lower = ci_lower,
    ci_upper = ci_upper,
    metrics = metrics
  )
  for (nm in names(optional)) {
    if (!is.null(optional[[nm]])) payload[[nm]] <- optional[[nm]]
  }
  payload <- c(payload, list(...))
  do.call(sift$result, c(list(type = "method_result"), payload))
}

sift$.merge_method_diagnostics <- function(defaults, diagnostics = NULL) {
  if (is.null(diagnostics)) return(defaults)
  if (!is.list(diagnostics) || is.null(names(diagnostics))) {
    stop("diagnostics must be a named list")
  }
  for (nm in names(diagnostics)) defaults[[nm]] <- diagnostics[[nm]]
  defaults
}

sift$.method_quantity_name <- function(value, field = "name") {
  name <- as.character(value)[1]
  if (is.na(name) || !nzchar(name) || nchar(name, type = "chars") > 40L ||
      !grepl("^[A-Za-z0-9_.(][A-Za-z0-9_.():^#]*$", name)) {
    stop(field, " must be an identifier/formula-shaped name of at most 40 characters")
  }
  name
}

sift$.positive_count <- function(value, field) {
  numeric <- suppressWarnings(as.numeric(value))
  integer <- suppressWarnings(as.integer(value))
  if (length(numeric) != 1L || is.na(numeric) || !is.finite(numeric) ||
      numeric <= 0 || numeric != integer) {
    stop(field, " must be a positive integer")
  }
  integer
}

sift$.special_regression_diagnostics <- function(
    convergence, specific = list(), diagnostics = NULL) {
  defaults <- c(list(
    convergence = convergence,
    specification = "warn",
    influence = "warn",
    multicollinearity = "warn",
    heteroskedasticity = "not_applicable",
    residual_distribution = "not_applicable"
  ), specific)
  sift$.merge_method_diagnostics(defaults, diagnostics)
}

sift$.normal_intervals <- function(estimates, standard_errors) {
  common <- intersect(names(estimates), names(standard_errors))
  list(
    lower = estimates[common] - stats::qnorm(0.975) * standard_errors[common],
    upper = estimates[common] + stats::qnorm(0.975) * standard_errors[common]
  )
}

#' Emit an aggregate MASS::polr ordinal regression fit.
#'
#' Outcome labels are intentionally replaced by threshold_1, threshold_2, ...
#' so category values never cross the disclosure boundary.
sift$from_ordinal_model <- function(
    fit, diagnostics = NULL, proportional_odds = "warn", ...) {
  if (!inherits(fit, "polr")) {
    stop("fit must inherit from MASS::polr")
  }
  link <- switch(as.character(fit$method), logistic = "logit",
                 probit = "probit", NULL)
  if (is.null(link)) stop("only logistic and probit ordinal links are supported")
  beta <- stats::coef(fit)
  thresholds <- fit$zeta
  if (length(thresholds) < 2L) stop("ordinal regression requires at least three categories")
  threshold_names <- paste0("threshold_", seq_along(thresholds))
  if (any(names(beta) %in% threshold_names)) {
    stop("predictor names collide with ordinal threshold identifiers")
  }
  estimates <- c(beta, stats::setNames(as.numeric(thresholds), threshold_names))
  vc <- stats::vcov(fit)
  se_all <- sqrt(diag(vc))
  se <- se_all[names(beta)]
  p <- 2 * stats::pnorm(abs(beta / se), lower.tail = FALSE)
  intervals <- sift$.normal_intervals(estimates, se)
  convergence <- is.null(fit$convergence) || identical(as.integer(fit$convergence), 0L)
  n <- as.integer(stats::nobs(fit))
  if (!identical(proportional_odds, "warn")) {
    stop("proportional_odds cannot be promoted without an executable assumption test")
  }
  ordinal_diagnostics <- sift$.special_regression_diagnostics(
    convergence, list(proportional_odds = "warn"), diagnostics)
  ordinal_diagnostics$proportional_odds <- "warn"
  sift$from_method(
    "ordinal_regression", n = n,
    diagnostics = ordinal_diagnostics,
    estimates = as.list(estimates), standard_errors = as.list(se),
    p_values = as.list(p), ci_lower = as.list(intervals$lower),
    ci_upper = as.list(intervals$upper),
    metrics = as.list(c(category_count = length(thresholds) + 1L,
                threshold_count = length(thresholds),
                log_likelihood = as.numeric(stats::logLik(fit)),
                aic = stats::AIC(fit))),
    model_form = paste0("ordered_", link), link = link,
    uncertainty_type = "classical", ...
  )
}

#' Emit an aggregate nnet::multinom fit with synthetic class identifiers.
sift$from_multinomial_model <- function(fit, diagnostics = NULL, ...) {
  if (!inherits(fit, "multinom")) stop("fit must inherit from nnet::multinom")
  sm <- summary(fit)
  beta <- sm$coefficients
  se <- sm$standard.errors
  if (is.null(dim(beta))) {
    stop("multinomial regression requires at least three outcome categories")
  }
  terms <- colnames(beta)
  estimates <- standard_errors <- p_values <- numeric()
  for (equation in seq_len(nrow(beta))) {
    for (column in seq_len(ncol(beta))) {
      key <- paste0("class_", equation, "#", terms[column])
      key <- sift$.method_quantity_name(key)
      estimates[key] <- beta[equation, column]
      standard_errors[key] <- se[equation, column]
      p_values[key] <- 2 * stats::pnorm(
        abs(beta[equation, column] / se[equation, column]), lower.tail = FALSE)
    }
  }
  intervals <- sift$.normal_intervals(estimates, standard_errors)
  response <- stats::model.response(stats::model.frame(fit))
  counts <- table(response)
  categories <- length(counts)
  min_category_n <- min(as.integer(counts))
  sift$from_method(
    "multinomial_regression", n = length(response),
    diagnostics = sift$.special_regression_diagnostics(
      identical(as.integer(fit$convergence), 0L),
      list(class_support = min_category_n >= 10L), diagnostics),
    estimates = as.list(estimates), standard_errors = as.list(standard_errors),
    p_values = as.list(p_values), ci_lower = as.list(intervals$lower),
    ci_upper = as.list(intervals$upper),
    metrics = as.list(c(category_count = categories,
                equation_count = nrow(beta),
                min_category_n = min_category_n,
                log_likelihood = as.numeric(stats::logLik(fit)),
                aic = stats::AIC(fit))),
    model_form = "multinomial_logit", link = "logit",
    uncertainty_type = "classical", ...
  )
}

#' Emit an aggregate pscl::zeroinfl Poisson or negative-binomial fit.
sift$from_zero_inflated_model <- function(fit, diagnostics = NULL, ...) {
  if (!inherits(fit, "zeroinfl")) stop("fit must inherit from pscl::zeroinfl")
  beta <- stats::coef(fit)
  names(beta) <- sub("^zero_", "inflate_", names(beta))
  vc <- stats::vcov(fit)
  se <- sqrt(diag(vc))
  names(se) <- names(beta)
  p <- 2 * stats::pnorm(abs(beta / se), lower.tail = FALSE)
  intervals <- sift$.normal_intervals(beta, se)
  response <- stats::model.response(stats::model.frame(fit))
  count_mean <- mean(response)
  ratio <- if (count_mean > 0) stats::var(response) / count_mean else 0
  distribution <- if (identical(fit$dist, "negbin")) {
    "zero_inflated_negative_binomial"
  } else "zero_inflated_poisson"
  sift$from_method(
    "zero_inflated_model", n = length(response),
    diagnostics = sift$.special_regression_diagnostics(
      isTRUE(fit$converged),
      list(zero_process_specification = "warn", overdispersion = ratio),
      diagnostics),
    estimates = as.list(beta), standard_errors = as.list(se),
    p_values = as.list(p), ci_lower = as.list(intervals$lower),
    ci_upper = as.list(intervals$upper),
    metrics = as.list(c(zero_fraction = mean(response == 0), count_mean = count_mean,
                variance_mean_ratio = ratio, parameter_count = length(beta),
                log_likelihood = as.numeric(stats::logLik(fit)),
                aic = stats::AIC(fit))),
    model_form = distribution, link = "log", uncertainty_type = "classical", ...
  )
}

#' Emit an aggregate lm fit containing a declared spline/non-linear basis.
sift$from_spline_model <- function(
    fit, basis_df, basis = "bspline", diagnostics = NULL, ...) {
  allowed <- c("bspline", "natural_spline", "restricted_cubic_spline", "polynomial")
  if (!inherits(fit, "lm")) stop("fit must inherit from lm")
  if (!basis %in% allowed) stop("basis is not supported")
  basis_df <- sift$.positive_count(basis_df, "basis_df")
  n <- as.integer(stats::nobs(fit))
  if (basis_df < 2L || basis_df >= n) stop("basis_df must be at least two and below n")
  sm <- summary(fit)
  table <- sm$coefficients
  columns <- colnames(stats::model.matrix(fit))
  if (nrow(table) != length(columns)) {
    stop("aliased or incomplete spline design cannot be emitted safely")
  }
  pattern <- switch(
    basis,
    bspline = "(^|:)bs\\(",
    natural_spline = "(^|:)ns\\(",
    restricted_cubic_spline = "(^|:)ns\\(",
    polynomial = "(^|:)(poly|I)\\("
  )
  basis_indices <- which(grepl(pattern, columns))
  if (length(basis_indices) != basis_df) {
    stop("declared basis_df does not match the fitted nonlinear design columns")
  }
  names_out <- character(length(columns))
  basis_counter <- 0L
  for (index in seq_along(columns)) {
    if (index %in% basis_indices) {
      basis_counter <- basis_counter + 1L
      names_out[index] <- paste0("basis_", basis_counter)
    } else if (columns[index] == "(Intercept)") {
      names_out[index] <- "intercept"
    } else {
      names_out[index] <- sift$.method_quantity_name(columns[index], "covariate name")
    }
  }
  if (anyDuplicated(names_out)) {
    stop("covariate names collide with synthetic spline-basis identifiers")
  }
  estimates <- stats::setNames(table[, 1], names_out)
  se <- stats::setNames(table[, 2], names_out)
  p <- stats::setNames(table[, 4], names_out)
  intervals <- sift$.normal_intervals(estimates, se)
  sift$from_method(
    "spline_regression", n = n,
    diagnostics = sift$.special_regression_diagnostics(
      TRUE, list(degrees_of_freedom_sensitivity = "warn"), diagnostics),
    estimates = as.list(estimates), standard_errors = as.list(se),
    p_values = as.list(p), ci_lower = as.list(intervals$lower),
    ci_upper = as.list(intervals$upper),
    metrics = as.list(c(basis_df = basis_df,
                basis_parameter_count = length(basis_indices),
                parameter_count = length(estimates),
                r_squared = sm$r.squared, aic = stats::AIC(fit),
                bic = stats::BIC(fit))),
    model_form = if (basis == "polynomial") "polynomial_regression" else "regression_spline",
    basis = basis, uncertainty_type = "classical", ...
  )
}

sift$.survey_design_metadata <- function(design) {
  if (!requireNamespace("survey", quietly = TRUE)) {
    stop("the R survey package is required")
  }
  weights <- as.numeric(stats::weights(design, type = "sampling"))
  if (!length(weights) || any(!is.finite(weights)) || any(weights <= 0)) {
    stop("survey probability weights must be finite and strictly positive")
  }
  effective_n <- sum(weights)^2 / sum(weights^2)
  weight_cv <- stats::sd(weights) / mean(weights)
  design_df <- as.numeric(survey::degf(design))
  if (!is.finite(design_df) || design_df < 1) stop("survey design has no residual degrees of freedom")
  if (inherits(design, "svyrep.design")) {
    raw_type <- tolower(as.character(design$type)[1])
    method <- if (grepl("fay", raw_type)) "fay" else if (grepl("brr", raw_type)) {
      "brr"
    } else if (grepl("jk|jack", raw_type)) "jackknife" else "bootstrap"
    replicates <- ncol(as.matrix(design$repweights))
    if (!is.finite(replicates) || replicates < 2L) stop("replicate design has fewer than two replicates")
    scale <- as.numeric(design$scale)[1]
    rscales <- as.numeric(design$rscales)
    mse <- isTRUE(design$mse)
    if (!is.finite(scale) || scale <= 0 || length(rscales) != replicates ||
        any(!is.finite(rscales)) || any(rscales <= 0)) {
      stop("replicate design scale/rscales metadata is invalid")
    }
    return(list(
      weights = weights, effective_n = effective_n, weight_cv = weight_cv,
      variance_method = method,
      metrics = list(strata_count = 0, psu_count = 0,
                     lonely_strata_count = 0, design_df = design_df,
                     lonely_certainty_count = 0, lonely_adjusted_count = 0,
                     fpc_fraction_min = 0, fpc_fraction_max = 0,
                     replicate_count = replicates, stage_count = 0,
                     secondary_psu_count = 0, replicate_mse = as.numeric(mse),
                     replicate_scale = scale,
                     replicate_rscale_min = min(rscales),
                     replicate_rscale_max = max(rscales))
    ))
  }
  cluster <- as.data.frame(design$cluster)
  strata <- as.data.frame(design$strata)
  if (!nrow(cluster) || !nrow(strata) || nrow(cluster) != length(weights)) {
    stop("survey design does not retain aligned strata and cluster metadata")
  }
  first_stratum <- interaction(strata[[1]], drop = TRUE)
  first_psu <- interaction(first_stratum, cluster[[1]], drop = TRUE)
  psu_per_stratum <- tapply(first_psu, first_stratum, function(value) length(unique(value)))
  lonely <- sum(psu_per_stratum == 1L)
  stage_count <- ncol(cluster)
  if (!stage_count %in% c(1L, 2L)) stop("only one- or two-stage survey designs are supported")
  secondary <- if (stage_count == 2L) {
    length(unique(interaction(first_psu, cluster[[2]], drop = TRUE)))
  } else 0L
  fractions <- 0
  row_first_fraction <- rep(0, length(weights))
  if (!is.null(design$fpc) && !is.null(design$fpc$sampsize) &&
      !is.null(design$fpc$popsize)) {
    sample_size <- as.matrix(design$fpc$sampsize)
    population_size <- as.matrix(design$fpc$popsize)
    valid <- is.finite(sample_size) & is.finite(population_size) & population_size > 0
    if (any(valid)) fractions <- sample_size[valid] / population_size[valid]
    valid_first <- valid[, 1]
    row_first_fraction[valid_first] <-
      sample_size[valid_first, 1] / population_size[valid_first, 1]
  }
  if (any(!is.finite(fractions)) || any(fractions < 0 | fractions > 1)) {
    stop("survey FPC metadata is invalid")
  }
  certainty <- 0L
  if (lonely > 0L) {
    fraction_by_stratum <- tapply(row_first_fraction, first_stratum, function(value) {
      if (max(value) - min(value) > 1e-10) stop("FPC must be constant within stratum")
      value[1]
    })
    certainty <- sum(psu_per_stratum == 1L & fraction_by_stratum == 1)
  }
  list(
    weights = weights, effective_n = effective_n, weight_cv = weight_cv,
    variance_method = "taylor_linearization",
    metrics = list(
      strata_count = length(unique(first_stratum)),
      psu_count = length(unique(first_psu)), lonely_strata_count = lonely,
      lonely_certainty_count = certainty, lonely_adjusted_count = 0,
      design_df = design_df, fpc_fraction_min = min(fractions),
      fpc_fraction_max = max(fractions), replicate_count = 0,
      stage_count = stage_count, secondary_psu_count = secondary
    )
  )
}

#' Compute and emit an R survey::svymean estimate (or binary proportion).
sift$from_survey_mean <- function(
    design, formula, proportion = FALSE, lonely_psu = "fail",
    diagnostics = NULL, ...) {
  if (!requireNamespace("survey", quietly = TRUE)) stop("the R survey package is required")
  if (!lonely_psu %in% c("fail", "adjust", "certainty")) stop("unsupported lonely PSU policy")
  metadata <- sift$.survey_design_metadata(design)
  if (lonely_psu == "certainty" &&
      metadata$metrics$lonely_certainty_count != metadata$metrics$lonely_strata_count) {
    stop("lonely_psu='certainty' requires FPC metadata proving a sampling fraction of one")
  }
  if (lonely_psu == "adjust") {
    metadata$metrics$lonely_adjusted_count <-
      metadata$metrics$lonely_strata_count - metadata$metrics$lonely_certainty_count
  }
  old <- getOption("survey.lonely.psu")
  on.exit(options(survey.lonely.psu = old), add = TRUE)
  options(survey.lonely.psu = lonely_psu)
  statistic <- survey::svymean(formula, design, na.rm = FALSE, deff = "replace")
  estimate_values <- stats::coef(statistic)
  if (length(estimate_values) != 1L) {
    stop("survey mean/proportion helper requires exactly one numeric estimand")
  }
  observed <- stats::model.frame(formula, design$variables)[[1]]
  if (proportion && (!is.numeric(observed) || any(!observed %in% c(0, 1)))) {
    stop("survey proportions require a binary numeric 0/1 outcome")
  }
  estimate <- as.numeric(estimate_values[1])
  variance <- as.numeric(stats::vcov(statistic)[1, 1])
  se <- sqrt(variance)
  deff <- as.numeric(survey::deff(statistic))[1]
  if (!is.finite(deff) || deff < 0) stop("survey design effect is unavailable")
  reference_variance <- if (deff > 0) variance / deff else variance
  df <- as.integer(metadata$metrics$design_df)
  critical <- stats::qt(0.975, df = df)
  p <- 2 * stats::pt(abs(estimate / se), df = df, lower.tail = FALSE)
  lower <- estimate - critical * se
  upper <- estimate + critical * se
  name <- if (proportion) "proportion" else "mean"
  if (proportion) { lower <- max(0, lower); upper <- min(1, upper) }
  metrics <- c(metadata$metrics, list(
    variance = variance, reference_variance = reference_variance,
    design_effect = deff, effective_sample_size = metadata$effective_n,
    weight_cv = metadata$weight_cv
  ))
  diag <- sift$.merge_method_diagnostics(list(
    weight_distribution = metadata$weight_cv, design_effect = deff,
    effective_sample_size = metadata$effective_n,
    strata_psu_support = metadata$metrics$design_df >= 1,
    variance_estimator = "pass",
    lonely_psu = if (metadata$metrics$lonely_strata_count == 0) "pass" else "warn"
  ), diagnostics)
  sift$from_method(
    if (proportion) "survey_proportion" else "survey_mean",
    n = length(metadata$weights), diagnostics = diag,
    estimates = stats::setNames(list(estimate), name),
    standard_errors = stats::setNames(list(se), name),
    p_values = stats::setNames(list(p), name),
    ci_lower = stats::setNames(list(lower), name),
    ci_upper = stats::setNames(list(upper), name), metrics = metrics,
    weight_type = "probability", variance_method = metadata$variance_method,
    lonely_psu_handling = lonely_psu, uncertainty_type = "design_based", ...
  )
}

#' Emit an already-fitted survey::svyglm with design-based covariance.
sift$from_survey_regression <- function(
    fit, lonely_psu = "fail", diagnostics = NULL, ...) {
  if (!inherits(fit, "svyglm")) stop("fit must inherit from survey::svyglm")
  design <- fit$survey.design
  metadata <- sift$.survey_design_metadata(design)
  if (lonely_psu == "certainty" &&
      metadata$metrics$lonely_certainty_count != metadata$metrics$lonely_strata_count) {
    stop("lonely_psu='certainty' requires FPC metadata proving a sampling fraction of one")
  }
  if (lonely_psu == "adjust") {
    metadata$metrics$lonely_adjusted_count <-
      metadata$metrics$lonely_strata_count - metadata$metrics$lonely_certainty_count
  }
  table <- summary(fit)$coefficients
  estimates <- table[, 1]; standard_errors <- table[, 2]
  p_values <- table[, ncol(table)]
  names_out <- vapply(names(estimates), sift$.method_quantity_name, character(1))
  names(estimates) <- names(standard_errors) <- names(p_values) <- names_out
  critical <- stats::qt(0.975, df = as.integer(metadata$metrics$design_df))
  lower <- estimates - critical * standard_errors
  upper <- estimates + critical * standard_errors
  design_cov <- stats::vcov(fit)
  naive_cov <- fit$naive.cov
  if (is.null(naive_cov) || any(diag(naive_cov) <= 0)) {
    stop("svyglm fit does not retain a valid with-replacement reference covariance")
  }
  design_effects <- diag(design_cov) / diag(naive_cov)
  metrics <- c(metadata$metrics, list(
    effective_sample_size = metadata$effective_n, weight_cv = metadata$weight_cv
  ))
  for (index in seq_along(names_out)) {
    metrics[[paste0("variance#", names_out[index])]] <- design_cov[index, index]
    metrics[[paste0("deff#", names_out[index])]] <- design_effects[index]
  }
  diag_out <- sift$.merge_method_diagnostics(list(
    weight_distribution = metadata$weight_cv,
    design_effect = max(design_effects),
    effective_sample_size = metadata$effective_n,
    strata_psu_support = metadata$metrics$design_df >= 1,
    variance_estimator = "pass",
    lonely_psu = if (metadata$metrics$lonely_strata_count == 0) "pass" else "warn"
  ), diagnostics)
  sift$from_method(
    "survey_regression", n = length(metadata$weights), diagnostics = diag_out,
    estimates = as.list(estimates), standard_errors = as.list(standard_errors),
    p_values = as.list(p_values), ci_lower = as.list(lower),
    ci_upper = as.list(upper), metrics = metrics,
    weight_type = "probability", variance_method = metadata$variance_method,
    lonely_psu_handling = lonely_psu, uncertainty_type = "design_based", ...
  )
}

sift$.reliability_statistics <- function(items) {
  if (!requireNamespace("psych", quietly = TRUE)) {
    stop("the R psych package is required for reliability analysis")
  }
  values <- as.matrix(items)
  storage.mode(values) <- "double"
  if (ncol(values) < 3L || any(!is.finite(values)) || any(apply(values, 2, stats::sd) <= 0)) {
    stop("reliability items must contain at least three finite non-constant columns")
  }
  standardized <- scale(values)
  item_rest <- vapply(seq_len(ncol(values)), function(index) {
    stats::cor(standardized[, index], rowSums(standardized[, -index, drop = FALSE]))
  }, numeric(1))
  if (any(!is.finite(item_rest)) || min(item_rest) < 0) {
    stop("negative item-rest correlation detected; explicitly declare reverse_items")
  }
  alpha_fit <- psych::alpha(values, check.keys = FALSE, warnings = FALSE)
  omega_fit <- suppressMessages(psych::omega(values, nfactors = 1, plot = FALSE))
  alpha <- as.numeric(alpha_fit$total$std.alpha)[1]
  omega <- as.numeric(omega_fit$omega.tot)[1]
  if (!is.finite(alpha) || !is.finite(omega) || alpha < 0 || alpha > 1 ||
      omega < 0 || omega > 1) stop("reliability coefficient is inadmissible")
  c(alpha = alpha, omega_total = omega, min_item_rest_correlation = min(item_rest))
}

#' Emit standardized alpha and one-factor omega with bootstrap intervals.
sift$from_reliability <- function(
    items, reverse_items = integer(0), bootstrap_replicates = 500L,
    seed = 20260822L, diagnostics = NULL, ...) {
  values <- as.matrix(items)
  storage.mode(values) <- "double"
  if (nrow(values) < 10L || ncol(values) < 3L || any(!is.finite(values))) {
    stop("reliability requires at least 10 complete rows and three items")
  }
  reversed <- as.integer(reverse_items)
  if (length(reversed) && (any(is.na(reversed)) || anyDuplicated(reversed) ||
      any(reversed < 1L | reversed > ncol(values)))) {
    stop("reverse_items must contain unique one-based item indexes")
  }
  for (index in reversed) {
    column <- values[, index]
    if (max(column) == min(column)) stop("a reversed item is constant")
    values[, index] <- min(column) + max(column) - column
  }
  repetitions <- as.integer(bootstrap_replicates)
  safe_seed <- as.integer(seed)
  if (is.na(repetitions) || repetitions < 200L || repetitions > 5000L ||
      is.na(safe_seed) || safe_seed < 0L) {
    stop("bootstrap_replicates must be 200..5000 and seed non-negative")
  }
  point <- sift$.reliability_statistics(values)
  set.seed(safe_seed)
  boot <- matrix(NA_real_, nrow = repetitions, ncol = 2L)
  for (iteration in seq_len(repetitions)) {
    indexes <- sample.int(nrow(values), nrow(values), replace = TRUE)
    candidate <- try(sift$.reliability_statistics(values[indexes, , drop = FALSE]), silent = TRUE)
    if (!inherits(candidate, "try-error")) boot[iteration, ] <- candidate[1:2]
  }
  keep <- stats::complete.cases(boot)
  successes <- sum(keep)
  if (successes < max(200L, ceiling(0.9 * repetitions))) {
    stop("fewer than 90% of reliability bootstrap fits were admissible")
  }
  intervals <- apply(boot[keep, , drop = FALSE], 2, stats::quantile,
                     probs = c(0.025, 0.975), names = FALSE)
  intervals[1, 1] <- min(intervals[1, 1], point[["alpha"]])
  intervals[2, 1] <- max(intervals[2, 1], point[["alpha"]])
  intervals[1, 2] <- min(intervals[1, 2], point[["omega_total"]])
  intervals[2, 2] <- max(intervals[2, 2], point[["omega_total"]])
  diag <- sift$.merge_method_diagnostics(list(
    sampling_adequacy = if (nrow(values) >= 10L * ncol(values)) "pass" else "warn",
    fit_or_stability = if (successes >= ceiling(0.95 * repetitions)) "pass" else "warn",
    component_or_class_support = "pass", item_count = ncol(values),
    omega_or_alpha_interval = "pass", item_direction = "pass"
  ), diagnostics)
  sift$from_method(
    "reliability", n = nrow(values), diagnostics = diag,
    estimates = list(alpha = point[["alpha"]], omega_total = point[["omega_total"]]),
    ci_lower = list(alpha = intervals[1, 1], omega_total = intervals[1, 2]),
    ci_upper = list(alpha = intervals[2, 1], omega_total = intervals[2, 2]),
    metrics = list(item_count = ncol(values), reversed_item_count = length(reversed),
                   min_item_rest_correlation = point[["min_item_rest_correlation"]],
                   bootstrap_replicates = repetitions,
                   bootstrap_success_count = successes),
    seed = safe_seed, uncertainty_type = "bootstrap", ...
  )
}

#' Emit aggregate CFA loadings and global fit indices from lavaan::cfa.
sift$from_lavaan_cfa <- function(fit, diagnostics = NULL, ...) {
  if (!requireNamespace("lavaan", quietly = TRUE)) stop("the R lavaan package is required")
  if (!inherits(fit, "lavaan") || !isTRUE(lavaan::lavInspect(fit, "converged"))) {
    stop("fit must be a converged lavaan CFA model")
  }
  table <- lavaan::parameterEstimates(fit)
  loadings <- table[table$op == "=~", , drop = FALSE]
  factor_sizes <- table(loadings$lhs)
  if (!nrow(loadings) || min(factor_sizes) < 3L) {
    stop("each confirmatory factor must have at least three indicators")
  }
  keys <- paste0("loading_", seq_len(nrow(loadings)))
  estimates <- stats::setNames(as.list(loadings$est), keys)
  standard_errors <- stats::setNames(as.list(loadings$se), keys)
  p_values <- stats::setNames(as.list(loadings$pvalue), keys)
  keep_se <- is.finite(loadings$se) & loadings$se >= 0
  keep_p <- is.finite(loadings$pvalue) & loadings$pvalue >= 0 & loadings$pvalue <= 1
  standard_errors <- standard_errors[keep_se]
  p_values <- p_values[keep_p]
  intervals <- sift$.normal_intervals(unlist(estimates), unlist(standard_errors))
  measures <- lavaan::fitMeasures(fit, c("cfi", "tli", "rmsea", "srmr", "chisq", "df"))
  if (any(!is.finite(measures)) || measures[["df"]] <= 0 || measures[["chisq"]] < 0 ||
      measures[["rmsea"]] < 0 || measures[["srmr"]] < 0 || measures[["srmr"]] > 1) {
    stop("lavaan CFA fit indices are invalid or the model is not identified")
  }
  n <- sum(as.numeric(lavaan::lavInspect(fit, "nobs")))
  indicator_count <- length(lavaan::lavNames(fit, type = "ov"))
  fit_status <- if (measures[["cfi"]] >= .90 && measures[["tli"]] >= .90 &&
                    measures[["rmsea"]] <= .08 && measures[["srmr"]] <= .08) "pass" else "warn"
  diag <- sift$.merge_method_diagnostics(list(
    sampling_adequacy = if (n >= 10L * indicator_count) "pass" else "warn",
    fit_or_stability = fit_status, component_or_class_support = "pass",
    cfi = measures[["cfi"]], tli = measures[["tli"]],
    rmsea = measures[["rmsea"]], srmr = measures[["srmr"]]
  ), diagnostics)
  sift$from_method(
    "confirmatory_factor_analysis", n = n, diagnostics = diag,
    estimates = estimates, standard_errors = standard_errors,
    p_values = p_values, ci_lower = as.list(intervals$lower),
    ci_upper = as.list(intervals$upper),
    metrics = list(factor_count = length(factor_sizes),
                   indicator_count = indicator_count, loading_count = nrow(loadings),
                   degrees_of_freedom = measures[["df"]],
                   chi_square = measures[["chisq"]], cfi = measures[["cfi"]],
                   tli = measures[["tli"]], rmsea = measures[["rmsea"]],
                   srmr = measures[["srmr"]]),
    uncertainty_type = "classical", ...
  )
}

#' Emit nested configural/metric/scalar lavaan invariance comparisons.
sift$.lavaan_invariance_contract <- function(fit) {
  options <- lavaan::lavInspect(fit, "options")
  partable <- lavaan::parTable(fit)
  structural <- partable[partable$op %in% c("=~", "~", "~~", "~1", "|"),
                          c("group", "lhs", "op", "rhs"), drop = FALSE]
  structural_signature <- sort(apply(structural, 1, paste, collapse = "\037"))
  equality_rows <- sum(partable$op == "==")
  labelled <- partable[nzchar(partable$label), c("group", "label"), drop = FALSE]
  repeated_labels <- if (nrow(labelled)) {
    sum(vapply(split(labelled$group, labelled$label),
               function(value) length(unique(value)) > 1L, logical(1)))
  } else 0L
  fitted_data <- try(lavaan::lavInspect(fit, "data"), silent = TRUE)
  case_index <- try(lavaan::lavInspect(fit, "case.idx"), silent = TRUE)
  if (inherits(fitted_data, "try-error") || inherits(case_index, "try-error")) {
    stop("lavaan fit does not expose fitted sample identity")
  }
  list(
    group_equal = sort(as.character(options$group.equal)),
    group_partial = sort(as.character(options$group.partial)),
    estimator = as.character(options$estimator),
    missing = as.character(options$missing),
    meanstructure = isTRUE(options$meanstructure),
    group_labels = as.character(lavaan::lavInspect(fit, "group.label")),
    structural_signature = structural_signature,
    equality_constraint_count = equality_rows + repeated_labels,
    data = fitted_data, case_index = case_index
  )
}

sift$from_lavaan_invariance <- function(
    configural, metric, scalar, diagnostics = NULL, ...) {
  if (!requireNamespace("lavaan", quietly = TRUE)) stop("the R lavaan package is required")
  fits <- list(configural, metric, scalar)
  if (any(!vapply(fits, inherits, logical(1), what = "lavaan")) ||
      any(!vapply(fits, function(x) isTRUE(lavaan::lavInspect(x, "converged")), logical(1)))) {
    stop("configural, metric, and scalar must be converged lavaan models")
  }
  nobs <- vapply(fits, function(x) sum(as.numeric(lavaan::lavInspect(x, "nobs"))), numeric(1))
  observed <- lapply(fits, lavaan::lavNames, type = "ov")
  groups <- vapply(fits, function(x) as.numeric(lavaan::lavInspect(x, "ngroups")), numeric(1))
  contracts <- lapply(fits, sift$.lavaan_invariance_contract)
  if (length(unique(nobs)) != 1L || groups[1] < 2L || length(unique(groups)) != 1L ||
      !identical(observed[[1]], observed[[2]]) || !identical(observed[[1]], observed[[3]])) {
    stop("invariance models must use the same observations, indicators, and at least two groups")
  }
  expected_equal <- list(character(0), "loadings", sort(c("intercepts", "loadings")))
  if (any(!vapply(seq_along(contracts), function(index) {
        identical(contracts[[index]]$group_equal, expected_equal[[index]]) &&
          length(contracts[[index]]$group_partial) == 0L
      }, logical(1)))) {
    stop("invariance sequence must be configural, loadings-equal, then loadings-and-intercepts-equal")
  }
  identity_fields <- c("estimator", "missing", "meanstructure", "group_labels",
                       "structural_signature", "data", "case_index")
  if (any(!vapply(identity_fields, function(field) {
        identical(contracts[[1]][[field]], contracts[[2]][[field]]) &&
          identical(contracts[[1]][[field]], contracts[[3]][[field]])
      }, logical(1)))) {
    stop("invariance models do not share the exact model structure and fitted sample")
  }
  constraint_counts <- vapply(contracts, function(contract) {
    contract$equality_constraint_count
  }, numeric(1))
  if (!(constraint_counts[1] < constraint_counts[2] &&
        constraint_counts[2] < constraint_counts[3])) {
    stop("lavaan partables do not prove increasingly constrained loading/intercept equality")
  }
  fm <- t(vapply(fits, function(x) lavaan::fitMeasures(
    x, c("cfi", "rmsea", "chisq", "df")), numeric(4)))
  if (any(!is.finite(fm)) || !(fm[1, "df"] < fm[2, "df"] && fm[2, "df"] < fm[3, "df"])) {
    stop("metric and scalar models must be increasingly constrained nested models")
  }
  nested <- lavaan::lavTestLRT(configural, metric, scalar)
  probability_column <- grep("Pr\\(>Chisq\\)", names(nested), value = TRUE)
  if (length(probability_column) != 1L || nrow(nested) != 3L) {
    stop("lavaan nested-model comparison did not return two chi-square tests")
  }
  nested_p <- as.numeric(nested[[probability_column]])[2:3]
  if (any(!is.finite(nested_p)) || any(nested_p < 0 | nested_p > 1)) stop("nested p-values invalid")
  # Matrix extraction retains the source column name (for example `cfi`).
  # Strip it before naming the deltas so R cannot silently create names such
  # as `metric_delta_cfi.cfi`, which would defeat the exact contract below.
  changes <- c(
    metric_delta_cfi = as.numeric(fm[1, "cfi"] - fm[2, "cfi"]),
    metric_delta_rmsea = as.numeric(fm[2, "rmsea"] - fm[1, "rmsea"]),
    scalar_delta_cfi = as.numeric(fm[2, "cfi"] - fm[3, "cfi"]),
    scalar_delta_rmsea = as.numeric(fm[3, "rmsea"] - fm[2, "rmsea"])
  )
  metric_ok <- changes[["metric_delta_cfi"]] <= .01 && changes[["metric_delta_rmsea"]] <= .015
  scalar_ok <- changes[["scalar_delta_cfi"]] <= .01 && changes[["scalar_delta_rmsea"]] <= .015
  diag <- sift$.merge_method_diagnostics(list(
    sampling_adequacy = if (nobs[1] >= 10L * length(observed[[1]])) "pass" else "warn",
    fit_or_stability = "pass", component_or_class_support = "pass",
    configural_fit = if (fm[1, "cfi"] >= .90 && fm[1, "rmsea"] <= .08) "pass" else "warn",
    metric_change = if (metric_ok) "pass" else "warn",
    scalar_change = if (scalar_ok) "pass" else "warn"
  ), diagnostics)
  metrics <- list(group_count = groups[1], indicator_count = length(observed[[1]]))
  for (index in seq_len(3L)) {
    prefix <- c("configural", "metric", "scalar")[index]
    for (field in c("cfi", "rmsea", "chisq", "df")) {
      metrics[[paste0(prefix, "_", field)]] <- fm[index, field]
    }
  }
  sift$from_method(
    "measurement_invariance", n = nobs[1], diagnostics = diag,
    estimates = as.list(changes),
    p_values = list(metric_nested = nested_p[1], scalar_nested = nested_p[2]),
    metrics = metrics, uncertainty_type = "classical", ...
  )
}

#' Emit a stable multi-start poLCA solution without posterior rows.
sift$from_polca <- function(
    fits, likelihood_tolerance = 1e-4, minimum_class_n = 10L,
    diagnostics = NULL, ...) {
  if (!requireNamespace("poLCA", quietly = TRUE)) stop("the R poLCA package is required")
  if (!is.list(fits) || length(fits) < 5L ||
      any(!vapply(fits, inherits, logical(1), what = "poLCA"))) {
    stop("latent class reporting requires at least five fitted poLCA starts")
  }
  class_count <- function(x) {
    # poLCA 1.6.0.2 does not retain the input `nclass` as a top-level field.
    # Prove it from independent fitted structures instead: class proportions,
    # posterior columns, and every manifest response-probability matrix.
    candidates <- c(length(x$P), if (is.matrix(x$posterior)) ncol(x$posterior) else 0L)
    probability_rows <- if (is.list(x$probs) && length(x$probs)) {
      vapply(x$probs, function(value) {
        if (!is.matrix(value)) return(0L)
        as.integer(nrow(value))
      }, integer(1))
    } else integer()
    candidates <- as.integer(c(candidates, probability_rows))
    if (!length(candidates) || any(is.na(candidates)) || any(candidates < 2L) ||
        length(unique(candidates)) != 1L) {
      stop("poLCA fit does not consistently prove its fitted class count")
    }
    candidates[1]
  }
  classes <- vapply(fits, class_count, integer(1))
  sample_sizes <- vapply(fits, function(x) as.integer(x$N), integer(1))
  loglik <- vapply(fits, function(x) as.numeric(x$llik), numeric(1))
  if (length(unique(classes)) != 1L || classes[1] < 2L ||
      length(unique(sample_sizes)) != 1L || any(!is.finite(loglik))) {
    stop("poLCA starts must fit the same sample and class count")
  }
  formulas <- vapply(fits, function(x) paste(deparse(x$call$formula), collapse = ""), character(1))
  manifest_names <- lapply(fits, function(x) names(x$y))
  covariate_names <- lapply(fits, function(x) names(x$x))
  if (length(unique(formulas)) != 1L ||
      any(!vapply(fits[-1], function(x) identical(x$y, fits[[1]]$y) &&
                    identical(x$x, fits[[1]]$x), logical(1))) ||
      any(!vapply(manifest_names[-1], identical, logical(1), manifest_names[[1]])) ||
      any(!vapply(covariate_names[-1], identical, logical(1), covariate_names[[1]]))) {
    stop("poLCA starts must use the same formula, manifest variables, covariates, and fitted sample")
  }
  for (left in seq_len(length(fits) - 1L)) {
    for (right in seq.int(left + 1L, length(fits))) {
      if (identical(fits[[left]], fits[[right]])) {
        stop("multi-start evidence cannot repeat the same fitted poLCA object")
      }
    }
  }
  starts <- lapply(fits, function(x) x$probs.start)
  if (any(vapply(starts, is.null, logical(1)))) {
    stop("poLCA fits do not retain their realized random starting probabilities")
  }
  for (left in seq_len(length(starts) - 1L)) {
    for (right in seq.int(left + 1L, length(starts))) {
      if (identical(starts[[left]], starts[[right]])) {
        stop("poLCA multi-start fits must use distinct realized starting probabilities")
      }
    }
  }
  tolerance <- as.numeric(likelihood_tolerance)
  if (!is.finite(tolerance) || tolerance <= 0) stop("likelihood_tolerance must be positive")
  best_index <- which.max(loglik); best <- fits[[best_index]]
  stable <- sum(max(loglik) - loglik <= tolerance)
  if (stable < 2L) stop("multi-start latent-class solution was not reproduced")
  posterior <- as.matrix(best$posterior)
  if (nrow(posterior) != sample_sizes[1] || ncol(posterior) != classes[1] ||
      any(!is.finite(posterior)) || any(posterior < 0 | posterior > 1) ||
      max(abs(rowSums(posterior) - 1)) > 1e-6) stop("poLCA posterior matrix is invalid")
  expected_sizes <- colSums(posterior)
  minimum <- as.integer(minimum_class_n)
  if (min(expected_sizes) < minimum) stop("a latent class is below minimum expected support")
  entropy <- 1 + sum(posterior * log(pmax(posterior, .Machine$double.eps))) /
    (nrow(posterior) * log(ncol(posterior)))
  estimates <- as.list(stats::setNames(expected_sizes / sum(expected_sizes),
                                       paste0("class_", seq_along(expected_sizes))))
  ordered <- sort(loglik, decreasing = TRUE)
  diag <- sift$.merge_method_diagnostics(list(
    sampling_adequacy = if (sample_sizes[1] >= 10L * classes[1]) "pass" else "warn",
    fit_or_stability = "pass", component_or_class_support = "pass",
    class_sizes = "pass", entropy = if (entropy >= .6) "pass" else "warn",
    solution_stability = "pass"
  ), diagnostics)
  sift$from_method(
    "latent_class", n = sample_sizes[1], diagnostics = diag,
    estimates = estimates,
    metrics = list(class_count = classes[1], start_count = length(fits),
                   stable_start_count = stable, min_expected_class_n = min(expected_sizes),
                   normalized_entropy = entropy, best_log_likelihood = ordered[1],
                   second_best_gap = ordered[1] - ordered[2],
                   likelihood_tolerance = tolerance,
                   aic = as.numeric(best$aic), bic = as.numeric(best$bic)),
    uncertainty_type = "classical", ...
  )
}

#' Compute and emit a Student-t confidence interval for a sample mean.
#'
#' The raw vector is used only inside the sandbox. Only N, mean, standard
#' error, missing count and confidence limits cross the result boundary.
sift$from_descriptive_confidence_interval <- function(
    x, name = "mean", confidence = 0.95, diagnostics = NULL, ...) {
  confidence <- as.numeric(confidence)
  if (length(confidence) != 1L || !is.finite(confidence) ||
      confidence <= 0 || confidence >= 1) {
    stop("confidence must be strictly between 0 and 1")
  }
  values <- as.numeric(x)
  clean <- values[is.finite(values)]
  if (length(clean) < 2L) stop("at least two finite observations are required")
  fit <- stats::t.test(clean, conf.level = confidence)
  n <- length(clean)
  diag <- sift$.merge_method_diagnostics(list(
    missingness = as.integer(length(values) - n),
    effective_sample_size = as.integer(n),
    confidence_level = confidence
  ), diagnostics)
  key <- sift$.method_quantity_name(name)
  sift$from_method(
    "descriptive_confidence_interval", n = n, diagnostics = diag,
    estimates = setNames(list(as.numeric(fit$estimate)), key),
    standard_errors = setNames(list(as.numeric(stats::sd(clean) / sqrt(n))), key),
    ci_lower = setNames(list(as.numeric(fit$conf.int[1])), key),
    ci_upper = setNames(list(as.numeric(fit$conf.int[2])), key),
    uncertainty_type = "classical", ...
  )
}

#' Emit a rank-based `htest` from wilcox.test or kruskal.test.
sift$from_nonparametric_test <- function(
    fit, n, name = "rank_test", group_sizes = NULL,
    ties_checked = "not_applicable", diagnostics = NULL, ...) {
  if (!inherits(fit, "htest")) stop("fit must be an htest result")
  n <- sift$.positive_count(n, "n")
  size_status <- "not_applicable"
  if (!is.null(group_sizes)) {
    sizes <- as.integer(group_sizes)
    size_status <- length(sizes) > 0L && all(!is.na(sizes)) &&
      all(sizes >= 0L) && sum(sizes) == n
  }
  diag <- sift$.merge_method_diagnostics(list(
    group_sample_sizes = size_status,
    ties_and_zero_differences = ties_checked
  ), diagnostics)
  key <- sift$.method_quantity_name(name)
  sift$from_method(
    "nonparametric_test", n = n, diagnostics = diag,
    estimates = setNames(list(as.numeric(fit$statistic)[1]), key),
    p_values = setNames(list(as.numeric(fit$p.value)[1]), key),
    uncertainty_type = "classical", ...
  )
}

#' Emit an R prop.test result with aggregate expected-count diagnostics.
sift$from_proportion_test <- function(
    fit, nobs, name = "proportion", diagnostics = NULL, ...) {
  if (!inherits(fit, "htest")) stop("fit must be a prop.test htest result")
  totals <- as.numeric(nobs)
  props <- as.numeric(fit$estimate)
  if (length(totals) != length(props) || any(!is.finite(totals)) ||
      any(totals <= 0)) stop("nobs must match the fitted proportions")
  totals <- vapply(
    totals, sift$.positive_count, integer(1), field = "each nobs value"
  )
  estimate <- if (length(props) == 1L) props[1] else props[1] - props[2]
  successes <- props * totals
  expected_ok <- all(successes >= 5) && all(totals - successes >= 5)
  diag <- sift$.merge_method_diagnostics(list(
    group_sample_sizes = "pass",
    expected_cell_counts = if (expected_ok) "pass" else "warn"
  ), diagnostics)
  key <- sift$.method_quantity_name(name)
  args <- list(
    method_id = "proportion_test", n = as.integer(sum(totals)),
    diagnostics = diag,
    estimates = setNames(list(as.numeric(estimate)), key),
    p_values = setNames(list(as.numeric(fit$p.value)), key),
    metrics = list(chi_squared = as.numeric(fit$statistic)),
    uncertainty_type = "classical"
  )
  if (!is.null(fit$conf.int) && length(fit$conf.int) == 2L) {
    args$ci_lower <- setNames(list(as.numeric(fit$conf.int[1])), key)
    args$ci_upper <- setNames(list(as.numeric(fit$conf.int[2])), key)
  }
  do.call(sift$from_method, c(args, list(...)))
}

#' Emit an ANOVA or ANCOVA table from a fitted base-R lm/aov model.
sift$from_anova <- function(
    fit, method_id = "anova", diagnostics = NULL, ...) {
  if (!method_id %in% c("anova", "ancova")) {
    stop("method_id must be 'anova' or 'ancova'")
  }
  tab <- tryCatch(stats::anova(fit), error = function(e) NULL)
  if (is.null(tab)) stop("fit must support stats::anova")
  effects <- rownames(tab)
  keep <- !grepl("residual", effects, ignore.case = TRUE)
  f_col <- grep("^F$|F value", colnames(tab), value = TRUE)[1]
  p_col <- grep("Pr\\(>F\\)", colnames(tab), value = TRUE)[1]
  metrics <- list()
  p_values <- list()
  if (!is.na(f_col)) {
    metrics <- as.list(as.numeric(tab[keep, f_col]))
    names(metrics) <- effects[keep]
  }
  if (!is.na(p_col)) {
    p_values <- as.list(as.numeric(tab[keep, p_col]))
    names(p_values) <- effects[keep]
  }
  invisible(lapply(names(metrics), sift$.method_quantity_name,
                   field = "ANOVA effect name"))
  invisible(lapply(names(p_values), sift$.method_quantity_name,
                   field = "ANOVA effect name"))
  defaults <- list(
    group_sample_sizes = "not_applicable",
    residual_distribution = "warn",
    homogeneity_of_variance = "warn"
  )
  if (method_id == "ancova") defaults$parallel_slopes <- "warn"
  diag <- sift$.merge_method_diagnostics(defaults, diagnostics)
  fit_n <- sift$.positive_count(stats::nobs(fit), "model nobs")
  sift$from_method(
    method_id, n = fit_n, diagnostics = diag,
    metrics = metrics, p_values = p_values,
    uncertainty_type = "classical", ...
  )
}

#' Emit a repeated-measures omnibus htest, including subject/record counts.
#'
#' `friedman.test` is supported directly; more detailed mixed-effects work can
#' continue to use from_method after fitting nlme::lme.
sift$from_repeated_measures <- function(
    fit, n, subjects, records = n, diagnostics = NULL, ...) {
  if (!inherits(fit, "htest")) stop("fit must be an htest result")
  n <- sift$.positive_count(n, "n")
  subjects <- sift$.positive_count(subjects, "subjects")
  records <- sift$.positive_count(records, "records")
  if (subjects > records || records > n) {
    stop("require subjects <= records <= n")
  }
  diag <- sift$.merge_method_diagnostics(list(
    cluster_count = subjects,
    cluster_size = as.numeric(records / subjects),
    complete_cases = if (records == n) "pass" else "warn",
    sphericity_or_correction = "not_applicable"
  ), diagnostics)
  sift$from_method(
    "repeated_measures_test", n = n, subjects = subjects,
    records = records, diagnostics = diag,
    metrics = list(omnibus_statistic = as.numeric(fit$statistic)[1]),
    p_values = list(omnibus = as.numeric(fit$p.value)[1]),
    uncertainty_type = "classical", ...
  )
}

#' Apply stats::p.adjust and emit bounded raw/adjusted p-value maps.
sift$from_multiple_testing <- function(
    p_values, n, method = "holm", alpha = 0.05, labels = NULL, ...) {
  method_map <- c(
    holm = "holm", bonferroni = "bonferroni",
    benjamini_hochberg = "BH"
  )
  if (!method %in% names(method_map)) {
    stop("method must be holm, bonferroni, or benjamini_hochberg")
  }
  n <- sift$.positive_count(n, "n")
  alpha <- as.numeric(alpha)
  if (length(alpha) != 1L || !is.finite(alpha) || alpha <= 0 || alpha >= 1) {
    stop("alpha must be strictly between 0 and 1")
  }
  values <- as.numeric(p_values)
  if (length(values) < 1L || length(values) > 100L ||
      any(!is.finite(values)) || any(values < 0 | values > 1)) {
    stop("p_values must contain 1..100 finite probabilities")
  }
  if (is.null(labels)) labels <- paste0("hypothesis_", seq_along(values))
  labels <- as.character(labels)
  if (length(labels) != length(values)) stop("labels length must match p_values")
  labels <- vapply(labels, sift$.method_quantity_name, character(1), field = "label")
  if (anyDuplicated(labels)) stop("labels must be unique")
  corrected <- stats::p.adjust(values, method = unname(method_map[[method]]))
  estimates <- as.list(values); names(estimates) <- labels
  adjusted <- as.list(corrected); names(adjusted) <- labels
  sift$from_method(
    "multiple_testing_correction", n = n,
    diagnostics = list(
      hypothesis_family = "pass", correction_applied = "pass"
    ),
    estimates = estimates, p_values = adjusted,
    metrics = list(
      hypothesis_count = as.numeric(length(values)),
      rejection_count = as.numeric(sum(corrected <= alpha)),
      alpha = as.numeric(alpha)
    ),
    multiple_testing = method, ...
  )
}

# Missing-data helpers emit bounded aggregates only.  Joint row-level
# patterns and imputed values never cross the Sift result boundary.
sift$from_missingness_pattern <- function(
    data, complete_case_warning_threshold = 0.10, diagnostics = NULL, ...) {
  if (!is.data.frame(data) || nrow(data) < 1L || ncol(data) < 1L || ncol(data) > 100L) {
    stop("data must be a non-empty data.frame with 1..100 columns")
  }
  threshold <- as.numeric(complete_case_warning_threshold)
  if (length(threshold) != 1L || !is.finite(threshold) || threshold <= 0 || threshold >= 1) {
    stop("complete_case_warning_threshold must be between 0 and 1")
  }
  missing <- is.na(data)
  complete_rate <- mean(rowSums(missing) == 0L)
  pattern_count <- nrow(unique(as.data.frame(missing)))
  pattern_sizes <- table(interaction(as.data.frame(missing), drop = TRUE))
  diag <- sift$.merge_method_diagnostics(list(
    missingness_pattern = "pass", complete_case_rate = complete_rate,
    complete_case_warning = if ((1 - complete_rate) >= threshold) "warn" else "pass"
  ), diagnostics)
  sift$from_method(
    "missingness_pattern", n = nrow(data), diagnostics = diag,
    metrics = as.list(c(
      variable_count = ncol(data), missing_fraction = mean(missing),
      complete_case_rate = complete_rate,
      complete_case_warning_threshold = threshold,
      missingness_pattern_count = pattern_count,
      largest_pattern_fraction = max(pattern_sizes) / nrow(data)
    )), ...
  )
}

sift$from_single_imputation <- function(
    data, scope, strategy = "median", diagnostics = NULL, ...) {
  if (!is.data.frame(data) && !is.matrix(data)) {
    stop("data must be a numeric data.frame or matrix")
  }
  matrix_data <- as.matrix(data)
  storage.mode(matrix_data) <- "double"
  if (nrow(matrix_data) < 1L || ncol(matrix_data) < 1L || ncol(matrix_data) > 100L) {
    stop("data must be non-empty with 1..100 columns")
  }
  if (!scope %in% c("prediction_preprocessing", "deterministic_nuisance_covariate")) {
    stop("scope must be prediction_preprocessing or deterministic_nuisance_covariate")
  }
  if (!strategy %in% c("mean", "median")) stop("strategy must be mean or median")
  missing <- is.na(matrix_data)
  if (!any(missing)) stop("single-imputation audit requires missing values")
  completed <- matrix_data
  for (column in seq_len(ncol(completed))) {
    if (!any(missing[, column])) next
    observed <- completed[!missing[, column], column]
    if (!length(observed)) stop("single imputation refuses an all-missing column")
    fill <- if (strategy == "mean") mean(observed) else stats::median(observed)
    completed[missing[, column], column] <- fill
  }
  if (any(!is.finite(completed))) stop("imputation produced non-finite values")
  diag <- sift$.merge_method_diagnostics(list(
    missingness_pattern = "pass", imputation_scope = "pass",
    inferential_uncertainty_not_claimed = "pass"
  ), diagnostics)
  sift$from_method(
    "single_imputation", n = nrow(completed), diagnostics = diag,
    metrics = as.list(c(
      feature_count = ncol(completed), output_feature_count = ncol(completed),
      missing_fraction = mean(missing),
      affected_row_fraction = mean(rowSums(missing) > 0L),
      imputed_cell_count = sum(missing)
    )), imputation_scope = scope, imputation_model = "simple_deterministic", ...
  )
  invisible(completed)
}

# ---------------------------------------------------------------------------
# Causal-design helpers — only aggregate diagnostics cross the boundary
# ---------------------------------------------------------------------------

sift$.causal_status <- function(value, field) {
  allowed <- c("pass", "warn", "fail", "not_applicable")
  if (!(is.logical(value) && length(value) == 1L && !is.na(value)) &&
      !(is.character(value) && length(value) == 1L && value %in% allowed)) {
    stop(field, " must be a diagnostic status")
  }
  value
}

sift$.causal_arrays <- function(X, treatment, outcome) {
  X <- as.matrix(X); storage.mode(X) <- "double"
  treatment <- as.integer(treatment); outcome <- as.numeric(outcome)
  if (nrow(X) != length(treatment) || nrow(X) != length(outcome) || nrow(X) < 20L ||
      any(!is.finite(X)) || any(!is.finite(outcome)) ||
      !identical(sort(unique(treatment)), c(0L, 1L))) {
    stop("X, binary treatment, and outcome require at least 20 aligned finite rows")
  }
  if (min(table(treatment)) < 10L) stop("each treatment arm requires at least 10 observations")
  list(X = X, treatment = treatment, outcome = outcome)
}

sift$.max_abs_smd <- function(X, treatment, weights = rep(1, length(treatment))) {
  values <- vapply(seq_len(ncol(X)), function(j) {
    stats <- lapply(c(1L, 0L), function(arm) {
      keep <- treatment == arm; w <- weights[keep]; z <- X[keep, j]
      m <- stats::weighted.mean(z, w)
      v <- stats::weighted.mean((z - m)^2, w)
      c(mean = m, var = v)
    })
    scale <- sqrt(max((stats[[1]]["var"] + stats[[2]]["var"]) / 2, 0))
    difference <- abs(stats[[1]]["mean"] - stats[[2]]["mean"])
    if (scale == 0 && difference == 0) 0 else difference / max(scale, 1e-12)
  }, numeric(1))
  max(values)
}

sift$.propensity_scores <- function(X, treatment) {
  frame <- as.data.frame(X)
  frame$.treatment <- treatment
  fit <- stats::glm(.treatment ~ ., data = frame, family = stats::binomial())
  scores <- as.numeric(stats::predict(fit, type = "response"))
  if (any(!is.finite(scores))) stop("propensity model produced non-finite scores")
  scores
}

sift$.overlap_fraction <- function(scores, treatment) {
  lower <- max(min(scores[treatment == 1L]), min(scores[treatment == 0L]))
  upper <- min(max(scores[treatment == 1L]), max(scores[treatment == 0L]))
  if (lower >= upper) 0 else mean(scores >= lower & scores <= upper)
}

sift$from_propensity_matching <- function(
    X, treatment, outcome, estimand = "att", falsification_status, ...) {
  if (estimand != "att") stop("propensity nearest-neighbour matching identifies ATT only")
  values <- sift$.causal_arrays(X, treatment, outcome)
  X <- values$X; treatment <- values$treatment; outcome <- values$outcome
  scores <- sift$.propensity_scores(X, treatment)
  treated <- which(treatment == 1L); controls <- which(treatment == 0L)
  matched <- vapply(treated, function(i) controls[which.min(abs(scores[controls] - scores[i]))], integer(1))
  differences <- outcome[treated] - outcome[matched]
  effect <- mean(differences)
  matched_X <- rbind(X[treated, , drop = FALSE], X[matched, , drop = FALSE])
  matched_t <- c(rep(1L, length(treated)), rep(0L, length(treated)))
  before <- sift$.max_abs_smd(X, treatment)
  after <- sift$.max_abs_smd(matched_X, matched_t)
  overlap <- sift$.overlap_fraction(scores, treatment)
  effective <- length(treated) + length(unique(matched))
  sift$from_method(
    "matching", n = length(treatment), treated = length(treated), controls = length(controls),
    diagnostics = list(
      propensity_overlap = if (overlap >= .8) "pass" else "warn",
      standardized_mean_differences = if (after <= .1) "pass" else "warn",
      effective_matched_sample = as.numeric(effective),
      effect_uncertainty = "not_applicable",
      design_specific_falsification = sift$.causal_status(falsification_status, "falsification_status")
    ),
    estimates = list(att = effect),
    metrics = list(
      effect = effect, max_abs_smd_before = before, max_abs_smd_after = after,
      overlap_fraction = overlap, effective_sample_size = as.numeric(effective),
      treated_score_p05=as.numeric(stats::quantile(scores[treatment==1L],.05)),
      treated_score_p95=as.numeric(stats::quantile(scores[treatment==1L],.95)),
      control_score_p05=as.numeric(stats::quantile(scores[treatment==0L],.05)),
      control_score_p95=as.numeric(stats::quantile(scores[treatment==0L],.95))
    ), estimand = "att",
    design = "propensity_nearest_neighbor", ...
  )
}

sift$from_propensity_weighting <- function(
    X, treatment, outcome, estimand = "ate", falsification_status, ...) {
  if (!estimand %in% c("ate", "att")) stop("estimand must be ate or att")
  values <- sift$.causal_arrays(X, treatment, outcome)
  X <- values$X; treatment <- values$treatment; outcome <- values$outcome
  scores <- sift$.propensity_scores(X, treatment)
  if (any(scores <= 1e-6 | scores >= 1 - 1e-6)) stop("propensity scores violate numerical positivity")
  weights <- if (estimand == "ate") treatment/scores + (1-treatment)/(1-scores)
             else treatment + (1-treatment)*scores/(1-scores)
  keep_t <- treatment == 1L; keep_c <- !keep_t
  mean_t <- stats::weighted.mean(outcome[keep_t], weights[keep_t])
  mean_c <- stats::weighted.mean(outcome[keep_c], weights[keep_c])
  effect <- mean_t - mean_c
  ess <- sum(weights)^2/sum(weights^2)
  before <- sift$.max_abs_smd(X, treatment)
  after <- sift$.max_abs_smd(X, treatment, weights)
  overlap <- sift$.overlap_fraction(scores, treatment); max_weight <- max(weights)
  estimate <- setNames(list(effect), estimand)
  sift$from_method(
    "propensity_weighting", n = length(treatment), treated = sum(treatment),
    controls = sum(1-treatment), diagnostics = list(
      propensity_overlap = if (overlap >= .8) "pass" else "warn",
      weight_extremes = if (max_weight <= 10) "pass" else "warn",
      standardized_mean_differences = if (after <= .1) "pass" else "warn",
      effective_sample_size = ess,
      effect_uncertainty = "not_applicable",
      design_specific_falsification = sift$.causal_status(falsification_status, "falsification_status")
    ), estimates = estimate,
    metrics = list(
      effect = effect, max_abs_smd_before = before, max_abs_smd_after = after,
      overlap_fraction = overlap, effective_sample_size = ess, max_weight = max_weight,
      treated_score_p05=as.numeric(stats::quantile(scores[treatment==1L],.05)),
      treated_score_p95=as.numeric(stats::quantile(scores[treatment==1L],.95)),
      control_score_p05=as.numeric(stats::quantile(scores[treatment==0L],.05)),
      control_score_p95=as.numeric(stats::quantile(scores[treatment==0L],.95))
    ), estimand = estimand,
    design = "inverse_probability_weighting", ...
  )
}

sift$.synthetic_weights <- function(target, donors) {
  k <- ncol(donors)
  # Fix the final log-weight at zero. A full K-vector softmax has a flat
  # additive direction, which makes BFGS report non-convergence even when the
  # donor weights are already optimal.
  softmax <- function(theta) {
    full <- c(theta, 0); e <- exp(full - max(full)); e/sum(e)
  }
  objective <- function(theta) mean((target - donors %*% softmax(theta))^2)
  fit <- stats::optim(rep(0, k-1L), objective, method = "BFGS",
                      control = list(maxit = 20000, reltol = 1e-12))
  if (fit$convergence != 0L) stop("synthetic-control optimization failed")
  softmax(fit$par)
}

sift$from_synthetic_control <- function(
    treated_series, donor_series, intervention_index, falsification_status, ...) {
  treated <- as.numeric(treated_series); donors <- as.matrix(donor_series); storage.mode(donors) <- "double"
  if (nrow(donors) != length(treated) || any(!is.finite(treated)) || any(!is.finite(donors))) {
    stop("finite time-by-donor inputs must align")
  }
  pre <- sift$.positive_count(intervention_index, "intervention_index")
  post <- length(treated)-pre
  if (pre < 3L || post < 1L || ncol(donors) < 3L) stop("need >=3 pre, >=1 post, and >=3 donors")
  weights <- sift$.synthetic_weights(treated[seq_len(pre)], donors[seq_len(pre),,drop=FALSE])
  gap <- treated - as.numeric(donors %*% weights)
  pre_rmse <- sqrt(mean(gap[seq_len(pre)]^2)); post_rmse <- sqrt(mean(gap[(pre+1):length(gap)]^2))
  effect <- mean(gap[(pre+1):length(gap)]); ratio <- post_rmse/max(pre_rmse,1e-12)
  placebo_ratios <- numeric(ncol(donors))
  for (j in seq_len(ncol(donors))) {
    pool <- donors[, -j, drop=FALSE]
    w <- sift$.synthetic_weights(donors[seq_len(pre),j], pool[seq_len(pre),,drop=FALSE])
    pg <- donors[,j] - as.numeric(pool %*% w)
    pre_p <- sqrt(mean(pg[seq_len(pre)]^2)); post_p <- sqrt(mean(pg[(pre+1):length(pg)]^2))
    placebo_ratios[j] <- post_p/max(pre_p,1e-12)
  }
  placebo_p <- (1+sum(placebo_ratios >= ratio))/(1+length(placebo_ratios))
  max_weight <- max(weights); fit_ratio <- pre_rmse/max(stats::sd(treated[seq_len(pre)]),1e-12)
  n <- as.integer((ncol(donors)+1L)*length(treated))
  sift$from_method(
    "synthetic_control", n=n, donors=ncol(donors), pre_periods=pre, post_periods=post,
    diagnostics=list(
      pre_treatment_fit=if(fit_ratio<=.2) "pass" else "warn",
      placebo_distribution="pass",
      donor_weight_concentration=if(max_weight<=.8) "pass" else "warn",
      effect_uncertainty="not_applicable",
      design_specific_falsification=sift$.causal_status(falsification_status,"falsification_status")
    ), estimates=list(unit_time_att=effect),
    metrics=list(effect=effect,pre_rmse=pre_rmse,post_rmse=post_rmse,
                 placebo_p_value=placebo_p,max_donor_weight=max_weight),
    estimand="unit_time_att",design="synthetic_control",...
  )
}

sift$from_treatment_heterogeneity <- function(
    X, treatment, outcome, falsification_status, seed=42L, test_fraction=.4, ...) {
  if (!requireNamespace("rpart",quietly=TRUE)) stop("rpart is required")
  values <- sift$.causal_arrays(X,treatment,outcome)
  X<-values$X;treatment<-values$treatment;outcome<-values$outcome
  if(test_fraction<.2 || test_fraction>.5) stop("test_fraction must be between .2 and .5")
  set.seed(as.integer(seed)); train <- integer(); test <- integer()
  for(arm in c(0L,1L)) {
    ids<-which(treatment==arm); held<-sample(ids,ceiling(length(ids)*test_fraction))
    test<-c(test,held);train<-c(train,setdiff(ids,held))
  }
  frame<-as.data.frame(X);names(frame)<-paste0("x",seq_len(ncol(X)));frame$y<-outcome
  fits<-lapply(c(0L,1L),function(arm) rpart::rpart(y~.,data=frame[intersect(train,which(treatment==arm)),],
                                                   control=rpart::rpart.control(minbucket=5,cp=.001)))
  cate<-as.numeric(stats::predict(fits[[2]],frame[test,]))-as.numeric(stats::predict(fits[[1]],frame[test,]))
  average<-mean(cate); cate_sd<-stats::sd(cate); qs<-stats::quantile(cate,c(.25,.75))
  contrast<-mean(cate[cate>=qs[2]])-mean(cate[cate<=qs[1]])
  propensity_frame<-as.data.frame(X[train,,drop=FALSE])
  propensity_frame$.treatment<-treatment[train]
  pfit<-stats::glm(.treatment~.,data=propensity_frame,family=stats::binomial())
  ptest<-as.numeric(stats::predict(
    pfit,newdata=as.data.frame(X[test,,drop=FALSE]),type="response"
  )); ptest<-pmin(pmax(ptest,1e-3),1-1e-3)
  transformed<-(treatment[test]-ptest)*outcome[test]/(ptest*(1-ptest))
  calibration<-stats::cor(cate,transformed)
  if(!is.finite(calibration)) stop("heterogeneity calibration is undefined")
  ps<-as.numeric(stats::predict(pfit,type="response")); overlap<-sift$.overlap_fraction(ps,treatment[train])
  balance<-sift$.max_abs_smd(X[train,,drop=FALSE],treatment[train])
  sift$from_method(
    "treatment_effect_heterogeneity",n=length(treatment),seed=as.integer(seed),
    diagnostics=list(propensity_overlap=if(overlap>=.8)"pass" else "warn",
      standardized_mean_differences=if(balance<=.1)"pass" else "warn",honest_sample_splitting="pass",
      subgroup_multiplicity="not_applicable",heterogeneity_calibration=calibration,
      effect_uncertainty="not_applicable",
      design_specific_falsification=sift$.causal_status(falsification_status,"falsification_status")),
    estimates=list(average_predicted_cate=average),
    metrics=list(average_cate=average,cate_sd=cate_sd,q4_q1_contrast=contrast,
                 calibration_correlation=calibration,overlap_fraction=overlap,
                 max_abs_smd_before=balance),
    estimand="average_predicted_cate",design="honest_t_learner",...
  )
}

sift$.sensemakr_robustness_value <- function(t_statistic,dof,q=1,alpha=.05) {
  if(dof<=1||q<=0||alpha<=0||alpha>1)stop("dof>1, q>0, and 0<alpha<=1 are required")
  fq<-q*abs(t_statistic/sqrt(dof));fcrit<-abs(stats::qt(alpha/2,df=dof-1))/sqrt(dof-1)
  fqa<-fq-fcrit
  if(fqa<=0)return(0)
  binding<-2/(1+sqrt(1+4/fqa^2));fq2<-fq^2;fc2<-fcrit^2
  extreme<-if(fq2>fc2)(fq2-fc2)/(1+fq2) else 0
  if(fcrit>0&&fq>1/fcrit)extreme else binding
}

sift$from_causal_sensitivity <- function(
    model, coefficient, falsification_status, alpha=.05, q=1, ...) {
  name<-sift$.method_quantity_name(coefficient,"coefficient")
  table<-summary(model)$coefficients
  if(!name %in% rownames(table)) stop("coefficient is not present in model")
  n<-sift$.positive_count(stats::nobs(model),"model nobs");df<-stats::df.residual(model)
  if(!is.finite(alpha)||alpha<=0||alpha>=1||df<=1||!is.finite(q)||q<=0) stop("alpha, q, and residual df must be valid")
  tvalue<-abs(as.numeric(table[name,"t value"]))
  rv0<-sift$.sensemakr_robustness_value(tvalue,df,q=q,alpha=1)
  rva<-sift$.sensemakr_robustness_value(tvalue,df,q=q,alpha=alpha)
  estimate<-as.numeric(table[name,"Estimate"])
  sift$from_method(
    "causal_sensitivity",n=n,diagnostics=list(robustness_value=rv0,assumption_grid="pass",
      design_specific_falsification=sift$.causal_status(falsification_status,"falsification_status")),
    estimates=setNames(list(estimate),name),
    metrics=list(robustness_value_zero=rv0,robustness_value_alpha=rva,t_statistic=tvalue,
      q=q,alpha=alpha,margin_equal_r2_01=rv0-.01,margin_equal_r2_05=rv0-.05,
      margin_equal_r2_10=rv0-.10),
    estimand="robustness_value",design="omitted_variable_sensitivity",...
  )
}

# ---------------------------------------------------------------------------
# Time-series helpers — chronological evaluation, aggregate output only
# ---------------------------------------------------------------------------

sift$.time_series_values <- function(series,frequency,time_index=NULL,cadence=NULL,
                                     ordered=NULL,regular=NULL,minimum=20L) {
  values<-as.numeric(series);frequency<-sift$.positive_count(frequency,"frequency")
  if(!is.null(ordered)||!is.null(regular))
    stop("ordered/regular flags are not evidence; supply a time_index and cadence")
  if(length(values)<minimum||any(!is.finite(values)))
    stop("time series has too few finite observations")
  if(is.null(time_index)) {
    if(!is.ts(series))stop("time_index is required unless series is a ts object")
    time_index<-stats::time(series)
    if(is.null(cadence))cadence<-1/stats::frequency(series)
  }
  if(length(time_index)!=length(values))stop("time_index length must equal series length")
  if(inherits(time_index,"POSIXt")) {
    # POSIXct is double seconds in base R: cadence verification is therefore
    # limited to the timestamp resolution representable at the observed epoch.
    numeric_index<-as.numeric(time_index)
    if(inherits(cadence,"difftime"))cadence<-as.numeric(cadence,units="secs")
  } else if(inherits(time_index,"Date")) {
    numeric_index<-as.numeric(time_index)
    if(inherits(cadence,"difftime"))cadence<-as.numeric(cadence,units="days")
  } else numeric_index<-as.numeric(time_index)
  cadence<-as.numeric(cadence)
  if(length(cadence)!=1L||!is.finite(cadence)||cadence<=0||any(!is.finite(numeric_index)))
    stop("time_index and cadence must be finite and cadence must be positive")
  steps<-diff(numeric_index)
  if(any(steps<=0))stop("time_index must be strictly increasing with no duplicates")
  tolerance<-max(abs(cadence)*1e-9,.Machine$double.eps*16)
  if(any(abs(steps-cadence)>tolerance+1e-9*abs(cadence)))
    stop("time_index spacing is irregular or inconsistent with cadence")
  list(values=values,frequency=frequency,proof=list(
    cadence_min_ratio=min(steps)/cadence,cadence_max_ratio=max(steps)/cadence,
    time_span_steps=(tail(numeric_index,1)-numeric_index[1])/cadence))
}

sift$.unit_root_stationarity <- function(values) {
  if(!requireNamespace("urca",quietly=TRUE))
    stop("R stationarity qualification requires the maintained 'urca' package")
  adf<-urca::ur.df(values,type="drift",selectlags="AIC")
  kpss<-urca::ur.kpss(values,type="mu",lags="short")
  adf_stat<-as.numeric(adf@teststat[1,"tau2"]);adf_critical<-as.numeric(adf@cval["tau2","5pct"])
  # urca exposes KPSS critical values as a one-row matrix.  One-dimensional
  # name indexing returns NA on current urca releases; select the named column
  # explicitly so the consensus check cannot fail with a missing condition.
  kpss_stat<-as.numeric(kpss@teststat);kpss_critical<-as.numeric(kpss@cval[1,"5pct"])
  list(metrics=list(stationarity_statistic=adf_stat,adf_statistic=adf_stat,
    adf_critical_05=adf_critical,kpss_statistic=kpss_stat,kpss_critical_05=kpss_critical),
    status=if(adf_stat<adf_critical&&kpss_stat<kpss_critical)"pass" else "warn")
}

sift$.ljung_box <- function(residuals) {
  lag<-max(1L,min(10L,length(residuals)%/%5L))
  p<-as.numeric(stats::Box.test(residuals,lag=lag,type="Ljung-Box")$p.value)
  list(p=p,status=if(p>.05)"pass" else "warn")
}

sift$.forecast_metrics <- function(actual,forecast,lower,upper) {
  list(rmse=sqrt(mean((actual-forecast)^2)),mae=mean(abs(actual-forecast)),
       prediction_interval_coverage=mean(actual>=lower&actual<=upper),
       prediction_interval_mean_width=mean(upper-lower),nominal_coverage=.95,
       mean_forecast=mean(forecast),mean_actual=mean(actual))
}

sift$from_stationarity_diagnostic <- function(
    series,frequency,time_index=NULL,cadence=NULL,ordered=NULL,regular=NULL,...) {
  input<-sift$.time_series_values(series,frequency,time_index,cadence,ordered,regular)
  stationarity<-sift$.unit_root_stationarity(input$values)
  metrics<-c(stationarity$metrics,input$proof)
  sift$from_method("stationarity_diagnostic",n=length(input$values),
    diagnostics=list(temporal_order="pass",regular_frequency="pass",missingness="pass",
      stationarity_consensus=stationarity$status),metrics=metrics,
    frequency=input$frequency,...)
}

sift$from_seasonal_decomposition <- function(
    series,frequency,time_index=NULL,cadence=NULL,ordered=NULL,regular=NULL,robust=TRUE,...) {
  input<-sift$.time_series_values(series,frequency,time_index,cadence,ordered,regular,
                                  max(20L,2L*as.integer(frequency)))
  if(input$frequency<2L)stop("seasonal decomposition requires frequency >=2")
  fit<-stats::stl(stats::ts(input$values,frequency=input$frequency),
                  s.window="periodic",robust=robust)
  parts<-fit$time.series;resid<-as.numeric(parts[,"remainder"])
  trend<-as.numeric(parts[,"trend"]);seasonal<-as.numeric(parts[,"seasonal"])
  rv<-stats::var(resid);total<-max(stats::var(input$values),1e-12)
  trend_strength<-max(0,1-rv/max(stats::var(trend+resid),1e-12))
  seasonal_strength<-max(0,1-rv/max(stats::var(seasonal+resid),1e-12))
  metrics<-list(trend_strength=min(1,trend_strength),seasonal_strength=min(1,seasonal_strength),
                residual_sd=stats::sd(resid),residual_variance_share=rv/total)
  metrics<-c(metrics,input$proof)
  sift$from_method("seasonal_decomposition",n=length(input$values),
    diagnostics=list(temporal_order="pass",regular_frequency="pass",
      period_support=if(length(input$values)>=3L*input$frequency)"pass" else "warn",
      residual_share=if(metrics$residual_variance_share<=.5)"pass" else "warn"),
    metrics=metrics,frequency=input$frequency,...)
}

sift$.validate_arima_order <- function(order) {
  parsed<-as.integer(order)
  if(length(parsed)!=3L||any(is.na(parsed))||any(parsed<0L|parsed>5L)||any(parsed!=as.numeric(order)))
    stop("ARIMA p,d,q must each be integers between 0 and 5")
  parsed
}

sift$from_arima <- function(
    series,order,holdout,frequency,time_index=NULL,cadence=NULL,ordered=NULL,regular=NULL,...) {
  input<-sift$.time_series_values(series,frequency,time_index,cadence,ordered,regular,40L)
  h<-sift$.positive_count(holdout,"holdout");values<-input$values
  if(h<3L||h>=length(values)%/%2L)
    stop("ARIMA requires ordered regular data and chronological holdout")
  order<-sift$.validate_arima_order(order);train<-head(values,-h);actual<-tail(values,h)
  fit<-stats::arima(train,order=order,method="ML");prediction<-stats::predict(fit,n.ahead=h)
  forecast<-as.numeric(prediction$pred);se<-as.numeric(prediction$se)
  metrics<-sift$.forecast_metrics(actual,forecast,forecast-1.96*se,forecast+1.96*se)
  metrics<-c(metrics,input$proof)
  metrics$aic<-as.numeric(fit$aic);metrics$bic<-as.numeric(-2*fit$loglik+log(length(train))*length(fit$coef))
  lb<-sift$.ljung_box(stats::residuals(fit));metrics$ljung_box_p_value<-lb$p
  ar<-fit$coef[grep("^ar",names(fit$coef))];ma<-fit$coef[grep("^ma",names(fit$coef))]
  ar_roots<-if(length(ar))abs(polyroot(c(1,-ar))) else numeric()
  ma_roots<-if(length(ma))abs(polyroot(c(1,ma))) else numeric()
  sift$from_method("arima",n=length(values),diagnostics=list(
    temporal_order="pass",regular_frequency="pass",
    stationarity=if(order[2]>0L)"pass" else if(length(ar_roots)==0L||min(ar_roots)>1)"pass" else "warn",
    ar_stationarity=if(length(ar_roots)==0L||min(ar_roots)>1)"pass" else "warn",
    residual_autocorrelation=lb$status,
    ma_invertibility=if(length(ma_roots)==0L||min(ma_roots)>1)"pass" else "warn",
    holdout_leakage="pass",prediction_interval_coverage=
      if(metrics$prediction_interval_coverage>=.8)"pass" else "warn"),
    estimates=list(mean_holdout_forecast=metrics$mean_forecast),metrics=metrics,
    evaluation_split="held_out",frequency=input$frequency,
    training_observations=length(train),test_observations=h,
    interval_method="model_based_gaussian",...)
}

sift$from_exponential_smoothing <- function(
    series,holdout,frequency,time_index=NULL,cadence=NULL,ordered=NULL,regular=NULL,
    seasonal="additive",...) {
  input<-sift$.time_series_values(series,frequency,time_index,cadence,ordered,regular,40L)
  h<-sift$.positive_count(holdout,"holdout");values<-input$values
  if(h<3L||h>=length(values)%/%2L)stop("chronological holdout required")
  if(!seasonal%in%c("additive","multiplicative","none"))stop("invalid seasonal method")
  train<-head(values,-h);actual<-tail(values,h)
  fit<-stats::HoltWinters(stats::ts(train,frequency=input$frequency),
      seasonal=if(seasonal=="none")FALSE else seasonal)
  prediction<-stats::predict(fit,n.ahead=h,prediction.interval=TRUE,level=.95)
  forecast<-as.numeric(prediction[,"fit"]);lower<-as.numeric(prediction[,"lwr"]);upper<-as.numeric(prediction[,"upr"])
  metrics<-sift$.forecast_metrics(actual,forecast,lower,upper)
  widths<-upper-lower;metrics<-c(metrics,input$proof)
  metrics$first_interval_width<-widths[1];metrics$last_interval_width<-tail(widths,1)
  residuals<-as.numeric(stats::residuals(fit));metrics$residual_sd<-stats::sd(residuals,na.rm=TRUE)
  lb<-sift$.ljung_box(residuals[is.finite(residuals)]);metrics$ljung_box_p_value<-lb$p
  sift$from_method("exponential_smoothing",n=length(values),diagnostics=list(
    temporal_order="pass",regular_frequency="pass",residual_autocorrelation=lb$status,
    holdout_leakage="pass",prediction_interval_coverage=
      if(metrics$prediction_interval_coverage>=.8)"pass" else "warn"),
    estimates=list(mean_holdout_forecast=metrics$mean_forecast),metrics=metrics,
    evaluation_split="held_out",frequency=input$frequency,
    training_observations=length(train),test_observations=h,
    interval_method="holtwinters_state_space",...)
}

sift$from_interrupted_time_series <- function(
    series,intervention_index,frequency,time_index=NULL,cadence=NULL,ordered=NULL,regular=NULL,
    falsification_status,...) {
  input<-sift$.time_series_values(series,frequency,time_index,cadence,ordered,regular,40L);values<-input$values
  cut<-sift$.positive_count(intervention_index,"intervention_index")
  if(cut<15L||length(values)-cut<10L)stop("ITS requires >=15 pre and >=10 post")
  time<-seq_along(values)-1L;xreg<-cbind(time=time,level_change=as.numeric(time>=cut),
                                         slope_change=pmax(0,time-cut))
  fit<-stats::arima(values,order=c(1,0,0),xreg=xreg,include.mean=TRUE,method="ML")
  vc<-fit$var.coef;keys<-c("level_change","slope_change");estimate<-as.list(fit$coef[keys])
  se<-as.list(sqrt(diag(vc))[keys]);z<-fit$coef[keys]/sqrt(diag(vc))[keys]
  p<-as.list(2*stats::pnorm(-abs(z)));lo<-as.list(fit$coef[keys]-1.96*sqrt(diag(vc))[keys])
  hi<-as.list(fit$coef[keys]+1.96*sqrt(diag(vc))[keys])
  lb<-sift$.ljung_box(stats::residuals(fit));metrics<-list(aic=as.numeric(fit$aic),
    bic=as.numeric(-2*fit$loglik+log(length(values))*length(fit$coef)),ljung_box_p_value=lb$p)
  pre_time<-time[seq_len(cut)];pre_mid<-cut%/%2L
  pre_full<-stats::lm(values[seq_len(cut)]~pre_time+I(pre_time>=pre_mid)+I(pmax(0,pre_time-pre_mid)))
  pre_reduced<-stats::lm(values[seq_len(cut)]~pre_time)
  pretrend_p<-as.numeric(stats::anova(pre_reduced,pre_full)$`Pr(>F)`[2])
  metrics<-c(metrics,list(pretrend_stability_p_value=pretrend_p),input$proof)
  sift$from_method("interrupted_time_series",n=length(values),diagnostics=list(
    temporal_order="pass",regular_frequency="pass",
    pre_intervention_trend=if(pretrend_p>.05)"pass" else "warn",
    intervention_timing="pass",residual_autocorrelation=lb$status,
    design_specific_falsification=sift$.causal_status(falsification_status,"falsification_status")),
    estimates=estimate,standard_errors=se,p_values=p,ci_lower=lo,ci_upper=hi,
    metrics=metrics,uncertainty_type="classical",frequency=input$frequency,
    pre_periods=cut,post_periods=length(values)-cut,...)
}

sift$from_forecast_backtest <- function(
    series,order,initial,frequency,time_index=NULL,cadence=NULL,ordered=NULL,regular=NULL,...) {
  input<-sift$.time_series_values(series,frequency,time_index,cadence,ordered,regular,40L);values<-input$values
  start<-sift$.positive_count(initial,"initial");order<-sift$.validate_arima_order(order)
  if(start<max(30L,2L*input$frequency)||length(values)-start<5L)
    stop("rolling-origin backtest requires ordered regular data and >=5 origins")
  origins<-seq.int(start+1L,length(values));actual<-values[origins]
  forecast<-lower<-upper<-baseline<-numeric(length(origins))
  for(j in seq_along(origins)) {
    origin<-origins[j];history<-values[seq_len(origin-1L)]
    fit<-stats::arima(history,order=order,method="ML");pred<-stats::predict(fit,n.ahead=1)
    forecast[j]<-pred$pred[1];lower[j]<-forecast[j]-1.96*pred$se[1];upper[j]<-forecast[j]+1.96*pred$se[1]
    baseline[j]<-history[length(history)]
  }
  metrics<-sift$.forecast_metrics(actual,forecast,lower,upper)
  metrics<-c(metrics,input$proof)
  metrics$baseline_rmse<-sqrt(mean((actual-baseline)^2));metrics$origins<-as.numeric(length(origins))
  sift$from_method("forecast_backtest",n=length(values),folds=length(origins),diagnostics=list(
    temporal_order="pass",regular_frequency="pass",rolling_origin_backtest="pass",
    holdout_leakage="pass",prediction_interval_coverage=
      if(metrics$prediction_interval_coverage>=.8)"pass" else "warn",
    baseline_comparison=if(metrics$rmse<=metrics$baseline_rmse)"pass" else "warn"),
    metrics=metrics,evaluation_split="rolling_origin",frequency=input$frequency,
    training_observations=start,test_observations=length(origins),
    interval_method="model_based_gaussian",...)
}

# ---------------------------------------------------------------------------
# Domain/design helpers — aggregate outputs only
# ---------------------------------------------------------------------------

sift$from_power_precision <- function(
    effect_sizes,alpha=.05,target_power=.8,allocation_ratio=1,alternative="two.sided",...) {
  effects<-as.numeric(effect_sizes)
  if(length(effects)<1L||length(effects)>12L||any(!is.finite(effects)|effects<.01|effects>5)||
     anyDuplicated(effects)||!identical(effects,sort(effects)))stop("effect sizes must be unique increasing positive scenarios")
  if(!is.finite(alpha)||alpha<=0||alpha>=1||!is.finite(target_power)||target_power<=.5||target_power>=1||
     !is.finite(allocation_ratio)||allocation_ratio<=0||allocation_ratio>10||
     !alternative%in%c("two.sided","one.sided"))stop("invalid prospective power assumptions")
  estimates<-metrics<-list();maximum<-0L
  for(index in seq_along(effects)) {
    name<-paste0("scenario_",index)
    # stats::power.t.test supports equal allocation. For unequal allocation,
    # solve the maintained noncentral-t power function with uniroot.
    if(abs(allocation_ratio-1)<1e-12) {
      first<-ceiling(stats::power.t.test(delta=effects[index],sd=1,sig.level=alpha,
        power=target_power,type="two.sample",alternative=alternative,strict=TRUE)$n)
    } else {
      achieved<-function(first) {
        second<-ceiling(first*allocation_ratio);df<-first+second-2
        ncp<-effects[index]/sqrt(1/first+1/second)
        if (identical(alternative, "two.sided")) {
          critical<-stats::qt(1-alpha/2,df)
          stats::pt(-critical,df,ncp)+stats::pt(critical,df,ncp,lower.tail=FALSE)
        } else {
          critical<-stats::qt(1-alpha,df)
          stats::pt(critical,df,ncp,lower.tail=FALSE)
        }
      }
      first<-ceiling(stats::uniroot(function(x)achieved(x)-target_power,c(2,1e6))$root)
    }
    second<-ceiling(first*allocation_ratio);total<-first+second;maximum<-max(maximum,total)
    estimates[[name]]<-as.numeric(total);metrics[[paste0("effect_size#",name)]]<-effects[index]
    metrics[[paste0("group1_n#",name)]]<-as.numeric(first);metrics[[paste0("group2_n#",name)]]<-as.numeric(second)
  }
  metrics<-c(metrics,list(scenario_count=length(effects),alpha=alpha,target_power=target_power,
                          allocation_ratio=allocation_ratio))
  payload<-c(list(type="method_result",method_id="power_precision",n=maximum,
    diagnostics=list(effect_size_scenarios="pass",alpha_and_power="pass",prospective_design="pass"),
    estimates=estimates,metrics=metrics,
    test_alternative=if(alternative=="two.sided")"two_sided" else "larger"),list(...))
  payload[["_via_helper"]]<-"power_precision_v1"
  sift$.write_result(payload)
}

sift$from_simulation_design <- function(
    effect_size,group_n,replications=2000L,alpha=.05,seed=1729L,...) {
  effect<-as.numeric(effect_size);size<-sift$.positive_count(group_n,"group_n")
  reps<-sift$.positive_count(replications,"replications");seed<-sift$.positive_count(seed,"seed")
  if(!is.finite(effect)||effect<.01||effect>5||size<5L||reps<1000L||reps>20000L||reps*size>5000000L||
     !is.finite(alpha)||alpha<=0||alpha>=1)stop("invalid simulation design assumptions")
  set.seed(seed);control<-matrix(stats::rnorm(reps*size),nrow=reps)
  treated<-matrix(stats::rnorm(reps*size,mean=effect),nrow=reps)
  mean_difference<-rowMeans(treated)-rowMeans(control)
  pooled<-((size-1)*apply(treated,1,stats::var)+(size-1)*apply(control,1,stats::var))/(2*size-2)
  statistic<-mean_difference/sqrt(pooled*(2/size));p<-2*stats::pt(-abs(statistic),df=2*size-2)
  rejected<-sum(p<alpha);power<-rejected/reps;mcse<-sqrt(power*(1-power)/reps)
  mc_interval<-stats::binom.test(rejected,reps,conf.level=.95)$conf.int
  analytic<-stats::power.t.test(n=size,delta=effect,sd=1,sig.level=alpha,type="two.sample",
                                alternative="two.sided",strict=TRUE)$power
  payload<-c(list(type="method_result",method_id="simulation_design",n=2L*size,
    diagnostics=list(seed_recorded="pass",
    replication_count=as.numeric(reps),monte_carlo_standard_error=mcse,scenario_sensitivity="warn"),
    estimates=list(empirical_power=power),ci_lower=list(empirical_power=as.numeric(mc_interval[1])),
    ci_upper=list(empirical_power=as.numeric(mc_interval[2])),metrics=list(effect_size=effect,
      group_n=as.numeric(size),alpha=alpha,replications=as.numeric(reps),rejection_count=as.numeric(rejected),
      monte_carlo_standard_error=mcse,analytic_power=as.numeric(analytic),
      absolute_analytic_difference=abs(power-analytic)),seed=seed,replicates=reps,
      interval_method="clopper_pearson_binomial"),list(...))
  payload[["_via_helper"]]<-"simulation_design_v1"
  sift$.write_result(payload)
}


# ---------------------------------------------------------------------------
# Convenience helpers that pull structured payloads out of common R objects
# ---------------------------------------------------------------------------

#' From an `lm` fit, emit a linear_regression payload.
#'
#' The helper covers the fields the Sift linear_regression schema
#' accepts. Any extra kwargs passed via `...` are included too (dropped
#' by the sanitizer if not whitelisted).
#'
#' Also prints R's native `summary(model)` table to stdout so the
#' researcher sees the familiar regression output in the TUI's raw
#' log panel. Stdout never reaches Claude (executor strips it before
#' anything returns to the sanitizer), so printing here is only for
#' the researcher's benefit.
sift$from_lm <- function(model, ...) {
  s <- summary(model)
  print(s)

  # Per-class dispatch — the previous version assumed an lm/glm shape
  # ("Estimate" / "Std. Error" columns) and aborted with "undefined
  # columns selected" on Cox (coxph) and fixest fits, both of which
  # the comment block above claims to support. Now we detect class
  # explicitly and route extraction.
  is_cox    <- inherits(model, "coxph")
  is_fixest <- inherits(model, "fixest")
  is_mixed  <- inherits(model, "merMod")  # lmerMod, glmerMod, nlmerMod
  is_glm    <- inherits(model, "glm") && !is_mixed
  # ``glmerMod`` inherits from both glm and merMod; treat as mixed.
  is_lm     <- inherits(model, "lm") && !is_glm && !is_cox && !is_fixest && !is_mixed

  # Coefficient table location: fixest puts it in $coeftable, not
  # $coefficients (the $coefficients slot on a fixest summary is the
  # point-estimate vector).
  ce <- if (is_fixest) {
    as.data.frame(s$coeftable)
  } else {
    as.data.frame(s$coefficients)
  }
  ce_cols <- colnames(ce)

  # Estimate column: "Estimate" (lm/glm/fixest), "coef" (coxph), or
  # positional fallback to col 1.
  est_col <- if ("Estimate" %in% ce_cols) "Estimate" else
             if ("coef"     %in% ce_cols) "coef"     else
             ce_cols[1]
  # SE column: "Std. Error" (lm/glm/fixest), "se(coef)" (coxph), or
  # positional fallback to col 2 (Cox has exp(coef) in col 2 — that's
  # wrong, but we never reach this branch because "se(coef)" is the
  # only hit on coxph).
  se_col  <- if ("Std. Error" %in% ce_cols) "Std. Error" else
             if ("se(coef)"   %in% ce_cols) "se(coef)"   else
             ce_cols[2]
  # Test-stat column: t value (lm/fixest), z value (glm), z (coxph).
  stat_col <- if ("t value" %in% ce_cols) "t value" else
              if ("z value" %in% ce_cols) "z value" else
              if ("z"       %in% ce_cols) "z"       else
              if (ncol(ce) >= 3) ce_cols[3] else NA_character_
  # p-value column: prefer named matches, then fall back to the LAST
  # column when there are at least 4 columns (Cox has 5 with
  # Pr(>|z|) at position 5; lm/glm have 4 with Pr at position 4).
  # ``lmer`` without ``lmerTest`` emits only 3 columns (no p-value
  # at all) — the unconditional ``ce_cols[ncol(ce)]`` fallback used
  # to land on "t value" and mis-stamp it as p_values. The
  # ncol >= 4 guard refuses the fallback for the 3-column shape so
  # the helper omits p_values rather than misreports them.
  p_col <- if ("Pr(>|t|)" %in% ce_cols) "Pr(>|t|)" else
           if ("Pr(>|z|)" %in% ce_cols) "Pr(>|z|)" else
           if (ncol(ce) >= 4) ce_cols[ncol(ce)] else NA_character_

  coefs <- as.list(ce[, est_col]); names(coefs) <- rownames(ce)
  ses   <- as.list(ce[, se_col]);  names(ses)   <- rownames(ce)
  tvals <- if (!is.na(stat_col)) {
    v <- as.list(ce[, stat_col]); names(v) <- rownames(ce); v
  } else NULL
  pvals <- if (!is.na(p_col)) {
    v <- as.list(ce[, p_col]); names(v) <- rownames(ce); v
  } else NULL

  # Response / predictors. For Cox the LHS is Surv(time, event) — a
  # call, not a symbol. ``all.vars()`` returns the variable names in
  # order, so all.vars(Surv(t_obs, cens))[1] = "t_obs" is the time
  # variable, which is the right thing to report as the response.
  lhs <- attr(terms(model), "variables")[[2]]
  response <- all.vars(lhs)[1]
  if (is_mixed) {
    # ``term.labels`` for a merMod includes the random-effect
    # grouping factors (e.g. "school" in ``y ~ x + (1 | school)``)
    # alongside the fixed-effect predictors. fixef() returns just
    # the fixed-effect coefficients; their names are the predictor
    # surface the model thinks of as the "regressors of interest".
    fe_names <- tryCatch(names(lme4::fixef(model)),
                         error = function(e) character(0))
    predictors <- as.list(fe_names[!fe_names %in%
                                   c("(Intercept)", "intercept", "const")])
  } else {
    predictors <- as.list(as.character(attr(terms(model), "term.labels")))
  }

  # Sample size: nobs(coxph) returns the number of events, not records.
  # m$n is records; m$nevent is failures. Use m$n for n on Cox so the
  # SDC min-N gate sees the sample size, not the event count.
  n_val <- if (is_cox) as.integer(model$n) else as.integer(nobs(model))

  # Aggregate diagnostics — collinearity / numerical stability. Pure
  # aggregates over the design matrix, no per-row leak. Failures are
  # silent — the field is omitted rather than blowing up the emit.
  vif_list    <- tryCatch(sift$.compute_vif(model),               error = function(e) NULL)
  cond_num    <- tryCatch(sift$.compute_condition_number(model),  error = function(e) NULL)
  vcov_nested <- tryCatch(sift$.compute_vcov(model),              error = function(e) NULL)

  args <- list(
    # ``coefficient_table_with_fit_stats`` is the canonical bucket
    # name (covers OLS / glm / coxph / fixest — anything that emits
    # a coefficient table). ``linear_regression`` is kept as a
    # legacy alias in the sanitizer's dispatch table for back-compat
    # with stored payloads; new emissions use the descriptive name
    # so the model sees a type that matches what's in the bucket.
    type = "coefficient_table_with_fit_stats",
    n = n_val,
    response_variable = response,
    predictor_variables = predictors,
    coefficients = coefs,
    standard_errors = ses,
    t_statistics = tvals,
    p_values = pvals
  )

  # Estimator-appropriate fit metrics. Each branch only emits fields
  # that are meaningful for its class — emitting r_squared on a glm
  # fit (where summary()$r.squared is NULL) used to ride along as a
  # null and force the sanitizer to drop it with a transformation
  # note on every GLM payload.
  if (is_lm) {
    args$r_squared          <- s$r.squared
    args$adj_r_squared      <- s$adj.r.squared
    args$residual_std_error <- s$sigma
    args$degrees_of_freedom <- as.integer(s$df[2])
    if (!is.null(s$fstatistic)) {
      args$f_statistic <- unname(s$fstatistic["value"])
      args$f_p_value   <- unname(pf(s$fstatistic["value"],
                                    s$fstatistic["numdf"],
                                    s$fstatistic["dendf"],
                                    lower.tail = FALSE))
    }
  }
  if (is_glm) {
    # McFadden-equivalent for GLM: 1 - residual_deviance/null_deviance.
    # Deviance ratio matches McFadden when the link is canonical; for
    # non-canonical links it's the conventional GLM pseudo-R² reported
    # by Stata's ``glm`` and statsmodels' ``.prsquared``.
    if (!is.null(model$null.deviance) && !is.null(model$deviance) &&
        is.finite(model$null.deviance) && is.finite(model$deviance) &&
        model$null.deviance > 0) {
      args$pseudo_r_squared <- 1 - model$deviance / model$null.deviance
      chi2 <- model$null.deviance - model$deviance
      df_chi <- tryCatch(
        as.integer(model$df.null - model$df.residual),
        error = function(e) NA_integer_
      )
      if (!is.na(df_chi) && df_chi > 0 && chi2 >= 0) {
        args$chi_squared <- as.numeric(chi2)
        args$chi_squared_p_value <- as.numeric(
          pchisq(chi2, df = df_chi, lower.tail = FALSE)
        )
      }
    }
    if (!is.null(model$df.residual)) {
      args$degrees_of_freedom <- as.integer(model$df.residual)
    }
    ll <- tryCatch(as.numeric(logLik(model)), error = function(e) NULL)
    if (!is.null(ll) && is.finite(ll)) args$log_likelihood <- ll
    aic_v <- tryCatch(as.numeric(AIC(model)), error = function(e) NULL)
    if (!is.null(aic_v) && is.finite(aic_v)) args$aic <- aic_v
    bic_v <- tryCatch(as.numeric(BIC(model)), error = function(e) NULL)
    if (!is.null(bic_v) && is.finite(bic_v)) args$bic <- bic_v
  }
  if (is_cox) {
    # Cox PH: subject + failure counts, Harrell's C, LR test, log-lik.
    if (!is.null(model$n))      args$n_subjects <- as.integer(model$n)
    if (!is.null(model$nevent)) args$n_failures <- as.integer(model$nevent)
    cs <- tryCatch(s$concordance, error = function(e) NULL)
    if (!is.null(cs) && length(cs) >= 1 && is.finite(cs[1])) {
      args$concordance <- as.numeric(cs[1])
    }
    lr <- tryCatch(s$logtest, error = function(e) NULL)
    if (!is.null(lr) && length(lr) >= 3 &&
        is.finite(lr["test"]) && is.finite(lr["pvalue"])) {
      args$chi_squared <- as.numeric(lr["test"])
      args$chi_squared_p_value <- as.numeric(lr["pvalue"])
    }
    ll <- tryCatch(as.numeric(logLik(model)), error = function(e) NULL)
    if (!is.null(ll) && is.finite(ll)) args$log_likelihood <- ll
    aic_v <- tryCatch(as.numeric(AIC(model)), error = function(e) NULL)
    if (!is.null(aic_v) && is.finite(aic_v)) args$aic <- aic_v
    bic_v <- tryCatch(as.numeric(BIC(model)), error = function(e) NULL)
    if (!is.null(bic_v) && is.finite(bic_v)) args$bic <- bic_v
  }
  if (is_mixed) {
    # lme4::lmer / glmer / nlmer. Fixed-effects coefficient table is
    # already extracted via the standard ``summary(m)$coefficients``
    # path (Estimate / Std. Error / t-or-z / optional p). What's
    # specific to mixed models: variance components, per-level
    # group counts, REML vs ML fit method, ICC for one-level fits.
    #
    # Variance components live in ``VarCorr(model)`` — a list of
    # per-group covariance matrices plus an ``sc`` attribute for
    # residual SD. Each matrix's diagonal entries are variances
    # (random-intercept variance, random-slope variance). We emit
    # *variances* — sqrt'd values would duplicate signal and the
    # intercept-slope covariance stays inside ``vcov`` for callers
    # that genuinely need it.
    vc <- tryCatch(lme4::VarCorr(model), error = function(e) NULL)
    re_var <- list()
    if (!is.null(vc)) {
      for (grp_name in names(vc)) {
        mat <- vc[[grp_name]]
        if (!is.matrix(mat)) next
        rn <- rownames(mat)
        if (is.null(rn)) next
        for (i in seq_along(rn)) {
          v <- as.numeric(mat[i, i])
          if (!is.finite(v) || v < 0) next
          key <- if (rn[i] %in% c("(Intercept)", "intercept"))
                    grp_name else paste0(grp_name, ".", rn[i])
          re_var[[key]] <- v
        }
      }
      sc <- attr(vc, "sc")
      if (!is.null(sc) && length(sc) == 1 && is.finite(sc) && sc >= 0) {
        re_var[["residual"]] <- as.numeric(sc^2)
      }
    }
    if (length(re_var) > 0) args$random_effects_variance <- re_var

    # Per-level group counts. lme4::ngrps() returns a named integer
    # vector keyed by grouping-factor name. Same disclosure profile
    # as ``fixed_effects`` and ``n_clusters`` — column name +
    # cardinality, no level identities.
    ng <- tryCatch(lme4::ngrps(model), error = function(e) NULL)
    if (!is.null(ng) && length(ng) > 0) {
      ng_dict <- list()
      for (i in seq_along(ng)) {
        ng_dict[[names(ng)[i]]] <- as.integer(ng[i])
      }
      args$n_groups_per_level <- ng_dict
    }

    # Fit method. ``isREML(m)`` for lmer, FALSE for glmer (always ML).
    # ``getME(model, "is_REML")`` works across the merMod hierarchy.
    is_reml <- tryCatch(lme4::isREML(model), error = function(e) NULL)
    if (!is.null(is_reml)) {
      args$fit_method <- if (isTRUE(is_reml)) "REML" else "ML"
    }

    # Intraclass correlation — only well-defined for one-grouping,
    # intercept-only random-effect specifications. Compute as
    # sigma_u² / (sigma_u² + sigma_e²) when there's exactly one
    # group + a residual term in ``random_effects_variance``.
    if (length(re_var) == 2 && "residual" %in% names(re_var)) {
      grp_var_name <- setdiff(names(re_var), "residual")
      if (length(grp_var_name) == 1) {
        s_u2 <- re_var[[grp_var_name]]
        s_e2 <- re_var[["residual"]]
        if (is.finite(s_u2) && is.finite(s_e2) && (s_u2 + s_e2) > 0) {
          args$icc <- as.numeric(s_u2 / (s_u2 + s_e2))
        }
      }
    }

    ll <- tryCatch(as.numeric(logLik(model)), error = function(e) NULL)
    if (!is.null(ll) && is.finite(ll)) args$log_likelihood <- ll
    aic_v <- tryCatch(as.numeric(AIC(model)), error = function(e) NULL)
    if (!is.null(aic_v) && is.finite(aic_v)) args$aic <- aic_v
    bic_v <- tryCatch(as.numeric(BIC(model)), error = function(e) NULL)
    if (!is.null(bic_v) && is.finite(bic_v)) args$bic <- bic_v
    n_obs <- tryCatch(as.integer(nobs(model)), error = function(e) NULL)
    if (!is.null(n_obs)) args$n <- n_obs
    df_resid <- tryCatch(as.integer(df.residual(model)), error = function(e) NULL)
    if (!is.null(df_resid) && !is.na(df_resid)) args$degrees_of_freedom <- df_resid
  }
  if (is_fixest) {
    # fixest::feols (and family). Coefficients and SE columns are
    # already extracted above; here we add fit metrics, absorbed-FE
    # cardinality, and cluster-robust metadata. Critically: emit
    # FE-dimension SIZES, never the level identifiers themselves —
    # listing the 1,247 firms is not OK, reporting "firm FE absorbed,
    # 1,247 levels" is. Same rule for cluster cardinalities.
    r2 <- tryCatch(
      fixest::fitstat(model, type = "r2", verbose = FALSE)$r2,
      error = function(e) NULL
    )
    if (!is.null(r2) && is.finite(r2)) args$r_squared <- as.numeric(r2)
    ar2 <- tryCatch(
      fixest::fitstat(model, type = "ar2", verbose = FALSE)$ar2,
      error = function(e) NULL
    )
    if (!is.null(ar2) && is.finite(ar2)) args$adj_r_squared <- as.numeric(ar2)
    ll <- tryCatch(as.numeric(logLik(model)), error = function(e) NULL)
    if (!is.null(ll) && is.finite(ll)) args$log_likelihood <- ll
    aic_v <- tryCatch(as.numeric(AIC(model)), error = function(e) NULL)
    if (!is.null(aic_v) && is.finite(aic_v)) args$aic <- aic_v
    bic_v <- tryCatch(as.numeric(BIC(model)), error = function(e) NULL)
    if (!is.null(bic_v) && is.finite(bic_v)) args$bic <- bic_v
    fv <- model$fixef_vars
    fs <- model$fixef_sizes
    if (!is.null(fv) && !is.null(fs) && length(fv) == length(fs) && length(fv) > 0) {
      fe_summary <- list()
      for (i in seq_along(fv)) {
        fe_summary[[as.character(fv[i])]] <- as.integer(fs[i])
      }
      args$fixed_effects <- fe_summary
    }
    # Cluster-robust SE metadata. fixest stores the cluster formula
    # in m$call$cluster (e.g. ``~g`` or ``~g + h`` for two-way).
    # ``attr(summary(m)$cov.scaled, "G")`` carries per-dimension
    # cluster counts as an integer vector when single-dim; for
    # multi-way it collapses to a single value (the minimum count)
    # so we can't always recover per-dim counts cleanly. Emit the
    # NAME list always; emit ``n_clusters`` only when the count
    # vector matches the name vector in length (single-dim case).
    cl_call <- tryCatch(model$call$cluster, error = function(e) NULL)
    if (!is.null(cl_call)) {
      cl_names <- tryCatch(all.vars(cl_call), error = function(e) character(0))
      if (length(cl_names) > 0) {
        args$cluster_variables <- as.list(cl_names)
        args$robust_se_type <- "cluster"
        Gvec <- tryCatch(
          attr(s$cov.scaled, "G"),
          error = function(e) NULL
        )
        if (!is.null(Gvec) && length(Gvec) == length(cl_names)) {
          nc <- list()
          for (i in seq_along(cl_names)) {
            nc[[cl_names[i]]] <- as.integer(Gvec[i])
          }
          args$n_clusters <- nc
        }
      }
    }
    # Non-cluster fixest variance flavours. ``feols(..., vcov="hetero")``
    # selects White HC; ``vcov=NW(lag)`` / ``vcov="NW"`` selects
    # Newey-West HAC. We probe the call (cleaner than reaching into
    # the summary object across fixest versions) and map onto the
    # sanitizer's canonical enum. Cluster already won above; skip the
    # remap there.
    if (is.null(args$robust_se_type)) {
      vcov_arg <- tryCatch(model$call$vcov, error = function(e) NULL)
      vcov_label <- ""
      if (!is.null(vcov_arg)) {
        if (is.character(vcov_arg)) {
          vcov_label <- tolower(as.character(vcov_arg))
        } else if (is.call(vcov_arg)) {
          # NW(...) / conley(...) / etc. — head of the call gives
          # the helper name.
          vcov_label <- tolower(as.character(vcov_arg[[1]]))
        }
      }
      rse <- NULL
      if (nzchar(vcov_label)) {
        if (vcov_label %in% c("hetero", "white", "hc1", "hc")) {
          rse <- "hc1"
        } else if (vcov_label == "hc0") {
          rse <- "hc0"
        } else if (vcov_label == "hc2") {
          rse <- "hc2"
        } else if (vcov_label == "hc3") {
          rse <- "hc3"
        } else if (vcov_label %in% c("nw", "newey_west", "newey-west")) {
          rse <- "hac_newey_west"
        } else if (vcov_label == "bootstrap") {
          rse <- "bootstrap"
        }
      }
      if (!is.null(rse)) {
        args$robust_se_type <- rse
      }
    }
  }

  if (!is.null(vif_list) && length(vif_list) > 0) args$vif <- vif_list
  if (!is.null(cond_num)) args$condition_number <- cond_num
  if (!is.null(vcov_nested) && length(vcov_nested) > 0) args$vcov <- vcov_nested

  # Panel-data post-estimation diagnostics. ``plm`` fits expose
  # the relevant tests as functions taking the fitted model; this
  # block runs each one inside ``tryCatch`` so a single failure
  # (typically: test not defined for this fit's effect= specification)
  # doesn't abort the helper. Each test contributes two scalars
  # (chi² + p) and the sanitizer's allowlist accepts both. Researchers
  # using ``plm`` get the diagnostics without having to compute them
  # script-side and pass through ``...``; researchers using ``fixest``
  # or core ``lm`` continue to pass through ``...`` because those
  # packages don't ship panel diagnostics with the same vocabulary.
  if (inherits(model, "plm")) {
    # F-test for fixed effects (pooled OLS vs FE). ``plm::pFtest``
    # takes ``(fe_fit, pooled_fit)``; if the caller only passes the
    # FE fit, fall back to plm::pFtest with model alone where
    # supported. ``plm::pFtest`` is the canonical interface.
    fe_F <- tryCatch(plm::pFtest(model, NULL), error = function(e) NULL)
    if (!is.null(fe_F) && !is.null(fe_F$statistic) && is.finite(fe_F$statistic)) {
      args$f_test_fe_chi2 <- as.numeric(fe_F$statistic)
      if (!is.null(fe_F$p.value) && is.finite(fe_F$p.value)) {
        args$f_test_fe_p <- as.numeric(fe_F$p.value)
      }
    }
    # Breusch-Pagan LM test for random effects (pooled OLS vs RE).
    bp <- tryCatch(plm::plmtest(model, type = "bp"), error = function(e) NULL)
    if (!is.null(bp) && !is.null(bp$statistic) && is.finite(bp$statistic)) {
      args$breusch_pagan_chi2 <- as.numeric(bp$statistic)
      if (!is.null(bp$p.value) && is.finite(bp$p.value)) {
        args$breusch_pagan_p <- as.numeric(bp$p.value)
      }
    }
    # Wooldridge AR(1) test for serial correlation in idiosyncratic
    # errors. ``plm::pwartest`` is the standard implementation;
    # ``plm::pbgtest`` is an alternative (Breusch-Godfrey adapted
    # for panel) that requires more model assumptions.
    wt <- tryCatch(plm::pwartest(model), error = function(e) NULL)
    if (!is.null(wt) && !is.null(wt$statistic) && is.finite(wt$statistic)) {
      args$wooldridge_ar1_chi2 <- as.numeric(wt$statistic)
      if (!is.null(wt$p.value) && is.finite(wt$p.value)) {
        args$wooldridge_ar1_p <- as.numeric(wt$p.value)
      }
    }
    # Hausman test (FE vs RE) needs BOTH fits; can't auto-run from
    # one fit. Caller passes the chi² and p as kwargs (``hausman_chi2``,
    # ``hausman_p``) after running ``phtest(fe, re)`` themselves.
  }

  # Bind the structural regression bucket to a registry method only when
  # the fitted R class/family/link proves one exact identity. fixest and
  # plm objects can represent OLS, IV, panel, and DiD specifications, so
  # they deliberately remain unstamped here rather than being broadly
  # accepted as linear regression.
  registry_method_id <- NULL
  if (is_cox) {
    registry_method_id <- "cox_proportional_hazards"
  } else if (is_mixed) {
    registry_method_id <- if (inherits(model, "glmerMod"))
      "generalized_mixed_effects" else if (inherits(model, "lmerMod"))
      "linear_mixed_effects" else NULL
  } else if (is_glm) {
    fam <- tryCatch(tolower(model$family$family), error = function(e) "")
    link <- tryCatch(tolower(model$family$link), error = function(e) "")
    if (fam == "binomial" && link == "logit") {
      registry_method_id <- "logistic_regression"
    } else if (fam == "binomial" && link == "probit") {
      registry_method_id <- "probit_regression"
    } else if (fam == "poisson") {
      registry_method_id <- "poisson_regression"
    } else if (inherits(model, "negbin") ||
               startsWith(fam, "negative binomial")) {
      registry_method_id <- "negative_binomial_regression"
    } else if (fam == "gaussian" && link == "identity") {
      registry_method_id <- "linear_regression"
    }
  } else if (is_lm && !inherits(model, "plm")) {
    registry_method_id <- "linear_regression"
  }
  extras <- list(...)
  extras[["_registry_method_id"]] <- NULL
  extras[["_via_helper"]] <- NULL
  payload <- c(args, extras)
  if (!is.null(registry_method_id)) {
    payload[["_registry_method_id"]] <- registry_method_id
  }
  sift$.write_result(payload)
}


# vcov(model): full variance-covariance matrix of the coefficient
# estimates. Diagonals equal SE^2; off-diagonals enable Wald tests
# / joint significance / linear-combination CIs. Pure aggregate
# from sigma^2 * (X'X)^-1 with no per-row leak. Returns a nested
# named list keyed by coefficient name; ``NULL`` on any error.
sift$.compute_vcov <- function(model) {
  v <- tryCatch(stats::vcov(model), error = function(e) NULL)
  if (is.null(v) || !is.matrix(v)) return(NULL)
  rn <- rownames(v); cn <- colnames(v)
  if (is.null(rn) || is.null(cn)) return(NULL)
  out <- list()
  for (i in seq_along(rn)) {
    inner <- list()
    for (j in seq_along(cn)) {
      val <- v[i, j]
      if (is.finite(val)) inner[[cn[j]]] <- as.numeric(val)
    }
    if (length(inner) > 0) out[[rn[i]]] <- inner
  }
  if (length(out) == 0) NULL else out
}


# VIF per predictor: regress each predictor on the others, return
# 1 / (1 - R^2_aux). The intercept is excluded; perfectly collinear
# predictors are omitted (R^2_aux >= 1) so the caller treats their
# absence as "VIF undefined" rather than emitting Inf.
sift$.compute_vif <- function(model) {
  X <- tryCatch(model.matrix(model), error = function(e) NULL)
  if (is.null(X) || ncol(X) < 2 || nrow(X) < 2) return(NULL)
  cols <- colnames(X)
  intercept_alias <- c("(Intercept)", "intercept", "const")
  drop_intercept <- cols %in% intercept_alias
  out <- list()
  for (i in seq_along(cols)) {
    if (drop_intercept[i]) next
    name <- cols[i]
    xi <- X[, i]
    X_others <- X[, -i, drop = FALSE]
    if (ncol(X_others) == 0) next
    fit_aux <- tryCatch(
      stats::lm.fit(X_others, xi),
      error = function(e) NULL
    )
    if (is.null(fit_aux)) next
    ss_tot <- sum((xi - mean(xi))^2)
    ss_res <- sum(fit_aux$residuals^2)
    if (ss_tot <= 0 || ss_res < 0) next
    r2_aux <- 1 - ss_res / ss_tot
    if (r2_aux >= 1 || r2_aux < 0) next
    out[[name]] <- 1 / (1 - r2_aux)
  }
  if (length(out) == 0) NULL else out
}


# kappa(X): condition number of the design matrix. Higher values flag
# numerical instability that VIF (single-column at a time) can miss
# when collinearity is spread across many predictors.
sift$.compute_condition_number <- function(model) {
  X <- tryCatch(model.matrix(model), error = function(e) NULL)
  if (is.null(X)) return(NULL)
  k <- tryCatch(kappa(X, exact = TRUE), error = function(e) NULL)
  if (is.null(k) || !is.finite(k)) NULL else as.numeric(k)
}


#' From a clustering fit (``stats::kmeans`` or ``stats::hclust``),
#' emit a cluster_analysis payload.
#'
#' Dispatches on class:
#'   * ``kmeans`` — read cluster sizes, centroids, within-SS from
#'     the fit directly; no extra data needed.
#'   * ``hclust`` — hierarchical fits don't store the data or a
#'     cluster assignment (just the dendrogram). The caller must
#'     pass ``data`` (the matrix the dendrogram was built on) and
#'     ``k`` (the cut point), and the helper computes assignments
#'     via ``cutree(fit, k=k)`` plus centroids and within-SS from
#'     the data.
#'
#' DBSCAN / HDBSCAN intentionally aren't supported by this helper
#' — their inference-adequacy story (density parameters, noise-
#' point handling, no centroids by construction) is a separate
#' design pass. The helper raises with a clear pointer to the
#' generic ``sift$result(type="cluster_analysis", method="dbscan",
#' ...)`` path until a dedicated helper ships.
#'
#' Per-observation cluster assignments are NOT emitted on any path
#' — they're per-row data and have no slot on the allowlist. The
#' sanitizer's whole-cluster suppression gate fires on sizes below
#' ``min_n_descriptive`` and per-cluster precision clamping fires
#' on surviving centroids.
#'
#' Examples:
#'   # k-means
#'   m <- kmeans(df[, c("age","income","tenure")], centers = 4, nstart = 10)
#'   sift$from_cluster(m, variables = c("age","income","tenure"),
#'                     label = "customer segmentation")
#'
#'   # Hierarchical (Ward) — data + k required
#'   d <- dist(df[, c("age","income","tenure")])
#'   h <- hclust(d, method = "ward.D2")
#'   sift$from_cluster(h, data = df[, c("age","income","tenure")],
#'                     k = 4,
#'                     variables = c("age","income","tenure"),
#'                     linkage = "ward",
#'                     label = "ward clustering")
sift$from_cluster <- function(fit, variables = NULL, data = NULL,
                              k = NULL, linkage = NULL, label = NULL) {
  if (inherits(fit, "kmeans")) {
    return(sift$.from_kmeans_impl(fit, variables = variables, label = label))
  }
  if (inherits(fit, "hclust")) {
    if (is.null(data) || is.null(k)) {
      stop(
        "sift$from_cluster: hierarchical fits need ``data`` (the matrix ",
        "the dendrogram was built on) and ``k`` (the cut point) — hclust ",
        "doesn't store either."
      )
    }
    return(sift$.from_hclust_impl(fit, data = data, k = k,
                                  variables = variables,
                                  linkage = linkage, label = label))
  }
  if (inherits(fit, "dbscan") || inherits(fit, "hdbscan")) {
    stop(
      "sift$from_cluster: dedicated DBSCAN / HDBSCAN helper not yet ",
      "shipped. Construct the payload via ",
      "``sift$result(type=\"cluster_analysis\", method=\"dbscan\", ",
      "cluster_sizes=..., n_noise_points=..., variables=..., ...)`` ",
      "from the script — the cluster_analysis shape accepts dbscan ",
      "with centroids absent."
    )
  }
  stop(
    "sift$from_cluster: unknown clustering class ", class(fit)[1],
    ". Supported: kmeans, hclust. DBSCAN-family: use generic ",
    "``sift$result(type=\"cluster_analysis\", ...)`` until a ",
    "dedicated helper ships."
  )
}


# kmeans extraction — internal implementation of from_cluster's
# kmeans branch.
sift$.from_kmeans_impl <- function(fit, variables = NULL, label = NULL) {
  print(fit)
  centers <- fit$centers
  if (is.null(centers) || !is.matrix(centers)) {
    stop("sift$from_kmeans: fit$centers is missing or not a matrix")
  }
  n_clusters <- nrow(centers)
  n_features <- ncol(centers)

  # Variable names: prefer centers's column names; else accept the
  # caller's list; else fall back to feature_i.
  if (is.null(variables)) {
    vnames <- colnames(centers)
    if (is.null(vnames) || any(!nzchar(vnames))) {
      variables <- paste0("feature_", seq_len(n_features))
    } else {
      variables <- vnames
    }
  } else {
    variables <- as.character(variables)
    if (length(variables) != n_features) {
      stop(sprintf(
        "sift$from_kmeans: variables has %d entries but kmeans was fit on %d features",
        length(variables), n_features
      ))
    }
  }

  cluster_labels <- paste0("cluster_", seq_len(n_clusters))
  cluster_sizes <- list()
  centroids <- list()
  within_cluster_ss <- list()
  for (i in seq_len(n_clusters)) {
    cl <- cluster_labels[i]
    cluster_sizes[[cl]] <- as.integer(fit$size[i])
    centroid_row <- list()
    for (j in seq_len(n_features)) {
      centroid_row[[variables[j]]] <- as.numeric(centers[i, j])
    }
    centroids[[cl]] <- centroid_row
    if (!is.null(fit$withinss) && length(fit$withinss) >= i) {
      within_cluster_ss[[cl]] <- as.numeric(fit$withinss[i])
    }
  }

  # Total N is the sum of cluster sizes.
  n_obs <- sum(fit$size)
  totss <- fit$totss
  twss <- fit$tot.withinss
  bss <- fit$betweenss

  args <- list(
    type = "cluster_analysis",
    method = "kmeans",
    distance_metric = "euclidean",
    n_observations = as.integer(n_obs),
    n_clusters = as.integer(n_clusters),
    n_features = as.integer(n_features),
    variables = as.list(variables),
    cluster_labels = as.list(cluster_labels),
    cluster_sizes = cluster_sizes,
    centroids = centroids
  )
  if (length(within_cluster_ss) > 0) args$within_cluster_ss <- within_cluster_ss
  if (!is.null(twss) && is.finite(twss)) args$total_within_ss <- as.numeric(twss)
  if (!is.null(twss) && is.finite(twss)) args$inertia <- as.numeric(twss)
  if (!is.null(bss) && is.finite(bss)) args$between_cluster_ss <- as.numeric(bss)
  if (!is.null(totss) && is.finite(totss)) args$total_ss <- as.numeric(totss)
  if (!is.null(totss) && is.finite(totss) && totss > 0) {
    args$ss_ratio <- as.numeric(bss / totss)
  }
  if (!is.null(fit$iter) && is.finite(fit$iter)) {
    args$n_iterations <- as.integer(fit$iter)
  }
  if (!is.null(label)) args$label <- as.character(label)
  do.call(sift$result, args)
}


# Hierarchical extraction — internal implementation of from_cluster's
# hclust branch. hclust stores only the dendrogram (merge matrix +
# heights), not the data or any cluster assignment. The caller
# passes ``data`` (the matrix the dendrogram was built on) and
# ``k`` (the cut point); the helper computes cluster assignments
# via ``cutree(fit, k = k)`` and centroids + within-SS from the
# data + assignments.
#
# Privacy carve-out: the linkage matrix (``fit$merge``) and merge
# heights (``fit$height``) are NOT emitted — they're per-merge
# records over the data, structurally absent from the
# cluster_analysis allowlist. The dendrogram lives on the
# researcher's local R session.
sift$.from_hclust_impl <- function(fit, data, k, variables = NULL,
                                   linkage = NULL, label = NULL) {
  if (!is.numeric(k) || length(k) != 1 || k < 2 || k != as.integer(k)) {
    stop("sift$from_cluster: ``k`` must be a positive integer >= 2")
  }
  data <- as.matrix(data)
  if (nrow(data) < 2 || ncol(data) < 1) {
    stop("sift$from_cluster: ``data`` must be a non-degenerate matrix")
  }
  k <- as.integer(k)
  if (is.null(variables)) {
    vnames <- colnames(data)
    if (is.null(vnames) || any(!nzchar(vnames))) {
      variables <- paste0("feature_", seq_len(ncol(data)))
    } else {
      variables <- vnames
    }
  } else {
    variables <- as.character(variables)
    if (length(variables) != ncol(data)) {
      stop(sprintf(
        "sift$from_cluster: variables has %d entries but data has %d columns",
        length(variables), ncol(data)
      ))
    }
  }
  print(fit)

  # cutree returns an integer vector of length nrow(data) with
  # cluster ids 1..k. NEVER emitted — per-observation assignments
  # are structurally absent from the allowlist.
  assignments <- stats::cutree(fit, k = k)
  cluster_labels <- paste0("cluster_", seq_len(k))
  cluster_sizes <- list()
  centroids <- list()
  within_cluster_ss <- list()
  total_within_ss <- 0
  grand_mean <- colMeans(data)
  for (i in seq_len(k)) {
    cl <- cluster_labels[i]
    members <- which(assignments == i)
    n_i <- length(members)
    cluster_sizes[[cl]] <- as.integer(n_i)
    if (n_i == 0) next
    sub <- data[members, , drop = FALSE]
    centroid_row <- list()
    centroid_vec <- colMeans(sub)
    for (j in seq_len(ncol(data))) {
      centroid_row[[variables[j]]] <- as.numeric(centroid_vec[j])
    }
    centroids[[cl]] <- centroid_row
    # within-cluster SS for cluster i = sum over members of
    # ||x - centroid||² (squared Euclidean distance).
    deviations <- sweep(sub, 2, centroid_vec, FUN = "-")
    wss_i <- sum(deviations^2)
    within_cluster_ss[[cl]] <- as.numeric(wss_i)
    total_within_ss <- total_within_ss + wss_i
  }
  # Total SS (centered at grand mean); between-cluster SS = total - within.
  total_ss <- sum(sweep(data, 2, grand_mean, FUN = "-")^2)
  between_ss <- total_ss - total_within_ss

  # Cut height: the merge height at which exactly k clusters
  # remain. hclust's heights are in fit$height (length n-1).
  cut_height <- NA_real_
  if (!is.null(fit$height) && length(fit$height) >= (nrow(data) - k)) {
    # The cut for k clusters is between the (n-k)-th and (n-k+1)-th
    # merge heights; report the (n-k+1)-th as the threshold height
    # (the height ABOVE which only k clusters remain).
    idx <- length(fit$height) - k + 1
    if (idx >= 1 && idx <= length(fit$height)) {
      cut_height <- fit$height[idx]
    }
  }

  args <- list(
    type = "cluster_analysis",
    method = "hierarchical",
    distance_metric = if (!is.null(fit$dist.method)) fit$dist.method else "euclidean",
    n_observations = as.integer(nrow(data)),
    n_clusters = k,
    n_features = as.integer(ncol(data)),
    variables = as.list(variables),
    cluster_labels = as.list(cluster_labels),
    cluster_sizes = cluster_sizes,
    centroids = centroids
  )
  if (length(within_cluster_ss) > 0) {
    args$within_cluster_ss <- within_cluster_ss
  }
  args$total_within_ss <- as.numeric(total_within_ss)
  args$inertia <- as.numeric(total_within_ss)
  args$between_cluster_ss <- as.numeric(between_ss)
  args$total_ss <- as.numeric(total_ss)
  if (total_ss > 0) {
    args$ss_ratio <- as.numeric(between_ss / total_ss)
  }
  # Linkage method: prefer caller's argument; else read from the
  # fit's ``method`` slot (hclust stores ``"ward.D"`` /
  # ``"ward.D2"`` / ``"complete"`` / ``"average"`` / ``"single"`` /
  # ``"centroid"`` / ``"median"``). Normalize ward.D / ward.D2 to
  # ``"ward"`` since the sanitizer's enum doesn't distinguish.
  if (is.null(linkage)) linkage <- fit$method
  if (!is.null(linkage)) {
    linkage <- as.character(linkage)
    if (grepl("^ward", linkage)) linkage <- "ward"
    args$linkage <- linkage
  }
  if (is.finite(cut_height)) args$cut_height <- as.numeric(cut_height)
  if (!is.null(label)) args$label <- as.character(label)
  do.call(sift$result, args)
}


# Back-compat: ``sift$from_kmeans`` was the public name in earlier
# releases. The class-dispatched ``from_cluster`` is the new
# canonical entry point. Keep both during the transition.
sift$from_kmeans <- function(fit, variables = NULL, label = NULL) {
  sift$from_cluster(fit, variables = variables, label = label)
}


#' From a ``marginaleffects::avg_slopes`` or ``marginaleffects::slopes``
#' result, emit a marginal_effects payload.
#'
#' Wraps the ``marginaleffects`` package (the actively-maintained
#' successor to ``margins``). Both ``avg_slopes()`` (average
#' marginal effects) and ``slopes()`` at a single covariate vector
#' return a data.frame with the same column shape:
#'
#'   * ``term``       — variable name
#'   * ``estimate``   — marginal effect on the response scale
#'   * ``std.error``  — delta-method SE
#'   * ``statistic``  — Wald z (effect / SE)
#'   * ``p.value``
#'   * ``conf.low`` / ``conf.high`` — 95% CI
#'
#' Method mapping:
#'   * ``avg_slopes(fit)``         → ``"ame"``
#'   * ``slopes(fit, newdata="mean")``   → ``"mem"``
#'   * ``slopes(fit, newdata=df_at)``    → ``"at_representative"``
#'     (caller passes the conditioning point via ``at_values=``)
#'
#' Caller passes ``method`` explicitly because we can't always
#' detect ``mean`` vs ``representative`` from the result.
#' ``outcome_variable`` and ``model_family`` ride alongside so the
#' model can interpret the unit of the marginal effect (logit →
#' probability change; Poisson → count change; OLS → outcome change).
#'
#' Example (AME from a logit fit):
#'   m <- glm(y ~ age + female + income, data = df, family = binomial)
#'   ame <- marginaleffects::avg_slopes(m)
#'   sift$from_marginal_effects(
#'     ame, method = "ame", outcome_variable = "y",
#'     model_family = "logit", label = "AME from logit"
#'   )
#'
#' Representative-values form:
#'   slope_45F <- marginaleffects::slopes(
#'     m, newdata = data.frame(age = 45, female = 1, income = 30000)
#'   )
#'   sift$from_marginal_effects(
#'     slope_45F, method = "at_representative",
#'     at_values = list(age = 45, female = 1, income = 30000),
#'     outcome_variable = "y", model_family = "logit"
#'   )
#'
#' Disclosure note on ``at_values`` (only relevant for
#' ``method = "at_representative"``): each conditioning value is
#' precision-clamped by the sample N before it reaches the model —
#' at n=1000 you get ~4 sigfigs, at n=100 you get ~3. Pass
#' interpretable summary points (mean / median / percentiles /
#' round reference values from the literature). An exact-precision
#' value pulled from a single row is gated by the precision floor;
#' it won't cross as raw bytes, but the right interpretation is
#' still "this is a representative point at this precision".
sift$from_marginal_effects <- function(slopes_df,
                                       method,
                                       outcome_variable = NULL,
                                       model_family = NULL,
                                       at_values = NULL,
                                       label = NULL,
                                       n = NULL) {
  if (missing(method) || !is.character(method) || length(method) != 1) {
    stop(
      "sift$from_marginal_effects: ``method`` is required ",
      "(one of 'ame' / 'mem' / 'at_representative')"
    )
  }
  if (is.null(slopes_df) || !is.data.frame(slopes_df)) {
    stop(
      "sift$from_marginal_effects: ``slopes_df`` must be a data.frame ",
      "(marginaleffects::avg_slopes(...) or marginaleffects::slopes(...))"
    )
  }
  print(slopes_df)

  cn <- colnames(slopes_df)
  term_col <- if ("term" %in% cn) "term" else cn[1]
  est_col  <- if ("estimate" %in% cn) "estimate" else
              if ("dydx" %in% cn) "dydx" else
              cn[2]
  se_col   <- if ("std.error" %in% cn) "std.error" else
              if ("se" %in% cn) "se" else NA_character_
  stat_col <- if ("statistic" %in% cn) "statistic" else
              if ("z" %in% cn) "z" else NA_character_
  p_col    <- if ("p.value" %in% cn) "p.value" else
              if ("pvalue" %in% cn) "pvalue" else NA_character_
  lo_col   <- if ("conf.low" %in% cn) "conf.low" else NA_character_
  hi_col   <- if ("conf.high" %in% cn) "conf.high" else NA_character_

  terms <- as.character(slopes_df[[term_col]])
  # marginaleffects emits one row per (term, contrast) combination
  # for categorical variables; collapse to one row per term by
  # taking the first occurrence so the per-variable dict is
  # well-defined. Researchers with multi-contrast categoricals
  # should expand to indicator columns before fitting (same
  # advice the regression-bucket sanitizer gives on
  # formula-categorical names).
  if (anyDuplicated(terms) > 0) {
    first_idx <- !duplicated(terms)
    slopes_df <- slopes_df[first_idx, , drop = FALSE]
    terms <- terms[first_idx]
  }

  effects <- list(); ses <- list(); zs <- list(); ps <- list()
  los <- list(); his <- list()
  for (i in seq_along(terms)) {
    nm <- terms[i]
    v  <- as.numeric(slopes_df[[est_col]][i])
    if (is.finite(v)) effects[[nm]] <- v
    if (!is.na(se_col)) {
      sv <- as.numeric(slopes_df[[se_col]][i])
      if (is.finite(sv)) ses[[nm]] <- sv
    }
    if (!is.na(stat_col)) {
      sv <- as.numeric(slopes_df[[stat_col]][i])
      if (is.finite(sv)) zs[[nm]] <- sv
    }
    if (!is.na(p_col)) {
      sv <- as.numeric(slopes_df[[p_col]][i])
      if (is.finite(sv)) ps[[nm]] <- sv
    }
    if (!is.na(lo_col)) {
      sv <- as.numeric(slopes_df[[lo_col]][i])
      if (is.finite(sv)) los[[nm]] <- sv
    }
    if (!is.na(hi_col)) {
      sv <- as.numeric(slopes_df[[hi_col]][i])
      if (is.finite(sv)) his[[nm]] <- sv
    }
  }

  # n: row count of the fitted model. marginaleffects results carry
  # attr(slopes_df, "newdata") or attr(., "data") with the prediction
  # frame, but the SDC-relevant ``n`` is the underlying training
  # sample size. Probe ``attr(., "model")`` first, then fall back
  # to a caller-supplied ``n=``.
  if (is.null(n)) {
    m_attr <- attr(slopes_df, "model")
    if (!is.null(m_attr)) {
      n <- tryCatch(as.integer(nobs(m_attr)), error = function(e) NULL)
    }
  }

  args <- list(
    type = "marginal_effects",
    method = as.character(method),
    variables = as.list(terms),
    effects = effects
  )
  if (length(ses) > 0) args$standard_errors <- ses
  if (length(zs)  > 0) args$z_statistics    <- zs
  if (length(ps)  > 0) args$p_values        <- ps
  if (length(los) > 0) args$ci_lower        <- los
  if (length(his) > 0) args$ci_upper        <- his
  if (!is.null(n) && is.finite(n)) args$n <- as.integer(n)
  if (!is.null(outcome_variable)) args$outcome_variable <- as.character(outcome_variable)
  if (!is.null(model_family))     args$model_family     <- as.character(model_family)
  if (!is.null(at_values) && length(at_values) > 0) {
    clean_at <- list()
    for (k in names(at_values)) {
      v <- as.numeric(at_values[[k]])
      if (is.finite(v)) clean_at[[k]] <- v
    }
    if (length(clean_at) > 0) args$at_values <- clean_at
  }
  if (!is.null(label)) args$label <- as.character(label)
  do.call(sift$result, args)
}


#' From a ``stats::prcomp`` fit, emit a factor_decomposition payload.
#'
#' Wraps base R's ``prcomp`` (eigen-decomposition of the centered
#' and optionally scaled data matrix). The payload carries the
#' loadings matrix (variable × component), explained-variance
#' ratios, cumulative variance, eigenvalues, and PCA-derived
#' communalities. The full row × component factor-scores matrix
#' (``fit$x``) is researcher-only by construction — no field in
#' the sanitizer's ``factor_decomposition`` allowlist accepts it.
#'
#' By default we report all components prcomp computed (one per
#' input variable). The ``n_components`` argument trims to the top-k
#' for parsimony; the dropped components remain researcher-visible
#' on the original fit object.
#'
#' Example:
#'   m <- prcomp(df[, c("v1","v2","v3","v4","v5")], scale. = TRUE)
#'   sift$from_pca(m, label = "five-variable PCA")
#'
#' To trim to the top three components:
#'   sift$from_pca(m, n_components = 3)
sift$from_pca <- function(fit, n_components = NULL, label = NULL) {
  if (!inherits(fit, "prcomp")) {
    stop("sift$from_pca: ``fit`` must be a prcomp object.")
  }
  print(fit)
  rotation <- fit$rotation
  if (is.null(rotation) || !is.matrix(rotation)) {
    stop("sift$from_pca: fit$rotation is missing or not a matrix")
  }
  variables <- rownames(rotation)
  comp_labels_full <- colnames(rotation)
  total_k <- ncol(rotation)
  if (is.null(n_components)) n_components <- total_k
  n_components <- as.integer(min(n_components, total_k))
  comp_labels <- comp_labels_full[seq_len(n_components)]

  # nobs(fit) is the post-fit sample size. ``fit$x`` (the scores
  # matrix) carries it row-wise; we never emit ``$x``, only the
  # row count.
  n_obs <- if (!is.null(fit$x)) nrow(fit$x) else NA_integer_

  # Build loadings as {variable: {component: value}}.
  loadings <- list()
  for (v in variables) {
    row <- list()
    for (i in seq_len(n_components)) {
      row[[comp_labels[i]]] <- as.numeric(rotation[v, i])
    }
    loadings[[v]] <- row
  }

  # Explained variance: sdev² is the variance per component (the
  # eigenvalues when scale.=TRUE, since we work on the correlation
  # matrix). Ratio = eigenvalue / sum(eigenvalues). Cumulative is
  # the running sum.
  eigenvalues_full <- as.numeric(fit$sdev^2)
  total_var <- sum(eigenvalues_full)
  ev_ratio_full <- eigenvalues_full / total_var
  cum_var_full <- cumsum(ev_ratio_full)

  eigenvalues <- list()
  explained_variance <- list()
  explained_variance_ratio <- list()
  cumulative_variance <- list()
  for (i in seq_len(n_components)) {
    eigenvalues[[comp_labels[i]]] <- as.numeric(eigenvalues_full[i])
    explained_variance[[comp_labels[i]]] <- as.numeric(eigenvalues_full[i])
    explained_variance_ratio[[comp_labels[i]]] <- as.numeric(ev_ratio_full[i])
    cumulative_variance[[comp_labels[i]]] <- as.numeric(cum_var_full[i])
  }

  # Communalities: sum of squared loadings across the retained
  # components, per variable. = 1 when all components retained.
  communalities <- list()
  for (v in variables) {
    h2 <- sum(rotation[v, seq_len(n_components)]^2)
    communalities[[v]] <- as.numeric(h2)
  }

  args <- list(
    type = "factor_decomposition",
    method = "pca",
    rotation = "none",
    n_observations = as.integer(n_obs),
    n_variables = as.integer(length(variables)),
    n_components = n_components,
    variables = as.list(variables),
    components = as.list(comp_labels),
    loadings = loadings,
    explained_variance = explained_variance,
    explained_variance_ratio = explained_variance_ratio,
    cumulative_variance = cumulative_variance,
    eigenvalues = eigenvalues,
    communalities = communalities
  )
  if (!is.null(label)) args$label <- as.character(label)
  do.call(sift$result, args)
}


#' From a ``psych::fa`` factor analysis fit, emit a
#' factor_decomposition payload.
#'
#' Wraps the ``psych`` package's ``fa`` (the standard R factor-analysis
#' implementation — minimum residual / maximum likelihood / principal
#' factor extraction with all the conventional rotations). The
#' payload carries:
#'   * loadings:        {variable: {factor: value}}
#'   * communalities:   {variable: h²}
#'   * uniqueness:      {variable: 1 - h²}
#'   * eigenvalues + explained_variance (per factor)
#'   * KMO / Bartlett goodness-of-fit when available
#'
#' ``method`` defaults to the extraction routine from the fit
#' (``fit$fm``: "ml" / "minres" / "pa" / etc.) mapped onto the
#' sanitizer's enum (``maximum_likelihood`` / ``minimum_residual`` /
#' ``principal_factor`` / ``factor_analysis``). ``rotation`` similarly
#' comes from ``fit$rotation``. The full row × factor factor-scores
#' matrix (``fit$scores``) is researcher-only by structural absence —
#' no field on the sanitizer's allowlist accepts it.
#'
#' Example:
#'   library(psych)
#'   m <- fa(df[, c("v1","v2","v3","v4","v5")], nfactors = 2,
#'           rotate = "varimax", fm = "ml")
#'   sift$from_fa(m, label = "ML factor analysis with varimax")
sift$from_fa <- function(fit, label = NULL) {
  if (!(inherits(fit, "fa") || inherits(fit, "psych"))) {
    stop("sift$from_fa: ``fit`` must be a psych::fa result")
  }
  print(fit)

  load_mat <- unclass(fit$loadings)
  if (is.null(load_mat) || !is.matrix(load_mat)) {
    stop("sift$from_fa: fit$loadings missing or not a matrix")
  }
  variables  <- rownames(load_mat)
  fac_labels <- colnames(load_mat)
  n_factors  <- ncol(load_mat)
  if (is.null(variables) || length(variables) == 0) {
    stop("sift$from_fa: loadings matrix is missing variable names")
  }

  loadings <- list()
  for (v in variables) {
    row <- list()
    for (j in seq_len(n_factors)) {
      row[[fac_labels[j]]] <- as.numeric(load_mat[v, j])
    }
    loadings[[v]] <- row
  }

  # Method: map psych's ``fm`` codes onto the sanitizer enum.
  fm <- if (!is.null(fit$fm)) tolower(as.character(fit$fm)) else "minres"
  method <- switch(
    fm,
    "ml"     = "maximum_likelihood",
    "minres" = "minimum_residual",
    "pa"     = "principal_factor",
    "factor_analysis"
  )
  # Rotation: psych stores ``$rotation`` as the human-readable
  # name ("varimax", "promax", "oblimin", "none"); pass through
  # when in the sanitizer's valid set, otherwise drop to "none".
  valid_rot <- c("none", "varimax", "promax", "oblimin",
                 "quartimax", "equamax", "geomin", "bentlerT", "bifactor")
  rotation <- if (!is.null(fit$rotation) && fit$rotation %in% valid_rot)
                as.character(fit$rotation) else "none"

  # Communalities and uniqueness — psych exposes both directly.
  communalities <- list()
  uniqueness <- list()
  if (!is.null(fit$communality)) {
    for (v in variables) {
      h2 <- as.numeric(fit$communality[[v]])
      if (is.finite(h2)) communalities[[v]] <- h2
    }
  }
  if (!is.null(fit$uniquenesses)) {
    for (v in variables) {
      u <- as.numeric(fit$uniquenesses[[v]])
      if (is.finite(u)) uniqueness[[v]] <- u
    }
  }

  # Per-factor variance metrics. ``fit$Vaccounted`` is a small
  # named matrix; row "SS loadings" gives eigenvalues / explained
  # variance per factor, row "Proportion Var" gives the ratio,
  # row "Cumulative Var" gives the running cumulative.
  eigenvalues <- list()
  explained_variance <- list()
  explained_variance_ratio <- list()
  cumulative_variance <- list()
  if (!is.null(fit$Vaccounted) && is.matrix(fit$Vaccounted)) {
    rn <- rownames(fit$Vaccounted)
    grab_row <- function(name) {
      if (name %in% rn) as.numeric(fit$Vaccounted[name, ]) else NULL
    }
    ss <- grab_row("SS loadings")
    pv <- grab_row("Proportion Var")
    cv <- grab_row("Cumulative Var")
    for (j in seq_len(n_factors)) {
      if (!is.null(ss) && j <= length(ss) && is.finite(ss[j])) {
        eigenvalues[[fac_labels[j]]] <- as.numeric(ss[j])
        explained_variance[[fac_labels[j]]] <- as.numeric(ss[j])
      }
      if (!is.null(pv) && j <= length(pv) && is.finite(pv[j])) {
        explained_variance_ratio[[fac_labels[j]]] <- as.numeric(pv[j])
      }
      if (!is.null(cv) && j <= length(cv) && is.finite(cv[j])) {
        cumulative_variance[[fac_labels[j]]] <- as.numeric(cv[j])
      }
    }
  }

  # Sample size: psych stores ``fit$n.obs`` (or ``$nh``).
  n_obs <- if (!is.null(fit$n.obs)) as.integer(fit$n.obs) else NA_integer_
  if (is.na(n_obs) && !is.null(fit$nh)) n_obs <- as.integer(fit$nh)

  args <- list(
    type = "factor_decomposition",
    method = method,
    rotation = rotation,
    n_observations = n_obs,
    n_variables = as.integer(length(variables)),
    n_components = as.integer(n_factors),
    variables = as.list(variables),
    components = as.list(fac_labels),
    loadings = loadings
  )
  if (length(communalities) > 0)            args$communalities <- communalities
  if (length(uniqueness) > 0)               args$uniqueness <- uniqueness
  if (length(eigenvalues) > 0)              args$eigenvalues <- eigenvalues
  if (length(explained_variance) > 0)       args$explained_variance <- explained_variance
  if (length(explained_variance_ratio) > 0) args$explained_variance_ratio <- explained_variance_ratio
  if (length(cumulative_variance) > 0)      args$cumulative_variance <- cumulative_variance

  # Goodness-of-fit scalars exposed on ML fits.
  if (!is.null(fit$chi) && is.finite(fit$chi)) {
    args$chi_squared <- as.numeric(fit$chi)
  }
  if (!is.null(fit$PVAL) && is.finite(fit$PVAL)) {
    args$chi_squared_p_value <- as.numeric(fit$PVAL)
  }
  if (!is.null(fit$dof) && is.finite(fit$dof)) {
    args$degrees_of_freedom <- as.integer(fit$dof)
  }
  if (!is.null(fit$RMSEA) && length(fit$RMSEA) >= 1 && is.finite(fit$RMSEA[1])) {
    args$rmsea <- as.numeric(fit$RMSEA[1])
  }
  if (!is.null(fit$TLI) && is.finite(fit$TLI)) {
    args$tli <- as.numeric(fit$TLI)
  }

  if (!is.null(label)) args$label <- as.character(label)
  do.call(sift$result, args)
}


#' From a Sun-Abraham interaction-weighted event-study fit, emit a
#' did_event_study payload.
#'
#' Wraps ``fixest::feols()`` with ``fixest::sunab(cohort, time)`` in
#' the formula — the interaction-weighted estimator from Sun &
#' Abraham (2021). The estimator's natural output is one ATT per
#' event-time (already aggregated across cohorts via IW weights),
#' not per (cohort, event-time). We package this as a
#' did_event_study payload with a single synthetic cohort ``"all"``
#' whose ATT series IS the event-time aggregate; the
#' ``estimator: "sun_abraham"`` field tells the model the
#' aggregation happened inside the estimator.
#'
#' The caller passes ``n_treated`` (total treated units across the
#' cohorts that fed the IW weights) so the cohort-N gate has its
#' input. The helper can't recover this from the feols result
#' without re-walking the data — better to make it explicit.
#'
#' Example:
#'   m <- feols(y ~ sunab(cohort, period) | id + period,
#'              data = df, cluster = ~id)
#'   n_treated <- length(unique(df$id[df$cohort <= max(df$period)]))
#'   sift$from_sun_abraham(m, n_treated = n_treated,
#'                         outcome_variable = "y",
#'                         treatment_variable = "cohort")
sift$from_sun_abraham <- function(fit,
                                  n_treated,
                                  outcome_variable = NULL,
                                  treatment_variable = NULL,
                                  label = NULL,
                                  event_time_pattern = "(period|event_time|rel_time|et)::([^:]+)") {
  if (!inherits(fit, "fixest")) {
    stop("sift$from_sun_abraham: ``fit`` must be a fixest::feols result")
  }
  if (missing(n_treated) || !is.numeric(n_treated) || n_treated < 0) {
    stop("sift$from_sun_abraham: ``n_treated`` (total treated units) is required")
  }
  print(fit)

  # fixest's aggregate() collapses the cohort-by-event-time
  # interactions to per-event-time ATTs via IW weights. The
  # pattern argument captures the event-time portion of the
  # coefficient name (sunab produces names like "period::-3").
  agg <- tryCatch(
    aggregate(fit, event_time_pattern),
    error = function(e) {
      stop("sift$from_sun_abraham: aggregate(fit, ...) failed — was the fit ",
           "produced via feols(y ~ sunab(cohort, time) | ...)? ",
           "Error: ", conditionMessage(e))
    }
  )
  if (is.null(agg) || nrow(agg) == 0) {
    stop("sift$from_sun_abraham: aggregated coefficient table is empty")
  }

  # Extract event-time integer from each coefficient name.
  coef_names <- rownames(agg)
  m_extract <- regmatches(coef_names, regexec(event_time_pattern, coef_names))
  # Use the LAST capture group as the event-time integer, regardless
  # of how many groups the user supplied (default pattern has 2 groups
  # — prefix + integer; caller may pass a pattern with 1 group).
  # ``unname()`` strips any names ``sapply`` carries through from the
  # matched coefficient names — without it, ``as.list(event_times)``
  # would build a NAMED list, which the JSON serializer emits as an
  # OBJECT, breaking the sanitizer's "event_times must be a list of
  # finite numbers" check.
  event_times <- unname(sapply(m_extract, function(x) {
    if (length(x) >= 2) as.integer(x[length(x)]) else NA_integer_
  }))
  keep <- !is.na(event_times)
  if (!any(keep)) {
    stop("sift$from_sun_abraham: no event-time coefficients matched pattern")
  }
  event_times <- event_times[keep]
  agg <- agg[keep, , drop = FALSE]

  att_all <- list()
  se_all  <- list()
  p_all   <- list()
  ci_lo   <- list()
  ci_hi   <- list()
  for (i in seq_along(event_times)) {
    et_lab <- as.character(event_times[i])
    est <- as.numeric(agg[i, "Estimate"])
    se  <- as.numeric(agg[i, "Std. Error"])
    pv  <- as.numeric(agg[i, "Pr(>|t|)"])
    att_all[[et_lab]] <- est
    se_all[[et_lab]]  <- se
    p_all[[et_lab]]   <- pv
    if (is.finite(est) && is.finite(se) && se > 0) {
      ci_lo[[et_lab]] <- est - 1.96 * se
      ci_hi[[et_lab]] <- est + 1.96 * se
    }
  }

  args <- list(
    type = "did_event_study",
    estimator = "sun_abraham",
    aggregation_method = "dynamic",
    groups = list("all"),
    event_times = as.list(sort(event_times)),
    att = list(all = att_all),
    standard_errors = list(all = se_all),
    p_values = list(all = p_all),
    ci_lower = list(all = ci_lo),
    ci_upper = list(all = ci_hi),
    n_treated_per_group = list(all = as.integer(n_treated))
  )
  if (!is.null(outcome_variable))   args$outcome_variable   <- as.character(outcome_variable)
  if (!is.null(treatment_variable)) args$treatment_variable <- as.character(treatment_variable)
  if (!is.null(label))              args$label              <- as.character(label)
  do.call(sift$result, args)
}


#' From a TWFE event-study regression (any feols / lm with i(rel_time,
#' treated, ref=0) style interactions), emit a did_event_study payload.
#'
#' Differs from Sun-Abraham only in the estimator label and the
#' identification assumptions the model needs to know about
#' (TWFE-ES is biased under treatment-effect heterogeneity; the
#' Sun-Abraham IW estimator is the heterogeneity-robust version).
#' Same single-synthetic-cohort payload shape.
#'
#' Caller passes ``event_time_pattern`` matching the coefficient
#' names (regex with one capture group for the event-time integer).
#' Default matches ``rel_time::N`` / ``event_time::N`` / ``period::N``.
#'
#' Example:
#'   m <- feols(y ~ i(rel_time, treated, ref=-1) | id + period, data = df)
#'   sift$from_twfe_event_study(m, n_treated = sum(df$treated > 0),
#'                              outcome_variable = "y",
#'                              event_time_pattern = "rel_time::([^:]+):")
sift$from_twfe_event_study <- function(fit,
                                       n_treated,
                                       outcome_variable = NULL,
                                       treatment_variable = NULL,
                                       label = NULL,
                                       event_time_pattern = "(rel_time|event_time|period|et)::([^:]+)") {
  if (!inherits(fit, "fixest") && !inherits(fit, "lm")) {
    stop("sift$from_twfe_event_study: ``fit`` must be feols or lm")
  }
  if (missing(n_treated) || !is.numeric(n_treated) || n_treated < 0) {
    stop("sift$from_twfe_event_study: ``n_treated`` is required")
  }
  print(fit)

  ct <- tryCatch(
    if (inherits(fit, "fixest")) coeftable(fit) else as.data.frame(summary(fit)$coefficients),
    error = function(e) stop("sift$from_twfe_event_study: coefficient table unreachable: ", conditionMessage(e))
  )
  if (is.null(ct) || nrow(ct) == 0) {
    stop("sift$from_twfe_event_study: empty coefficient table")
  }

  coef_names <- rownames(ct)
  m_extract <- regmatches(coef_names, regexec(event_time_pattern, coef_names))
  # Use the LAST capture group as the event-time integer, regardless
  # of how many groups the user supplied (default pattern has 2 groups
  # — prefix + integer; caller may pass a pattern with 1 group).
  # ``unname()`` strips any names ``sapply`` carries through from the
  # matched coefficient names — without it, ``as.list(event_times)``
  # would build a NAMED list, which the JSON serializer emits as an
  # OBJECT, breaking the sanitizer's "event_times must be a list of
  # finite numbers" check.
  event_times <- unname(sapply(m_extract, function(x) {
    if (length(x) >= 2) as.integer(x[length(x)]) else NA_integer_
  }))
  keep <- !is.na(event_times)
  if (!any(keep)) {
    stop("sift$from_twfe_event_study: no event-time coefficients matched ",
         "the pattern. The default pattern matches rel_time::N / ",
         "event_time::N / period::N — pass ``event_time_pattern`` if ",
         "your design uses a different naming convention.")
  }
  event_times <- event_times[keep]
  ct <- ct[keep, , drop = FALSE]

  ce_cols <- colnames(ct)
  est_col <- if ("Estimate" %in% ce_cols) "Estimate" else ce_cols[1]
  se_col  <- if ("Std. Error" %in% ce_cols) "Std. Error" else ce_cols[2]
  p_col   <- if ("Pr(>|t|)" %in% ce_cols) "Pr(>|t|)" else
             if ("Pr(>|z|)" %in% ce_cols) "Pr(>|z|)" else
             ce_cols[ncol(ct)]

  att_all <- list(); se_all <- list(); p_all <- list()
  ci_lo <- list(); ci_hi <- list()
  for (i in seq_along(event_times)) {
    et_lab <- as.character(event_times[i])
    est <- as.numeric(ct[i, est_col])
    se  <- as.numeric(ct[i, se_col])
    pv  <- as.numeric(ct[i, p_col])
    att_all[[et_lab]] <- est
    se_all[[et_lab]]  <- se
    p_all[[et_lab]]   <- pv
    if (is.finite(est) && is.finite(se) && se > 0) {
      ci_lo[[et_lab]] <- est - 1.96 * se
      ci_hi[[et_lab]] <- est + 1.96 * se
    }
  }

  args <- list(
    type = "did_event_study",
    estimator = "twfe_event_study",
    aggregation_method = "dynamic",
    groups = list("all"),
    event_times = as.list(sort(event_times)),
    att = list(all = att_all),
    standard_errors = list(all = se_all),
    p_values = list(all = p_all),
    ci_lower = list(all = ci_lo),
    ci_upper = list(all = ci_hi),
    n_treated_per_group = list(all = as.integer(n_treated))
  )
  if (!is.null(outcome_variable))   args$outcome_variable   <- as.character(outcome_variable)
  if (!is.null(treatment_variable)) args$treatment_variable <- as.character(treatment_variable)
  if (!is.null(label))              args$label              <- as.character(label)
  do.call(sift$result, args)
}


#' From a ``did::att_gt`` MP object, emit a did_event_study payload.
#'
#' Wraps the Callaway-Sant'Anna heterogeneous-treatment DiD estimator.
#' The MP object carries pre-aggregation ATT(g, t) — one estimate per
#' (cohort g, calendar time t). The helper:
#'   * Pivots ATT(g, t) → ATT(g, event_time) where event_time = t - g
#'   * Pulls per-cohort treated counts from
#'     ``mp$DIDparams$cohort_counts`` (data.table with cohort + size)
#'   * Optionally runs ``aggte(mp, type="dynamic")`` to add the
#'     aggregate ATT and event-time-aggregated series
#'   * Tags ``estimator: "callaway_santanna"``
#'
#' Privacy carve-out: the sanitizer's cohort-N gate fires on
#' ``n_treated_per_group``. Cohorts below ``min_n_did_cohort`` are
#' dropped *whole* (entire cohort row stripped from the ATT matrix);
#' partial-cell publication would leak the cohort size through
#' which cells survived. The helper emits the raw cohort sizes
#' unchanged — the suppression decision belongs to the sanitizer.
#'
#' Example:
#'   mp <- att_gt(yname = "y", tname = "period", idname = "id",
#'                gname = "G", data = df,
#'                control_group = "nevertreated")
#'   sift$from_callaway_santanna(mp,
#'     outcome_variable = "y", treatment_variable = "G",
#'     label = "headline DiD")
sift$from_callaway_santanna <- function(mp,
                                        outcome_variable = NULL,
                                        treatment_variable = NULL,
                                        aggregation_method = "dynamic",
                                        label = NULL, ...) {
  if (!inherits(mp, "MP")) {
    stop(
      "sift$from_callaway_santanna: ``mp`` must be a did::att_gt result ",
      "(an ``MP`` object). Run ``att_gt(...)`` first and pass the result."
    )
  }
  print(mp)

  # Cohort labels (treated cohorts in mp$group; sorted ascending).
  cohorts <- sort(unique(mp$group))
  if (length(cohorts) == 0) {
    stop("sift$from_callaway_santanna: no treated cohorts found in mp$group")
  }
  cohort_labels <- as.character(cohorts)

  # Event-time grid: union of (t - g) across all (g, t) entries.
  event_times_all <- mp$t - mp$group
  event_time_grid <- sort(unique(event_times_all))

  # Build ATT(g, e) and SE(g, e) nested dicts.
  att_dict <- list()
  se_dict <- list()
  ci_lo_dict <- list()
  ci_hi_dict <- list()
  for (g in cohorts) {
    g_lab <- as.character(g)
    att_dict[[g_lab]] <- list()
    se_dict[[g_lab]] <- list()
    ci_lo_dict[[g_lab]] <- list()
    ci_hi_dict[[g_lab]] <- list()
    idx <- which(mp$group == g)
    # mp$c is the critical value for the CS uniform CI (one scalar);
    # multiply by se to get per-cell CI half-widths.
    crit <- if (!is.null(mp$c) && length(mp$c) >= 1 && is.finite(mp$c[1])) mp$c[1] else 1.96
    for (i in idx) {
      e <- mp$t[i] - mp$group[i]
      e_lab <- as.character(e)
      att_dict[[g_lab]][[e_lab]] <- as.numeric(mp$att[i])
      se_dict[[g_lab]][[e_lab]] <- as.numeric(mp$se[i])
      ci_lo_dict[[g_lab]][[e_lab]] <- as.numeric(mp$att[i] - crit * mp$se[i])
      ci_hi_dict[[g_lab]][[e_lab]] <- as.numeric(mp$att[i] + crit * mp$se[i])
    }
  }

  # Per-cohort treated counts. mp$DIDparams$cohort_counts is a
  # data.table with rows {cohort, cohort_size}. Skip the never-
  # treated row (cohort == Inf) and any cohort not in mp$group.
  cc <- mp$DIDparams$cohort_counts
  n_treated_per_group <- list()
  if (!is.null(cc)) {
    for (i in seq_len(nrow(cc))) {
      g_val <- cc$cohort[i]
      if (is.finite(g_val) && g_val %in% cohorts) {
        n_treated_per_group[[as.character(g_val)]] <-
          as.integer(cc$cohort_size[i])
      }
    }
  }

  args <- list(
    type = "did_event_study",
    estimator = "callaway_santanna",
    groups = as.list(cohort_labels),
    event_times = as.list(event_time_grid),
    att = att_dict,
    standard_errors = se_dict,
    ci_lower = ci_lo_dict,
    ci_upper = ci_hi_dict,
    n_treated_per_group = n_treated_per_group,
    aggregation_method = as.character(aggregation_method)
  )
  if (!is.null(outcome_variable))   args$outcome_variable   <- as.character(outcome_variable)
  if (!is.null(treatment_variable)) args$treatment_variable <- as.character(treatment_variable)
  if (!is.null(label))              args$label              <- as.character(label)

  # Pass through CS config so the model knows the identification
  # assumptions the estimator ran under.
  ctrl_grp <- mp$DIDparams$control_group
  if (!is.null(ctrl_grp) && nzchar(ctrl_grp)) {
    args$comparison_group <- as.character(ctrl_grp)
  }
  antic <- mp$DIDparams$anticipation
  if (!is.null(antic) && length(antic) == 1 && is.finite(antic)) {
    args$anticipation_periods <- as.integer(antic)
  }
  bp <- mp$DIDparams$base_period
  if (!is.null(bp) && nzchar(bp)) {
    args$base_period <- as.character(bp)
  }

  # Aggregate scalars via aggte() — wrap in tryCatch because some
  # data shapes (single cohort, balanced-only requests) can fail
  # inside aggte; omit aggregate fields rather than blowing up the
  # whole emit.
  es <- tryCatch(
    did::aggte(mp, type = aggregation_method),
    error = function(e) NULL
  )
  if (!is.null(es)) {
    if (!is.null(es$overall.att) && is.finite(es$overall.att)) {
      args$aggregate_att <- as.numeric(es$overall.att)
    }
    if (!is.null(es$overall.se) && is.finite(es$overall.se)) {
      args$aggregate_se <- as.numeric(es$overall.se)
      # Two-sided z-test p-value for the aggregate.
      if (es$overall.se > 0) {
        z <- abs(es$overall.att / es$overall.se)
        args$aggregate_p_value <- as.numeric(2 * pnorm(-z))
        crit <- if (!is.null(es$crit.val.egt) && length(es$crit.val.egt) >= 1
                    && is.finite(es$crit.val.egt[1])) es$crit.val.egt[1] else 1.96
        args$aggregate_ci_lower <- as.numeric(es$overall.att - crit * es$overall.se)
        args$aggregate_ci_upper <- as.numeric(es$overall.att + crit * es$overall.se)
      }
    }
  }

  do.call(sift$result, c(args, list(...)))
}


#' From an ``rdrobust`` fit, emit an ``rdd`` payload.
#'
#' Wraps the rdrobust package (CCT 2014) — the standard cross-language
#' implementation maintained by Calonico-Cattaneo-Titiunik. The
#' payload carries the three-flavor τ table (conventional, bias-
#' corrected, robust), bandwidth(s), kernel, polynomial order,
#' effective N per side, and the bandwidth selector name. For fuzzy
#' RDD pass ``fuzzy_treatment_variable`` (the endogenous treatment
#' indicator) so the estimator is tagged ``fuzzy_2sls``.
#'
#' Privacy carve-out is structural: the helper signature does not
#' accept density / binscatter / mccrary arguments at all (not even
#' to drop them). McCrary density tests and binscatter near the
#' cutoff are visual diagnostics for the researcher; they have no
#' field on the ``rdd`` shape's allowlist either, so even hand-
#' crafted payloads through ``sift$result(type="rdd", ...)`` cannot
#' smuggle them.
#'
#' Example:
#'   m <- rdrobust(y, x, c = 50000)
#'   sift$from_rdd(m, running_variable = "income",
#'                 outcome_variable = "voted", label = "headline RDD")
#'
#'   # Fuzzy RDD: pass the treatment-receipt indicator.
#'   m <- rdrobust(y, x, c = 50000, fuzzy = takeup)
#'   sift$from_rdd(m, running_variable = "income",
#'                 outcome_variable = "voted",
#'                 fuzzy_treatment_variable = "takeup")
sift$from_rdd <- function(fit,
                          running_variable = NULL,
                          outcome_variable = NULL,
                          fuzzy_treatment_variable = NULL,
                          first_stage_f = NULL,
                          label = NULL, ...) {
  if (!inherits(fit, "rdrobust")) {
    stop(
      "sift$from_rdd: ``fit`` must be an rdrobust object. ",
      "Use rdrobust::rdrobust(y, x, c = cutoff) and pass the result."
    )
  }
  # Privacy carve-out: refuse density/binscatter arguments if a
  # researcher tries to pass them through ``...`` — these are
  # researcher-only diagnostics for an RDD, not analytical fields.
  extra <- list(...)
  banned <- c("mccrary_density_curve", "mccrary_density",
              "binscatter_bins", "binscatter", "density_curve")
  for (b in banned) {
    if (!is.null(extra[[b]])) {
      stop(
        "sift$from_rdd: ``", b, "`` is a visual diagnostic for the ",
        "researcher and is not allowed on the rdd payload. The model ",
        "sees the analytical fields (tau / bandwidth / effective N); ",
        "ask the researcher qualitatively about manipulation evidence ",
        "if it bears on the design."
      )
    }
  }
  print(fit)

  args <- list(
    type = "rdd",
    estimator = if (!is.null(fuzzy_treatment_variable))
                  "fuzzy_2sls" else "local_polynomial"
  )
  if (!is.null(running_variable))  args$running_variable  <- as.character(running_variable)
  if (!is.null(outcome_variable))  args$outcome_variable  <- as.character(outcome_variable)
  if (!is.null(label))             args$label             <- as.character(label)

  # rdrobust's ``Estimate`` row is [tau.us, tau.bc, se.us, se.rb]:
  #   tau.us = conventional point estimate
  #   tau.bc = bias-corrected point estimate (also used for robust)
  #   se.us  = conventional SE
  #   se.rb  = robust SE
  # rdrobust's ``se`` / ``pv`` / ``ci`` rows map to:
  #   [1] Conventional, [2] Bias-Corrected, [3] Robust
  if (!is.null(fit$Estimate) && is.matrix(fit$Estimate) && ncol(fit$Estimate) >= 4) {
    args$tau_conventional   <- as.numeric(fit$Estimate[1, 1])
    args$tau_bias_corrected <- as.numeric(fit$Estimate[1, 2])
    args$tau_robust         <- as.numeric(fit$Estimate[1, 2])
  }
  if (!is.null(fit$se) && length(fit$se) >= 3) {
    args$se_conventional   <- as.numeric(fit$se[1])
    args$se_bias_corrected <- as.numeric(fit$se[2])
    args$se_robust         <- as.numeric(fit$se[3])
  }
  if (!is.null(fit$pv) && length(fit$pv) >= 3) {
    args$p_conventional   <- as.numeric(fit$pv[1])
    args$p_bias_corrected <- as.numeric(fit$pv[2])
    args$p_robust         <- as.numeric(fit$pv[3])
  }
  if (!is.null(fit$ci) && is.matrix(fit$ci) && nrow(fit$ci) >= 3 && ncol(fit$ci) >= 2) {
    args$ci_lower_conventional   <- as.numeric(fit$ci[1, 1])
    args$ci_upper_conventional   <- as.numeric(fit$ci[1, 2])
    args$ci_lower_bias_corrected <- as.numeric(fit$ci[2, 1])
    args$ci_upper_bias_corrected <- as.numeric(fit$ci[2, 2])
    args$ci_lower_robust         <- as.numeric(fit$ci[3, 1])
    args$ci_upper_robust         <- as.numeric(fit$ci[3, 2])
  }

  # Bandwidth(s): h = main, b = bias-correction. rdrobust stores
  # them as a 2x2 matrix (rows = h/b, cols = left/right).
  if (!is.null(fit$bws) && is.matrix(fit$bws) && nrow(fit$bws) >= 2 && ncol(fit$bws) >= 2) {
    args$bandwidth_left  <- as.numeric(fit$bws[1, 1])
    args$bandwidth_right <- as.numeric(fit$bws[1, 2])
    args$bandwidth_bias_correction_left  <- as.numeric(fit$bws[2, 1])
    args$bandwidth_bias_correction_right <- as.numeric(fit$bws[2, 2])
  }

  # Effective N inside the main bandwidth — the SDC-relevant counts
  # that gate the rdd payload. rdrobust's ``N_h`` is a 2-vector
  # [left, right].
  if (!is.null(fit$N_h) && length(fit$N_h) >= 2) {
    args$effective_n_left  <- as.integer(fit$N_h[1])
    args$effective_n_right <- as.integer(fit$N_h[2])
  }
  args$polynomial_order <- as.integer(fit$p)
  args$cutoff           <- as.numeric(fit$c)

  if (!is.null(fit$bwselect)) {
    args$bandwidth_selector <- as.character(fit$bwselect)
  }
  # rdrobust reports kernel capitalized ("Triangular"); the sanitizer
  # accepts lowercase only.
  if (!is.null(fit$kernel)) {
    args$kernel <- tolower(as.character(fit$kernel))
  }

  # Fuzzy first-stage F. rdrobust doesn't compute this automatically;
  # the caller passes it via the kwarg (compute first-stage F by
  # regressing the endogenous-treatment indicator on the running-
  # variable polynomial inside the bandwidth, then F-test the cutoff
  # dummy).
  if (!is.null(first_stage_f) && is.finite(as.numeric(first_stage_f))) {
    args$first_stage_f <- as.numeric(first_stage_f)
  }

  do.call(sift$result, args)
}


#' From a survival::survfit fit, emit a kaplan_meier payload (safe form).
#'
#' The sanitizer's ``kaplan_meier`` shape ships median survival (with
#' CI) plus survival at preset canonical horizons (``1y`` / ``3y`` /
#' ``5y`` / ``10y``) — each gated by per-horizon ``n_at_risk_h``.
#' The full step function (S(t) at every event time) is researcher-only
#' by construction; this helper does NOT emit it.
#'
#' Time-unit translation is the caller's responsibility. The
#' ``horizons`` argument is a named numeric vector mapping canonical
#' labels to the numeric time in whatever units the fit was built in:
#'
#'   # Data measured in years
#'   sift$from_kaplan_meier(fit, horizons = c("1y" = 1, "3y" = 3, "5y" = 5),
#'                          time_variable = "t_obs", event_variable = "cens")
#'
#'   # Data measured in months
#'   sift$from_kaplan_meier(fit, horizons = c("1y" = 12, "3y" = 36),
#'                          time_variable = "follow_up_months",
#'                          event_variable = "dead")
#'
#' For grouped / log-rank inference, pass the unstratified ``survfit``
#' for the horizon scalars and the ``survdiff`` result separately so
#' the helper can extract the chi² and df.
#'
#'   sd <- survdiff(Surv(t, e) ~ arm, data = df)
#'   sift$from_kaplan_meier(survfit(Surv(t, e) ~ 1, data = df),
#'                          horizons = c("1y" = 1, "3y" = 3),
#'                          time_variable = "t", event_variable = "e",
#'                          group_variable = "arm", survdiff = sd)
sift$from_kaplan_meier <- function(fit, horizons = NULL,
                                   time_variable = NULL,
                                   event_variable = NULL,
                                   group_variable = NULL,
                                   survdiff = NULL,
                                   label = NULL, ...) {
  if (!inherits(fit, "survfit")) {
    stop("sift$from_kaplan_meier: ``fit`` must be a survival::survfit object")
  }
  print(fit)
  if (!is.null(fit$strata)) {
    stop(
      "sift$from_kaplan_meier: ``fit`` is stratified. Fit an UNSTRATIFIED ",
      "survfit (e.g. ``survfit(Surv(t, e) ~ 1, data = df)``) for the ",
      "horizon scalars and pass the ``survdiff`` result separately for ",
      "log-rank inference."
    )
  }

  args <- list(type = "kaplan_meier")
  if (!is.null(time_variable))   args$time_variable   <- as.character(time_variable)
  if (!is.null(event_variable))  args$event_variable  <- as.character(event_variable)
  if (!is.null(group_variable))  args$group_variable  <- as.character(group_variable)
  if (!is.null(label))           args$label           <- as.character(label)

  args$n_subjects <- as.integer(fit$n)
  args$n_failures <- as.integer(sum(fit$n.event))

  # Median + CI. ``quantile.survfit`` returns NA when the curve doesn't
  # cross 0.5 (heavily censored studies); omit those fields gracefully
  # rather than emitting null and provoking a sanitizer transformation
  # note on every KM payload.
  med <- tryCatch(
    quantile(fit, 0.5, conf.int = TRUE),
    error = function(e) NULL
  )
  if (!is.null(med)) {
    mq <- med$quantile; ml <- med$lower; mh <- med$upper
    if (length(mq) >= 1 && is.finite(mq[1])) args$median_survival_time <- as.numeric(mq[1])
    if (length(ml) >= 1 && is.finite(ml[1])) args$median_survival_ci_lower <- as.numeric(ml[1])
    if (length(mh) >= 1 && is.finite(mh[1])) args$median_survival_ci_upper <- as.numeric(mh[1])
  }

  # Per-horizon S(t) and n.risk. ``horizons`` maps canonical labels
  # to numeric time values in the fit's units. ``summary(fit, times=)``
  # with ``extend = TRUE`` ensures lookups past the last event time
  # return NA rather than dropping the row.
  if (!is.null(horizons) && length(horizons) > 0) {
    labels <- names(horizons)
    if (is.null(labels) || any(!nzchar(labels))) {
      stop(
        "sift$from_kaplan_meier: ``horizons`` must be a NAMED vector ",
        "mapping canonical labels (\"1y\", \"3y\", \"5y\", \"10y\") to ",
        "numeric time values in the fit's units. Unnamed entries would ",
        "ship without horizon labels the sanitizer can recognise."
      )
    }
    times_at <- as.numeric(unname(horizons))
    s <- tryCatch(
      summary(fit, times = times_at, extend = TRUE),
      error = function(e) NULL
    )
    if (!is.null(s)) {
      for (i in seq_along(times_at)) {
        lab <- labels[i]
        s_val <- s$surv[i]
        n_val <- s$n.risk[i]
        if (!is.na(s_val) && is.finite(s_val)) {
          args[[paste0("survival_at_", lab)]] <- as.numeric(s_val)
        }
        if (!is.na(n_val) && is.finite(n_val)) {
          args[[paste0("n_at_risk_", lab)]] <- as.integer(n_val)
        }
        if (!is.null(s$lower)) {
          lo <- s$lower[i]
          if (!is.na(lo) && is.finite(lo)) {
            args[[paste0("survival_at_", lab, "_ci_lower")]] <- as.numeric(lo)
          }
        }
        if (!is.null(s$upper)) {
          hi <- s$upper[i]
          if (!is.na(hi) && is.finite(hi)) {
            args[[paste0("survival_at_", lab, "_ci_upper")]] <- as.numeric(hi)
          }
        }
      }
    }
  }

  # Log-rank χ² across groups. ``survdiff`` from survival reports
  # the chi² stat as ``$chisq`` and the df as ``length($n) - 1``
  # (one DF per group beyond the reference). Compute the p-value
  # via ``pchisq`` (chi² distribution upper tail).
  if (!is.null(survdiff)) {
    chi2 <- tryCatch(as.numeric(survdiff$chisq), error = function(e) NULL)
    if (!is.null(chi2) && is.finite(chi2)) {
      args$logrank_chi_squared <- chi2
      df <- length(survdiff$n) - 1L
      if (df >= 1) {
        args$logrank_p_value <- as.numeric(
          pchisq(chi2, df = df, lower.tail = FALSE)
        )
      }
      args$n_groups <- as.integer(length(survdiff$n))
    }
  }

  do.call(sift$result, c(args, list(...)))
}


#' From a `t.test` result, emit a t_test payload.
#'
#' Also prints the native t.test output — R formats this nicely
#' (test name, CI, p-value, sample means) so the researcher sees
#' the conventional view in the raw log panel.
sift$from_t_test <- function(res, ...) {
  print(res)
  # t.test returns different shapes depending on one-sample vs two-sample.
  is_two_sample <- grepl("two sample", res$method, ignore.case = TRUE)
  is_welch      <- grepl("welch",       res$method, ignore.case = TRUE)
  is_paired     <- grepl("paired",      res$method, ignore.case = TRUE)

  subtype <- if (is_welch) "welch"
             else if (is_paired) "paired"
             else if (is_two_sample) "two_sample"
             else "one_sample"

  # res$estimate can be length 1 (one_sample / paired) or length 2.
  ests <- res$estimate
  mean1 <- unname(ests[1])
  mean2 <- if (length(ests) >= 2) unname(ests[2]) else NULL

  # Sample sizes: for t.test, these come from the original data length,
  # which the test object doesn't carry — researcher must pass n1/n2
  # explicitly.
  args <- list(...)
  if (is.null(args$n1)) {
    stop("sift$from_t_test: n1 must be provided via `n1 = length(x)`. ",
         "The t.test object doesn't carry sample sizes.")
  }
  if (subtype %in% c("two_sample", "welch") && is.null(args$n2)) {
    stop("sift$from_t_test: n2 must be provided for ", subtype,
         " tests via `n2 = length(y)`.")
  }

  ci <- res$conf.int
  ci_list <- if (!is.null(ci) && length(ci) == 2) list(ci[1], ci[2]) else NULL

  payload <- c(
    list(
      type = "t_test",
      test_type = subtype,
      mean1 = mean1,
      t_statistic = unname(res$statistic),
      p_value = unname(res$p.value),
      degrees_of_freedom = unname(res$parameter),
      alternative = res$alternative
    ),
    args
  )
  if (!is.null(mean2)) payload$mean2 <- mean2
  if (!is.null(ci_list)) payload$confidence_interval <- ci_list

  do.call(sift$result, payload)
}


#' Emit a descriptive payload for a single variable.
#'
#' Prints a compact one-variable summary to stdout so the researcher
#' sees "variable: n=X, mean=Y, sd=Z, missing=M" in the raw log
#' panel. The caller provides the numbers; we don't recompute.
#'
#' `distinct_count` (optional) is the exact number of unique values.
#' Unlike `mean` / `sd` (floats, which the sanitizer rounds to an
#' N-appropriate number of significant figures), it is an allowed
#' *integer* field and passes through unrounded — so this is the
#' supported way to release an exact unique/cardinality count. The
#' whole-payload `n >= 10` minimum still applies. Compute it from the
#' data and pass it in, e.g.
#'   sift$from_summarize("ein", n = nrow(df), mean = mean(df$ein),
#'                        sd = sd(df$ein),
#'                        missing_count = sum(is.na(df$ein)),
#'                        distinct_count = length(unique(na.omit(df$ein))))
sift$from_summarize <- function(variable, n, mean, sd, missing_count,
                                   distinct_count = NULL,
                                   ...) {
  cat(sprintf(
    "%s: n=%d, mean=%.6g, sd=%.6g, missing=%d",
    variable, as.integer(n), mean, sd, as.integer(missing_count)
  ))
  if (!is.null(distinct_count)) {
    cat(sprintf(", distinct=%d", as.integer(distinct_count)))
  }
  cat("\n")
  # min_value / max_value are no longer accepted: the sanitizer drops
  # them in every payload because nothing in the payload binds the
  # reported values to the named variable's actual column. A future
  # Sift-owned bounds path (request_data extension) is the correct
  # surface; emit aggregates only here.
  args <- list(
    type = "descriptive",
    variable = variable,
    n = as.integer(n),
    mean = mean,
    sd = sd,
    missing_count = as.integer(missing_count)
  )
  # Attach distinct_count only when supplied. `list(x = NULL)` would
  # RETAIN a null element (R keeps named NULLs), so the payload would
  # carry `"distinct_count": null` and the sanitizer would drop it with
  # a noisy "expected int" transformation on every call that omits it.
  # Conditional assignment never creates the key.
  if (!is.null(distinct_count)) {
    args$distinct_count <- as.integer(distinct_count)
  }
  do.call(sift$result, c(args, list(...)))
}


#' Emit a magnitude_table payload (sum or mean of `value_var` by `group_var`).
#'
#' For each group, the helper computes three quantities:
#'   - value: the aggregate (sum or mean)
#'   - n: number of non-missing observations contributing
#'   - max_share: the largest single contributor's share of the total
#'
#' `max_share` is the dominance metric the sanitizer consults to apply
#' the (1, k)-dominance rule. It's required by the schema but NOT
#' forwarded to the frontier model — the sanitizer strips it after
#' consulting. Computed as `max(abs(x)) / sum(abs(x))` on the group's
#' values so mixed signs don't produce meaningless shares.
sift$from_magnitude_table <- function(df, group_var, value_var,
                                          aggregation = "sum", ...) {
  # Native-R preview for the researcher's raw log panel. Use the same
  # aggregation the payload will report so the printed table matches
  # what the sanitized result says.
  tryCatch({
    na_mask <- !is.na(df[[group_var]]) & !is.na(df[[value_var]])
    if (any(na_mask)) {
      agg_fn <- if (aggregation == "sum") sum else mean
      agg_df <- aggregate(
        df[[value_var]][na_mask],
        by = list(df[[group_var]][na_mask]),
        FUN = agg_fn
      )
      names(agg_df) <- c(group_var, paste0(aggregation, "_", value_var))
      cat(sprintf("Magnitude table: %s of %s by %s\n",
                  aggregation, value_var, group_var))
      print(agg_df)
    }
  }, error = function(e) {
    cat(sprintf("(native preview skipped: %s)\n", conditionMessage(e)))
  })

  if (!(aggregation %in% c("sum", "mean"))) {
    stop('sift$from_magnitude_table: aggregation must be "sum" or "mean", ',
         'got ', aggregation)
  }
  if (!(group_var %in% names(df))) {
    stop('sift$from_magnitude_table: group_var ', group_var,
         ' not in data')
  }
  if (!(value_var %in% names(df))) {
    stop('sift$from_magnitude_table: value_var ', value_var,
         ' not in data')
  }

  groups <- unique(df[[group_var]])
  # Drop NA group labels — they don't form a meaningful cell.
  groups <- groups[!is.na(groups)]

  cells <- list()
  for (g in groups) {
    vals <- df[[value_var]][df[[group_var]] == g & !is.na(df[[value_var]])]
    n <- length(vals)
    if (n == 0) {
      # No non-missing observations for this group — emit a zero cell
      # so the sanitizer suppresses it on n grounds.
      cells[[as.character(g)]] <- list(value = 0, n = 0L, max_share = 0)
      next
    }
    total_abs <- sum(abs(vals))
    if (aggregation == "sum") {
      value <- sum(vals)
    } else {
      value <- mean(vals)
    }
    # Guard against all-zero groups: max_share undefined; set to 0
    # (no contributor dominates because there's no magnitude).
    max_share <- if (total_abs == 0) 0 else max(abs(vals)) / total_abs
    cells[[as.character(g)]] <- list(
      value = as.numeric(value),
      n = as.integer(n),
      max_share = as.numeric(max_share)
    )
  }

  # Reject `...` arguments that would override fields the helper
  # computes from raw data. Without this guard, a caller could pass
  # `cells=list(...)` (or row_variable=..., etc.) and the
  # `c(list(computed), list(...))` concatenation below would emit
  # duplicate keys whose JSON serialization a downstream parser
  # resolves by last-occurrence, replacing the helper's computation.
  # The `_via_helper` marker (stamped at the end) would then
  # authenticate attacker-supplied values, and the sanitizer (which
  # trusts the marker to skip recomputing `max_share`) would let a
  # forged max_share=0 bypass the dominance gate.
  extras <- list(...)
  reserved <- c("type", "row_variable", "value_variable",
                "aggregation", "cells", "_via_helper")
  forbidden <- intersect(names(extras), reserved)
  if (length(forbidden) > 0) {
    stop("sift$from_magnitude_table: cannot override helper-",
         "computed fields via extra arguments: ",
         paste(sort(forbidden), collapse = ", "),
         ". These are computed from the data and bound to the ",
         "_via_helper provenance marker.")
  }

  # Bypass sift$result and write directly so the helper-provenance
  # marker (`_via_helper`) survives. The generic sift$result strips
  # `_via_helper` from caller-passed kwargs so a script can't forge
  # this marker through the public API. The sanitizer requires the
  # marker for magnitude_table because cell-level `max_share` is
  # consulted-only and stripped; without proof that max_share came
  # from raw-data computation a malicious script could publish a
  # dominance-violating value with `max_share=0` and skip the gate.
  payload <- c(
    list(
      type = "magnitude_table",
      row_variable = as.character(group_var),
      value_variable = as.character(value_var),
      aggregation = aggregation,
      cells = cells
    ),
    extras
  )
  payload[["_via_helper"]] <- "from_magnitude_table"
  sift$.write_result(payload)
}


#' From a 2D table / matrix, emit a crosstab payload.
#'
#' Input can be:
#'   - a 2D `table` object (from `table(x, y)`)
#'   - a numeric matrix with `dimnames`
#'
#' The helper copies over the dimension names as row/column variable
#' labels if present, falling back to `"row"` / `"col"` otherwise.
#' Crosstabs emit cells only — never margins. The sanitizer will drop
#' any margin-ish field by name if one slips in here.
sift$from_crosstab <- function(tbl, row_variable = NULL, col_variable = NULL,
                                  missing_count = 0L, ...) {
  if (!(is.table(tbl) || is.matrix(tbl))) {
    stop("sift$from_crosstab: expected a 2D table or matrix, got ",
         class(tbl)[1])
  }
  # Native preview for the raw log panel.
  print(tbl)
  if (length(dim(tbl)) != 2) {
    stop("sift$from_crosstab: expected a 2D structure, got ",
         length(dim(tbl)), " dimension(s). Use sift$from_table for 1D.")
  }

  rows <- dimnames(tbl)[[1]]
  cols <- dimnames(tbl)[[2]]
  if (is.null(rows)) rows <- paste0("row", seq_len(nrow(tbl)))
  if (is.null(cols)) cols <- paste0("col", seq_len(ncol(tbl)))

  dn <- names(dimnames(tbl))
  if (is.null(row_variable)) {
    row_variable <- if (!is.null(dn) && nzchar(dn[1])) dn[1] else "row"
  }
  if (is.null(col_variable)) {
    col_variable <- if (!is.null(dn) && length(dn) > 1 && nzchar(dn[2])) dn[2] else "col"
  }

  # Build the nested dict counts[row][col] = integer.
  counts <- list()
  for (r in rows) {
    inner <- list()
    for (c in cols) {
      inner[[c]] <- as.integer(tbl[r, c])
    }
    counts[[r]] <- inner
  }

  sift$result(
    type = "crosstab",
    row_variable = as.character(row_variable),
    col_variable = as.character(col_variable),
    counts = counts,
    missing_count = as.integer(missing_count),
    ...
  )
}


#' Emit a frequency_table payload.
sift$from_table <- function(variable, counts, n = NULL, missing_count = 0L, ...) {
  # `counts` should be a named integer vector / list. Normalize to a
  # named list of ints.
  if (is.table(counts)) {
    # Native preview — R formats a 1D table nicely as a two-row block.
    cat(sprintf("Frequency table: %s\n", variable))
    print(counts)
    d <- as.list(as.integer(counts))
    names(d) <- names(counts)
    counts <- d
  } else if (is.list(counts) || !is.null(names(counts))) {
    cat(sprintf("Frequency table: %s\n", variable))
    for (lvl in names(counts)) {
      cat(sprintf("  %s: %s\n", lvl, counts[[lvl]]))
    }
  }
  if (is.null(n)) {
    n <- sum(unlist(counts)) + missing_count
  }
  sift$result(
    type = "frequency_table",
    variable = variable,
    counts = counts,
    n = as.integer(n),
    missing_count = as.integer(missing_count),
    ...
  )
}


#' Pairwise correlation matrix from a data frame.
#'
#' By default correlates every numeric column; pass `variables` to
#' restrict to a named subset. `method` is `"pearson"` (default),
#' `"spearman"`, or `"kendall"`. Sample size N is the number of
#' COMPLETE rows over the chosen variables — emitting pairwise N
#' would let off-diagonals draw on different samples and make joint
#' inference dishonest.
#'
#' Also prints the matrix to stdout for the researcher's raw log.
sift$from_correlation <- function(df, variables = NULL,
                                   method = "pearson", ...) {
  valid_methods <- c("pearson", "spearman", "kendall")
  if (!(method %in% valid_methods)) {
    stop('sift$from_correlation: method must be one of ',
         paste(shQuote(valid_methods), collapse = ", "), ", got ",
         shQuote(method))
  }
  if (!is.data.frame(df)) {
    stop("sift$from_correlation: expected a data.frame, got ",
         class(df)[1])
  }
  if (is.null(variables)) {
    # Pick numeric / logical columns. Match the Python helper's
    # default: skip character / factor / Date.
    is_num_col <- vapply(df, function(col) {
      is.numeric(col) || is.logical(col)
    }, logical(1))
    variables <- names(df)[is_num_col]
  }
  if (length(variables) == 0) {
    stop("sift$from_correlation: no numeric columns found and no ",
         "`variables` provided")
  }
  missing_cols <- setdiff(variables, names(df))
  if (length(missing_cols) > 0) {
    stop("sift$from_correlation: variables not in df: ",
         paste(missing_cols, collapse = ", "))
  }
  sub <- df[, variables, drop = FALSE]
  complete <- stats::complete.cases(sub)
  n_complete <- sum(complete)
  missing_count <- nrow(sub) - n_complete
  cm <- stats::cor(sub[complete, , drop = FALSE], method = method)
  print(cm)
  # Build nested list-of-lists in declared variable order so the
  # JSON output is symmetric and stable regardless of `cor()`'s
  # internal column ordering.
  correlations <- list()
  for (rv in variables) {
    inner <- list()
    for (cv in variables) {
      v <- cm[rv, cv]
      if (is.finite(v)) inner[[cv]] <- as.numeric(v)
    }
    if (length(inner) > 0) correlations[[rv]] <- inner
  }
  sift$result(
    type = "correlation_matrix",
    n = as.integer(n_complete),
    variables = as.list(variables),
    method = method,
    correlations = correlations,
    missing_count = as.integer(missing_count),
    ...
  )
}


# ---------------------------------------------------------------------------
# Plot helpers — model-output visualizations only
# ---------------------------------------------------------------------------
#
# Plots produced via these helpers are surfaced to the model on the
# next turn as image attachments. RAW-DATA plots (a histogram of an
# observed variable, a scatter of all rows, a density of a column)
# are NOT covered here on purpose — they would expose the data
# itself, which is the privacy line Sift is built to keep.
#
# What IS covered:
#   - Residual diagnostics (plot_residuals): residuals vs fitted,
#     QQ plot, scale-location, histogram of residuals. All
#     functions of model output, not raw observations.
#   - Interaction / predicted-value plots (plot_interaction):
#     model's predicted response across a variable's range, holding
#     other predictors at their means. Shows model behavior, not
#     data points.
#
# Future helpers (coefficient forest plots, marginal effects)
# follow the same principle: things the model produced from the fit,
# not visualizations of the rows.
#
# Mechanism: helpers write a PNG into <run_dir>/_sift_plots/ and
# append a JSONL entry to <run_dir>/_sift_plots/manifest.jsonl. The
# bridge reads ONLY the manifest — anything else dropped into the
# directory by ggsave / plain plot() / etc. is invisible to the
# model regardless. That's the allowlist; mirrors how the sanitizer
# works for textual results.

sift$.plots_dir <- function() {
  result_path <- Sys.getenv("SIFT_RESULT_PATH")
  if (!nzchar(result_path)) return(NULL)
  d <- file.path(dirname(result_path), "_sift_plots")
  dir.create(d, showWarnings = FALSE, recursive = TRUE)
  d
}

# Return `base` if no file by that name exists in `d`; otherwise
# append _2, _3, ... before the extension. Without this, calling
# `plot_coefficients(fit1)` then `plot_coefficients(fit2)` in one
# script would (a) overwrite `coefficients.png` with fit2's image
# and (b) append a second manifest row pointing at the SAME file,
# so the model sees two "different" plots that are both fit2.
# `plot_interaction` already side-steps this by suffixing the
# variable name into the filename; the others need a counter.
sift$.unique_plot_name <- function(d, base) {
  if (!file.exists(file.path(d, base))) return(base)
  parts <- tools::file_path_sans_ext(base)
  ext <- tools::file_ext(base)
  i <- 2L
  repeat {
    candidate <- if (nzchar(ext)) paste0(parts, "_", i, ".", ext) else paste0(parts, "_", i)
    if (!file.exists(file.path(d, candidate))) return(candidate)
    i <- i + 1L
  }
}

# Scrub a categorical tick label that will be rendered into a
# model-visible plot image. Mirrors the Python helper's use of
# ``safe_text`` on raw category names: strip C0/C1 control chars,
# bidi overrides, zero-width chars, and BOM/word-joiner; cap to
# 24 chars; fall back to "[redacted]" if the scrub leaves nothing.
# Plot images bypass the JSON/text path's safety gate, so any raw
# string that reaches an axis label has to be cleaned here.
sift$.safe_tick_label <- function(v) {
  s <- tryCatch(as.character(v), error = function(e) "")
  if (length(s) != 1 || is.na(s)) s <- ""
  # C0 controls (U+0001-U+001F) and DEL (U+007F). R strings cannot
  # carry a literal U+0000 (the C-string representation forbids
  # it), so the NUL byte never reaches this function and doesn't
  # need a strip pass — and ``\x00`` in a string literal is a
  # parse error anyway.
  s <- gsub("[\x01-\x1f\x7f]", "", s, perl = TRUE, useBytes = FALSE)
  # C1 controls (U+0080-U+009F).
  s <- gsub("[-]", "", s, perl = TRUE, useBytes = FALSE)
  # Bidi overrides + isolates and zero-width chars: LRM/RLM,
  # LRE/RLE/PDF, LRO/RLO, LRI/RLI/FSI/PDI, ZWSP/ZWNJ/ZWJ, word
  # joiner, BOM, invisible-times / invisible-separator.
  s <- gsub(
    paste0(
      "[\u200b\u200c\u200d\u200e\u200f",
      "\u202a\u202b\u202c\u202d\u202e",
      "\u2060\u2062\u2063",
      "\u2066\u2067\u2068\u2069",
      "\ufeff]"
    ),
    "", s, perl = TRUE, useBytes = FALSE,
  )
  # Cap at 24 chars (matches the Python helper's per-tick limit).
  if (nchar(s) > 24) s <- paste0(substr(s, 1, 23), "…")
  if (!nzchar(s)) s <- "[redacted]"
  s
}


sift$.append_plot_manifest <- function(file, kind, label) {
  d <- sift$.plots_dir()
  if (is.null(d)) return(invisible(NULL))
  # Stamp every entry with the per-run token. The executor validates
  # this field after the script finishes and drops any entry whose
  # token is missing or wrong; that strips manifest rows a script
  # could otherwise have appended directly (saving a raw-data plot
  # under _sift_plots/ and labeling it "coefficients" to slip past
  # the disclosure-control allowlist for vision attachment). Same
  # posture as the result-payload _token field.
  entry <- list(file = file, kind = kind)
  if (!is.null(label) && nzchar(label)) entry$label <- label
  entry[["_token"]] <- sift$.run_token
  line <- sift$.to_json(entry)
  con <- file(file.path(d, "manifest.jsonl"), open = "a", encoding = "UTF-8")
  on.exit(close(con), add = TRUE)
  writeLines(line, con)
  invisible(NULL)
}

# Record a structured plot-helper failure so submit_script can surface
# it in the tool result the MODEL receives. Without this, helper
# failures only land in stderr and the model says "thumbnail should be
# visible above" while the researcher sees nothing.
sift$.append_plot_helper_error <- function(helper, message) {
  d <- sift$.plots_dir()
  if (is.null(d)) return(invisible(NULL))
  msg <- as.character(message)
  fix <- NULL
  lower <- tolower(msg)
  if (grepl("haven", lower) || grepl("could not find function .*read_dta", lower)) {
    fix <- "install.packages(\"haven\")"
  } else if (grepl("ggplot2", lower) || grepl("could not find function .*ggplot", lower)) {
    fix <- "install.packages(\"ggplot2\")"
  }
  entry <- list(helper = helper, error = "R error", message = msg)
  if (!is.null(fix)) entry$fix <- fix
  line <- sift$.to_json(entry)
  con <- file(file.path(d, "helper_errors.jsonl"),
              open = "a", encoding = "UTF-8")
  on.exit(close(con), add = TRUE)
  writeLines(line, con)
  invisible(NULL)
}

#' Write the four standard residual-diagnostic panels for an `lm`
#' (or anything `plot()` accepts as a fit) and register them with
#' the manifest so the model can see them on the next turn.
#'
#' Failures inside the helper are surfaced via `message()` (visible
#' to the researcher in the raw-log panel) but never raise — a
#' broken plot helper must not break the analysis script around it.
sift$plot_residuals <- function(model, label = NULL) {
  d <- sift$.plots_dir()
  if (is.null(d)) return(invisible(NULL))
  fname <- sift$.unique_plot_name(d, "residuals.png")
  res <- tryCatch({
    grDevices::png(file.path(d, fname),
                   width = 900, height = 700, res = 110)
    on.exit(grDevices::dev.off(), add = TRUE)
    op <- graphics::par(mfrow = c(2, 2))
    on.exit(graphics::par(op), add = TRUE)
    plot(model)
    TRUE
  }, error = function(e) {
    message("sift$plot_residuals failed: ", conditionMessage(e))
    sift$.append_plot_helper_error("plot_residuals", conditionMessage(e))
    FALSE
  })
  if (isTRUE(res)) {
    sift$.append_plot_manifest(
      fname, "residuals",
      if (is.null(label)) "Residual diagnostics" else label
    )
  }
  invisible(NULL)
}

#' Predicted-response curve across a single predictor, with other
#' predictors held at their means (numeric) or first level (factor).
#' Confidence bands are 1.96 * SE of the predicted mean.
#'
#' Optional ``xlab`` / ``ylab`` / ``title`` override the defaults
#' (which fall back to the variable name and "Predicted response").
#' If ``ggplot2`` is available the helper uses it for a cleaner
#' filled-ribbon CI; otherwise it falls back to base graphics.
#'
#' This is a model-output plot — the predicted line and bands come
#' from the fit's variance / coefficient estimates, not from the
#' data rows themselves. Same privacy posture as plot_residuals.
sift$plot_interaction <- function(model, var, label = NULL,
                                   xlab = NULL, ylab = NULL,
                                   title = NULL) {
  d <- sift$.plots_dir()
  if (is.null(d)) return(invisible(NULL))
  if (!is.character(var) || length(var) != 1) {
    message("sift$plot_interaction: `var` must be a single name string")
    return(invisible(NULL))
  }
  fname <- paste0("interaction_", make.names(var), ".png")
  xtitle <- if (is.null(xlab)) var else xlab
  ytitle <- if (is.null(ylab)) "Predicted response" else ylab
  ptitle <- if (is.null(title)) paste0("Predicted response by ", var) else title
  res <- tryCatch({
    md <- model$model
    if (is.null(md) || !var %in% names(md)) {
      stop("variable '", var, "' is not in the model frame")
    }
    template <- md[1, , drop = FALSE]
    for (col in names(template)) {
      v <- md[[col]]
      if (is.numeric(v))      template[[col]] <- mean(v, na.rm = TRUE)
      else if (is.factor(v))  template[[col]] <- levels(v)[1]
      else                    template[[col]] <- v[1]
    }
    xs <- md[[var]]
    # Disclosure-control: the rendered PNG is allowlisted for model
    # vision (kind="interaction"), so anything legible on the x-axis
    # crosses the SDC boundary. The previous version used min/max
    # (numeric) and full level lists (factor/categorical), which
    # surfaced raw extrema and rare-level identities the JSON
    # sanitizer would have refused. Match the Python helper's
    # disclosure-safe grid:
    #   - numeric: mean ± 2*sd; equivalent to the descriptive
    #     sanitizer's already-allowed mean+sd disclosure. Refuse
    #     when N is below threshold or variance is zero.
    #   - factor / categorical: drop levels with count below
    #     threshold; refuse when none remain.
    .CELL_SUPP_THRESH <- 10L
    .CAT_LEVEL_CAP <- 20L
    .suppression_note <- NULL
    if (is.numeric(xs)) {
      .clean <- xs[!is.na(xs)]
      if (length(.clean) < .CELL_SUPP_THRESH) {
        stop("variable '", var, "' has fewer than ", .CELL_SUPP_THRESH,
             " non-missing observations; below the disclosure threshold")
      }
      .mu <- mean(.clean)
      .sd <- stats::sd(.clean)
      if (!is.finite(.sd) || .sd <= 0) {
        stop("variable '", var,
             "' has zero variance — interaction plot would expose ",
             "the constant value")
      }
      grid <- seq(.mu - 2 * .sd, .mu + 2 * .sd, length.out = 100)
    } else {
      .clean <- xs[!is.na(xs)]
      .counts <- table(.clean)
      .visible <- .counts[.counts >= .CELL_SUPP_THRESH]
      if (length(.visible) == 0) {
        stop("variable '", var,
             "': no level meets the disclosure threshold (n >= ",
             .CELL_SUPP_THRESH, "); refusing to plot")
      }
      # Sort by frequency desc, keep top-K for readability.
      .visible <- sort(.visible, decreasing = TRUE)
      if (length(.visible) > .CAT_LEVEL_CAP) {
        .visible <- .visible[seq_len(.CAT_LEVEL_CAP)]
      }
      .keep_levels <- names(.visible)
      .dropped <- as.integer(sum(.counts < .CELL_SUPP_THRESH))
      if (.dropped > 0) {
        .suppression_note <- paste0(
          .dropped, " rare level(s) with count < ",
          .CELL_SUPP_THRESH, " suppressed"
        )
      }
      if (is.factor(xs)) {
        grid <- factor(.keep_levels, levels = levels(xs))
      } else {
        grid <- .keep_levels
      }
    }
    new <- template[rep(1, length(grid)), , drop = FALSE]
    new[[var]] <- grid
    pr <- stats::predict(model, newdata = new, se.fit = TRUE)
    lo <- pr$fit - 1.96 * pr$se.fit
    hi <- pr$fit + 1.96 * pr$se.fit

    have_gg <- requireNamespace("ggplot2", quietly = TRUE)
    if (have_gg && is.numeric(grid)) {
      # ggplot2 path: filled ribbon CI, theme_minimal, decent
      # default font sizes. Continuous-x only; for factor x we
      # fall through to a base barplot below.
      df <- data.frame(x = grid, fit = pr$fit, lo = lo, hi = hi)
      p <- ggplot2::ggplot(df, ggplot2::aes(x = x, y = fit)) +
        ggplot2::geom_ribbon(
          ggplot2::aes(ymin = lo, ymax = hi),
          fill = "#4C78A8", alpha = 0.20
        ) +
        ggplot2::geom_line(color = "#1F4E79", linewidth = 1) +
        ggplot2::labs(title = ptitle, x = xtitle, y = ytitle) +
        ggplot2::theme_minimal(base_size = 12) +
        ggplot2::theme(
          plot.title = ggplot2::element_text(face = "bold"),
          panel.grid.minor = ggplot2::element_blank()
        )
      ggplot2::ggsave(file.path(d, fname), plot = p,
                      width = 8, height = 5, dpi = 110)
    } else {
      # Base graphics fallback: still cleaner than the previous
      # default — filled polygon for the CI band, colored line,
      # margin-aware labels.
      grDevices::png(file.path(d, fname),
                     width = 900, height = 560, res = 110)
      on.exit(grDevices::dev.off(), add = TRUE)
      op <- graphics::par(mar = c(4.5, 4.5, 2.5, 1.5))
      on.exit(graphics::par(op), add = TRUE)
      if (is.numeric(grid)) {
        ylim <- range(c(lo, hi), na.rm = TRUE)
        plot(grid, pr$fit, type = "n",
             xlab = xtitle, ylab = ytitle, main = ptitle,
             ylim = ylim)
        graphics::polygon(
          c(grid, rev(grid)), c(lo, rev(hi)),
          col = grDevices::adjustcolor("#4C78A8", alpha.f = 0.20),
          border = NA
        )
        graphics::lines(grid, pr$fit, lwd = 2, col = "#1F4E79")
      } else {
        # Run tick labels through the same text-safety primitive
        # the Python helper applies. ``grid`` values come straight
        # from levels in the raw data, so without scrubbing, a
        # frequent category name with embedded control chars,
        # bidi overrides, zero-width chars, or prompt-like text
        # would render straight into the model-visible image,
        # bypassing the JSON/text safety gate. Empty after scrub
        # becomes "[redacted]" so the bar stays identifiable.
        labels <- vapply(grid, function(v) {
          sift$.safe_tick_label(v)
        }, character(1))
        graphics::barplot(
          pr$fit, names.arg = labels,
          ylab = ytitle, main = ptitle, col = "#4C78A8",
          border = "#1F4E79"
        )
      }
    }
    TRUE
  }, error = function(e) {
    message("sift$plot_interaction failed: ", conditionMessage(e))
    sift$.append_plot_helper_error("plot_interaction", conditionMessage(e))
    FALSE
  })
  if (isTRUE(res)) {
    sift$.append_plot_manifest(
      fname, "interaction",
      if (is.null(label)) paste0("Predicted response by ", var) else label
    )
  }
  invisible(NULL)
}


#' Forest plot of coefficient estimates with 95% CIs.
#'
#' Operates only on `coef(model)` and `confint(model)` — pure
#' functions of model output, never the raw data. The helper IS
#' the gate; there is no escape-hatch path that accepts an
#' arbitrary file. That would let a histogram of raw rows pose
#' as a coefficient plot — bypassing the privacy line the entire
#' system rests on.
sift$plot_coefficients <- function(model, label = NULL) {
  d <- sift$.plots_dir()
  if (is.null(d)) return(invisible(NULL))
  fname <- sift$.unique_plot_name(d, "coefficients.png")
  res <- tryCatch({
    cf <- coef(model)
    ci <- stats::confint(model)
    nms <- names(cf)
    if (is.null(nms)) nms <- as.character(seq_along(cf))
    # Drop the intercept by default — almost never on the same
    # scale as predictors. Researchers who want it can call
    # plot_coefficients on a fit without an intercept term.
    keep <- !(tolower(nms) %in% c("(intercept)", "intercept", "_cons"))
    if (!any(keep)) {
      stop("nothing to plot after dropping the intercept")
    }
    cf <- cf[keep]
    ci <- ci[keep, , drop = FALSE]
    nms <- nms[keep]
    # Order top-to-bottom matching the names vector — y axis goes
    # downward so we plot index 1 at the top.
    y <- seq_along(cf)
    grDevices::png(file.path(d, fname),
                   width = 900,
                   height = max(220, 60 * length(cf) + 120),
                   res = 110)
    on.exit(grDevices::dev.off(), add = TRUE)
    op <- graphics::par(mar = c(4.5, 7, 2, 2))
    on.exit(graphics::par(op), add = TRUE)
    xlim <- range(c(ci[, 1], ci[, 2]), finite = TRUE)
    plot(NA, xlim = xlim, ylim = c(length(cf) + 0.5, 0.5),
         yaxt = "n", xlab = "Coefficient (95% CI)", ylab = "",
         main = "Coefficients")
    graphics::abline(v = 0, lty = 2, col = "gray60")
    graphics::segments(ci[, 1], y, ci[, 2], y, lwd = 2, col = "#4C78A8")
    graphics::points(cf, y, pch = 16, cex = 1.4, col = "#4C78A8")
    graphics::axis(2, at = y, labels = nms, las = 1)
    TRUE
  }, error = function(e) {
    message("sift$plot_coefficients failed: ", conditionMessage(e))
    sift$.append_plot_helper_error("plot_coefficients", conditionMessage(e))
    FALSE
  })
  if (isTRUE(res)) {
    sift$.append_plot_manifest(
      fname, "coefficients",
      if (is.null(label)) "Coefficient estimates with 95% CIs" else label
    )
  }
  invisible(NULL)
}


#' Forest plot comparing one coefficient across multiple model fits.
#'
#' Use case: "Female gap, before vs after controls" — fit two
#' models, plot their named coefficient with CIs side-by-side.
#' ``models`` is a NAMED list of fits (the names become y-axis
#' labels). ``coef`` is the coefficient to extract from each fit
#' via ``coef(m)`` + ``vcov(m)``. SEs come from the diagonal of
#' the variance-covariance matrix.
#'
#' Without this helper, comparison plots forced cross-language
#' workflows: the model would extract estimates from R / Stata,
#' switch to Python or back to R, and hand-roll a forest plot —
#' often three attempts before one landed.
sift$plot_estimate_comparison <- function(models, coef, label = NULL) {
  d <- sift$.plots_dir()
  if (is.null(d)) return(invisible(NULL))
  fname <- sift$.unique_plot_name(d, "estimate_comparison.png")
  res <- tryCatch({
    if (!is.list(models) || length(models) < 2) {
      stop("`models` must be a list of at least 2 model fits")
    }
    if (!is.character(coef) || length(coef) != 1) {
      stop("`coef` must be a single coefficient name")
    }
    nms <- names(models)
    if (is.null(nms) || any(!nzchar(nms))) {
      nms <- paste0("Model ", seq_along(models))
    }

    n_models <- length(models)
    ests <- numeric(n_models)
    ses  <- numeric(n_models)
    for (i in seq_len(n_models)) {
      m <- models[[i]]
      cf <- stats::coef(m)
      if (!coef %in% names(cf)) {
        stop("coefficient '", coef, "' not in model: ", nms[i])
      }
      ests[i] <- cf[[coef]]
      vc <- stats::vcov(m)
      if (!coef %in% rownames(vc)) {
        stop("coefficient '", coef, "' not in vcov of model: ", nms[i])
      }
      ses[i] <- sqrt(vc[coef, coef])
    }
    los <- ests - 1.96 * ses
    his <- ests + 1.96 * ses

    grDevices::png(file.path(d, fname),
                   width = 900,
                   height = max(220, 80 * n_models + 100),
                   res = 110)
    on.exit(grDevices::dev.off(), add = TRUE)
    op <- graphics::par(mar = c(4.5, 8, 2.5, 2))
    on.exit(graphics::par(op), add = TRUE)
    xlim <- range(c(los, his), finite = TRUE)
    y <- seq_len(n_models)
    plot(NA, xlim = xlim, ylim = c(n_models + 0.5, 0.5),
         yaxt = "n", xlab = paste0(coef, " (95% CI)"), ylab = "",
         main = paste0("Estimate comparison: ", coef))
    graphics::abline(v = 0, lty = 2, col = "gray60")
    graphics::segments(los, y, his, y, lwd = 2, col = "#4C78A8")
    graphics::points(ests, y, pch = 16, cex = 1.4, col = "#4C78A8")
    graphics::axis(2, at = y, labels = nms, las = 1)
    TRUE
  }, error = function(e) {
    message("sift$plot_estimate_comparison failed: ", conditionMessage(e))
    sift$.append_plot_helper_error(
      "plot_estimate_comparison", conditionMessage(e)
    )
    FALSE
  })
  if (isTRUE(res)) {
    sift$.append_plot_manifest(
      fname, "estimate_comparison",
      if (is.null(label)) paste0("Estimate comparison: ", coef) else label
    )
  }
  invisible(NULL)
}


# ---------------------------------------------------------------------------
# Smoke test — skipped automatically in production because the env var is
# expected to be set by the executor. Researchers running `source("sift.R")`
# manually (e.g., for exploration) can ignore this file — calling result()
# without the env var raises a clear error.
# ---------------------------------------------------------------------------

invisible(NULL)
