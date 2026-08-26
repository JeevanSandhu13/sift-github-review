#!/usr/bin/env bash
set -euo pipefail

# Maintainer/CI-only reference environment for Sift's scientific differential
# qualification. R is not part of the shipped desktop runtime.

[[ "$(uname -s)" == "Linux" ]] || { echo "This helper supports Linux qualification hosts only." >&2; exit 1; }
[[ "$(id -u)" == "0" ]] || { echo "Run this qualification helper as root." >&2; exit 1; }
[[ -r /etc/os-release ]] || { echo "Cannot identify the operating system." >&2; exit 1; }
# shellcheck disable=SC1091
. /etc/os-release
[[ "${ID:-}" == "ubuntu" ]] || { echo "This qualification recipe is pinned for Ubuntu." >&2; exit 1; }

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install --yes \
    build-essential gfortran libcurl4-openssl-dev libssl-dev libxml2-dev \
    r-base r-cran-lavaan r-cran-lme4 r-cran-psych \
    r-cran-pscl r-cran-remotes r-cran-survey

Rscript --vanilla - <<'RSCRIPT'
options(repos = c(CRAN = "https://cloud.r-project.org"), timeout = 900)
required <- c(
  Rcpp = "1.1.2",
  RcppEigen = "0.3.4.0.2",
  data.table = "1.18.6.1",
  did = "2.5.1",
  rdrobust = "4.0.0",
  poLCA = "1.6.0.2",
  fixest = "0.14.2",
  marginaleffects = "0.32.0"
)
for (package in names(required)) {
  installed <- requireNamespace(package, quietly = TRUE)
  current <- if (installed) as.character(packageVersion(package)) else ""
  if (!identical(current, unname(required[[package]]))) {
    remotes::install_version(
      package, version = unname(required[[package]]),
      repos = getOption("repos"), upgrade = "never", dependencies = NA
    )
  }
}
for (package in names(required)) {
  stopifnot(requireNamespace(package, quietly = TRUE))
  stopifnot(identical(as.character(packageVersion(package)), unname(required[[package]])))
}
RSCRIPT

echo "Pinned R reference environment is ready for scientific qualification."
