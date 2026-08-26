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
[[ "${VERSION_CODENAME:-}" == "jammy" ]] || { echo "This qualification recipe is pinned for Ubuntu 22.04 (Jammy)." >&2; exit 1; }

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install --yes \
    build-essential gfortran libcurl4-openssl-dev libssl-dev libxml2-dev \
    r-base r-cran-lavaan r-cran-lme4 r-cran-psych \
    r-cran-pscl r-cran-remotes r-cran-survey

Rscript --vanilla - <<'RSCRIPT'
# Use a dated Posit Package Manager snapshot so CI receives reproducible,
# precompiled Ubuntu 22.04 binaries. Compiling the same dependency graph from
# CRAN source is both slower and liable to exceed a hosted runner's memory.
options(
  repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/jammy/2026-08-25"),
  timeout = 900
)
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
current_version <- function(package) {
  if (requireNamespace(package, quietly = TRUE)) {
    as.character(packageVersion(package))
  } else {
    ""
  }
}
needed <- names(required)[vapply(
  names(required),
  function(package) !identical(current_version(package), unname(required[[package]])),
  logical(1)
)]
if (length(needed)) {
  install.packages(needed, dependencies = NA)
}
for (package in names(required)) {
  stopifnot(requireNamespace(package, quietly = TRUE))
  stopifnot(identical(as.character(packageVersion(package)), unname(required[[package]])))
}
RSCRIPT

echo "Pinned R reference environment is ready for scientific qualification."
