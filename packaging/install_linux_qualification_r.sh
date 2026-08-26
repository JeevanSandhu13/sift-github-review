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
    build-essential gfortran libcurl4-openssl-dev libssl-dev libxml2-dev r-base

SIFT_R_SITE_LIBRARY="/usr/local/lib/R/site-library"
install -d --mode=0755 "${SIFT_R_SITE_LIBRARY}"
R_LIBS_SITE="${SIFT_R_SITE_LIBRARY}" R_LIBS_USER="${SIFT_R_SITE_LIBRARY}" Rscript --vanilla - <<'RSCRIPT'
# Use a dated Posit Package Manager snapshot so CI receives reproducible,
# precompiled Ubuntu 22.04 binaries. Compiling the same dependency graph from
# CRAN source is both slower and liable to exceed a hosted runner's memory.
site_library <- "/usr/local/lib/R/site-library"
# GitHub's runner can combine current R with Ubuntu packages built for an older
# R ABI. Resolve and install the complete qualification graph in the local site
# library, considering only that library and R's matching base library.
.libPaths(c(site_library, .Library))
stopifnot(!normalizePath("/usr/lib/R/site-library") %in% normalizePath(.libPaths()))
options(
  repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/jammy/2026-08-25"),
  HTTPUserAgent = sprintf(
    "R/%s R (%s)",
    getRversion(),
    paste(getRversion(), R.version["platform"], R.version["arch"], R.version["os"])
  ),
  timeout = 900
)
required <- c(
  lavaan = "0.7-2",
  lme4 = "2.0-6",
  psych = "2.6.5",
  pscl = "1.5.9",
  survey = "4.5",
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
  stopifnot(startsWith(normalizePath(find.package(package)), normalizePath(site_library)))
}
RSCRIPT

echo "Pinned R reference environment is ready for scientific qualification."
