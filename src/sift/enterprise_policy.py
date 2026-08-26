"""Admin-authored restrictions layered over researcher policy.

Session policy records researcher consent. Enterprise policy provides a
separate, system-managed floor that ordinary sessions cannot loosen.

- **Location.** The enterprise policy file lives outside the researcher's
  project directory entirely — resolved via ``SIFT_ENTERPRISE_POLICY``
  (an absolute path, for deployment flexibility and for tests) or a
  fixed OS-level config path (``/etc/sift/enterprise_policy.yaml`` on
  Linux, ``/Library/Application Support/Sift/enterprise_policy.yaml``
  on macOS, and ``%ProgramData%\\Sift\\enterprise_policy.yaml`` on
  Windows). A researcher with ordinary user permissions cannot write
  either location without administrator/root access. Sift relies on and
  verifies the operating system's ownership and permission boundary.

- **Direction.** Every combinator in this module takes a value already
  computed from the session policy and returns one that is AT LEAST
  as restrictive — never looser. There is no code path anywhere in
  this module that can widen a ceiling, remove a ban, or lower a
  suppression threshold. This is the same posture ``policy.py`` itself
  documents for privacy profiles vs. ``max_depth``: whichever value is
  stricter always wins, and that is enforced by construction (`min`/
  `max`/set-union over the restrictive direction), not by review
  discipline alone.

- **Absence vs. malformation, same split as ``policy.py``.** No file
  found at any resolved location -> ``load_enterprise_policy()``
  returns ``None`` and every combinator in this module is a no-op —
  an install with no enterprise config imposes no additional
  restriction, which is correct: "no organisation opted in" is not
  itself a security signal the way a broken researcher policy file
  is. A file IS found but is unreadable / unparseable / wrong shape ->
  fail closed to ``_FAIL_CLOSED_ENTERPRISE_POLICY``, the strictest
  representable configuration (schema names only, no DP, export
  approval required, high suppression floors). An admin who deployed
  a config file expressed an intent; a corrupted deployment must not
  silently revert to "no enterprise policy" any more than a corrupted
  ``policy.json`` may silently revert to the permissive default.

- **Scope.** Four policy axes: a global "never expose these fields" list layered on top
  of every dataset's own ``banned_variables``; floors on the
  suppression/min-N thresholds the sanitizer's ``SDCConfig`` uses
  ("aggregate_only" is implemented as "raise every N-floor",
  which is what actually forcing results toward aggregates means in
  this codebase's SDC machinery); a ceiling on ``dp_epsilon`` (or a
  blanket kill switch for differential privacy); and an
  export-approval workflow.

- **Export approval is a local, file-based workflow primitive, stated
  honestly.** Sift is a single-machine tool with no server, no
  network identity, and no built-in notion of "who is the approver."
  When ``export_approval_required`` is set, an export attempt writes a
  timestamped, uniquely-IDed REQUEST record under
  ``.sift/export_requests/`` instead of the export file, and returns
  that ID. Approving is a separate, explicit call
  (``approve_export``) that writes an APPROVAL record an approver with
  their own access to the machine (or a process built around this
  primitive — a governance office with SSH access, a script run by
  someone other than the researcher) can make after review. This
  module makes no claim to authenticate who calls ``approve_export`` —
  doing so honestly would require an identity system this project
  does not have. What it DOES guarantee: the export file itself is
  never written until an approval record exists, and every export
  bridge method in ``ui.py`` is gated the same way, so there is no
  path that produces the artifact without a recorded approval.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from sift.policy import PRIVACY_PROFILES, VALID_DEPTHS
from sift.sanitizer import SDCConfig
from sift.text_safety import banned_key

# Depth/profile rank tables, recomputed locally from policy.py's own
# public ordered tuples (``VALID_DEPTHS``, ``PRIVACY_PROFILES``) —
# read-only introspection of that boundary file's public contract;
# nothing in this module writes to or otherwise modifies it.
_DEPTH_ORDER: dict[str, int] = {d: i for i, d in enumerate(VALID_DEPTHS)}
_PROFILE_ORDER: dict[str, int] = {p: i for i, p in enumerate(PRIVACY_PROFILES)}

ENV_VAR = "SIFT_ENTERPRISE_POLICY"

# Fixed system-level search paths, checked in order after the env var.
# These are the paths an OS's own permission model protects from an
# ordinary researcher account — Sift relies on that, not on anything
# of its own.
_SYSTEM_PATHS: tuple[Path, ...] = (
    Path("/etc/sift/enterprise_policy.yaml"),
    Path("/Library/Application Support/Sift/enterprise_policy.yaml"),
)


def _windows_program_data_directory() -> Path:
    """Resolve Windows common application data without trusting user input.

    On Windows, use the shell known-folder API rather than the mutable process
    environment: a researcher can set ``PROGRAMDATA`` before launch, so using
    it as the authority would let them redirect the supposedly admin-owned
    policy lookup into their profile. The environment branch exists only for
    cross-platform tests where Windows APIs do not exist.
    """
    if os.name == "nt":  # pragma: no cover - exercised by Windows release CI
        try:
            import ctypes

            buffer = ctypes.create_unicode_buffer(32_768)
            # CSIDL_COMMON_APPDATA (ProgramData), SHGFP_TYPE_CURRENT.
            result = ctypes.windll.shell32.SHGetFolderPathW(  # type: ignore[attr-defined]
                None, 0x23, None, 0, buffer,
            )
            if result == 0 and buffer.value:
                return Path(buffer.value)
        except (AttributeError, OSError, ValueError):
            pass
        return Path(r"C:\ProgramData")
    return Path(
        os.environ.get("PROGRAMDATA")
        or os.environ.get("ProgramData")
        or r"C:\ProgramData"
    )


def _system_policy_paths() -> tuple[Path, ...]:
    """Return OS-admin policy locations in deterministic search order.

    ``ProgramData`` is resolved at call time through Windows' known-folder API
    because deployments may relocate it. The file beneath it must still be
    protected by the deployment's Windows ACLs, just as the POSIX paths rely
    on the permissions of their enclosing directories.
    """
    paths = list(_SYSTEM_PATHS)
    if platform.system() == "Windows":
        paths.insert(
            0,
            _windows_program_data_directory()
            / "Sift"
            / "enterprise_policy.yaml",
        )
    return tuple(paths)

# Testing-only escape hatch. When True, ``_env_path_is_trustworthy`` is
# skipped for env-var-resolved paths. This exists ONLY so this
# module's own test suite -- which creates and therefore owns every
# fixture file it points ``SIFT_ENTERPRISE_POLICY`` at -- can exercise
# the parsing/clamping logic below without standing up real privilege
# separation. It is deliberately a Python module attribute, not
# anything read from the environment or from the policy file itself:
# flipping it requires white-box access (``import`` this module and
# ``monkeypatch.setattr`` the attribute), which a packaged Sift binary
# gives a researcher no path to do. Setting an environment variable --
# the actual attack this module's trust check exists to close -- does
# not touch this flag.
_TESTING_TRUST_ENV_PATH_UNCONDITIONALLY = False


def _env_path_is_trustworthy(path: Path) -> bool:
    """True if ``path`` (resolved via the ``SIFT_ENTERPRISE_POLICY``
    env var) is plausibly admin-authored rather than something the
    researcher running Sift could have written themselves.

    This is the fix for a real gap: unlike the fixed ``_SYSTEM_PATHS``
    -- which are protected by the OS purely because of WHERE they are
    (a non-admin genuinely cannot write to ``/etc/sift/...`` on a
    correctly configured machine) -- an env-var-supplied path can
    point ANYWHERE, including a file the researcher created in their
    own home directory five seconds before launching Sift. Without
    this check, ``SIFT_ENTERPRISE_POLICY=~/mine.yaml`` (pointing at an
    empty, fully-permissive, self-authored file) would silently defeat
    the module's entire stated security property: "a researcher cannot
    lower this floor."

    Checks two things, both required: (a) the file is not owned by the
    current OS user -- ownership, not just current permission bits,
    because an owner can always ``chmod`` their own file back to
    writable, so a point-in-time writability check alone is not
    sufficient; and (b) the current process cannot write to it right
    now. On platforms with no POSIX uid concept (Windows), falls back
    to the writability check alone.
    """
    try:
        st = path.stat()
    except OSError:
        return False
    getuid = getattr(os, "getuid", None)
    if getuid is not None:
        try:
            if st.st_uid == getuid():
                return False
        except OSError:
            return False
    return not os.access(str(path), os.W_OK)


@dataclass(frozen=True)
class EnterprisePolicy:
    """A parsed, validated enterprise policy document.

    Every field defaults to "no additional restriction" — the
    all-defaults instance is a legal, fully permissive-at-this-layer
    policy (equivalent to no file being present at all). Only
    ``load_enterprise_policy()``'s fail-closed path constructs a
    non-default instance without an actual admin-authored file behind
    it.
    """
    version: int = 1
    max_depth_ceiling: str | None = None
    never_expose_fields: frozenset[str] = field(default_factory=frozenset)
    min_privacy_profile: str | None = None
    min_cell_suppression_threshold: int | None = None
    min_n_regression: int | None = None
    min_n_descriptive: int | None = None
    min_n_ttest_group: int | None = None
    dp_epsilon_ceiling: float | None = None
    allow_differential_privacy: bool = True
    export_approval_required: bool = False
    # Integration governance. ``None`` means this layer has no opinion;
    # an empty set means deny every integration of that kind.  A local-model
    # requirement is evaluated against the configured endpoint, not merely
    # the provider name, so a remote OpenAI-compatible gateway cannot pose as
    # a local deployment.
    allowed_model_providers: frozenset[str] | None = None
    allowed_database_backends: frozenset[str] | None = None
    allowed_cloud_sources: frozenset[str] | None = None
    require_local_model: bool = False
    allowed_endpoint_hosts: frozenset[str] | None = None
    allowed_regions: frozenset[str] | None = None
    # Managed-model deployments need a second scope below provider and
    # region.  These allowlists bind Google/Azure project or resource ids and
    # AWS account ids.  ``None`` means the enterprise layer has no opinion;
    # an empty set deliberately denies every managed deployment in that
    # category.  Direct provider credentials never satisfy these checks.
    allowed_cloud_projects: frozenset[str] | None = None
    allowed_cloud_accounts: frozenset[str] | None = None
    require_local_integrations: bool = False
    # Local product-governance controls. These require no external identity
    # provider: an administrator can disable the deliberate third-party
    # feedback submission, keep diagnostics local, prevent diagnostic bundle
    # export, and bound local diagnostic storage.
    allow_external_feedback: bool = True
    feedback_endpoint: str | None = None
    allow_local_diagnostics: bool = True
    allow_diagnostic_exports: bool = True
    diagnostic_retention_days_ceiling: int | None = None
    diagnostic_log_bytes_ceiling: int | None = None
    # Diagnostics only — never used for enforcement. Lets a Privacy
    # Inspector-style surface tell the researcher WHY a ceiling
    # exists ("your org set this"), rather than the reason looking
    # like an unexplained Sift default.
    source_path: str = ""


# The configuration this module falls back to when a policy file was
# found but could not be trusted. Deliberately at or near the
# strictest representable value on every axis — see the module
# docstring's "Absence vs. malformation" section for why.
_FAIL_CLOSED_ENTERPRISE_POLICY = EnterprisePolicy(
    max_depth_ceiling="names_only",
    min_privacy_profile="regulated",
    min_cell_suppression_threshold=25,
    min_n_regression=25,
    min_n_descriptive=25,
    min_n_ttest_group=25,
    dp_epsilon_ceiling=0.1,
    allow_differential_privacy=False,
    export_approval_required=True,
    allowed_model_providers=frozenset({"openai_compatible"}),
    allowed_database_backends=frozenset({"sqlite", "duckdb", "duckdb-file"}),
    allowed_cloud_sources=frozenset(),
    require_local_model=True,
    allowed_endpoint_hosts=frozenset({"localhost", "127.0.0.1", "::1"}),
    allowed_regions=frozenset(),
    allowed_cloud_projects=frozenset(),
    allowed_cloud_accounts=frozenset(),
    require_local_integrations=True,
    allow_external_feedback=False,
    allow_local_diagnostics=False,
    allow_diagnostic_exports=False,
    diagnostic_retention_days_ceiling=1,
    diagnostic_log_bytes_ceiling=1_048_576,
    source_path="<unreadable enterprise policy file>",
)


# ---------------------------------------------------------------------------
# Locate + load
# ---------------------------------------------------------------------------

def enterprise_policy_path() -> Path | None:
    """Resolve the enterprise policy file's location, if any.

    Search order: ``SIFT_ENTERPRISE_POLICY`` env var first (absolute
    path; lets a deployment pin an unusual location and lets tests
    point at a temp file without touching real system paths), then
    the fixed OS-level paths. Returns ``None`` if nothing is found at
    any of them — the ordinary case for a non-enterprise install.
    """
    env_path = os.environ.get(ENV_VAR)
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p
        # An operator SET the env var but the file isn't there — this
        # is a deployment misconfiguration, not "no enterprise
        # policy". Treated as "found but unreadable" by the caller
        # (load_enterprise_policy checks is_file() again and falls
        # through to fail-closed), not as absence.
        return p
    for candidate in _system_policy_paths():
        if candidate.is_file():
            return candidate
    return None


def _clamp_depth(value: Any) -> str | None:
    return value if isinstance(value, str) and value in _DEPTH_ORDER else None


def _clamp_profile(value: Any) -> str | None:
    return value if isinstance(value, str) and value in _PROFILE_ORDER else None


def _clamp_pos_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None


def _clamp_epsilon(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        if f > 0 and f == f and f not in (float("inf"), float("-inf")):
            return f
    return None


def _clamp_bool(value: Any, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _clamp_str_set(value: Any) -> frozenset[str]:
    """Used exclusively for ``never_expose_fields`` -- normalized via
    ``banned_key`` (safe_key + casefold), not bare ``safe_key``, so an
    admin's YAML entry matches a same-named dataset column regardless
    of case. This is the admin-controlled floor no session can loosen
    (see ``apply_banned_variables``); a silent case mismatch here
    would defeat that guarantee with no warning at all.
    """
    if not isinstance(value, list):
        return frozenset()
    return frozenset(
        banned_key(str(v)) for v in value if isinstance(v, str) and v
    )


def _clamp_optional_id_set(value: Any) -> frozenset[str] | None:
    """Validate an optional integration allowlist without conflating an
    intentional empty list (deny all) with an absent/invalid field."""
    if value is None:
        return None
    if not isinstance(value, list) or not all(
        isinstance(v, str) and v.strip() for v in value
    ):
        return None
    return frozenset(v.strip().casefold() for v in value)


def _clamp_https_url(value: Any) -> str | None:
    """Accept only an exact HTTPS endpoint without embedded credentials."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return None
    if (
        parsed.scheme.casefold() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        return None
    return value.strip()


_POLICY_FIELDS = frozenset(
    {
        "version",
        "max_depth_ceiling",
        "never_expose_fields",
        "min_privacy_profile",
        "min_cell_suppression_threshold",
        "min_n_regression",
        "min_n_descriptive",
        "min_n_ttest_group",
        "dp_epsilon_ceiling",
        "allow_differential_privacy",
        "export_approval_required",
        "allowed_model_providers",
        "allowed_database_backends",
        "allowed_cloud_sources",
        "require_local_model",
        "allowed_endpoint_hosts",
        "allowed_regions",
        "allowed_cloud_projects",
        "allowed_cloud_accounts",
        "require_local_integrations",
        "allow_external_feedback",
        "feedback_endpoint",
        "allow_local_diagnostics",
        "allow_diagnostic_exports",
        "diagnostic_retention_days_ceiling",
        "diagnostic_log_bytes_ceiling",
    }
)


def _policy_fields_are_valid(data: dict[str, Any]) -> bool:
    """Validate every field that is present in an enterprise document.

    Missing fields remain "no opinion". Present invalid or misspelled fields
    fail the document closed: silently ignoring an admin typo such as
    ``export_approval_requird`` would disable the exact control they meant to
    deploy.
    """
    if not set(data).issubset(_POLICY_FIELDS):
        return False
    validators = {
        "max_depth_ceiling": lambda value: _clamp_depth(value) is not None,
        "never_expose_fields": lambda value: isinstance(value, list)
        and all(isinstance(v, str) and bool(v) for v in value),
        "min_privacy_profile": lambda value: _clamp_profile(value) is not None,
        "min_cell_suppression_threshold": lambda value: _clamp_pos_int(value)
        is not None,
        "min_n_regression": lambda value: _clamp_pos_int(value) is not None,
        "min_n_descriptive": lambda value: _clamp_pos_int(value) is not None,
        "min_n_ttest_group": lambda value: _clamp_pos_int(value) is not None,
        "dp_epsilon_ceiling": lambda value: _clamp_epsilon(value) is not None,
        "allow_differential_privacy": lambda value: isinstance(value, bool),
        "export_approval_required": lambda value: isinstance(value, bool),
        "allowed_model_providers": lambda value: value is None
        or (
            isinstance(value, list)
            and all(isinstance(v, str) and bool(v.strip()) for v in value)
        ),
        "allowed_database_backends": lambda value: value is None
        or (
            isinstance(value, list)
            and all(isinstance(v, str) and bool(v.strip()) for v in value)
        ),
        "allowed_cloud_sources": lambda value: value is None
        or (
            isinstance(value, list)
            and all(isinstance(v, str) and bool(v.strip()) for v in value)
        ),
        "require_local_model": lambda value: isinstance(value, bool),
        "allowed_endpoint_hosts": lambda value: value is None
        or (
            isinstance(value, list)
            and all(isinstance(v, str) and bool(v.strip()) for v in value)
        ),
        "allowed_regions": lambda value: value is None
        or (
            isinstance(value, list)
            and all(isinstance(v, str) and bool(v.strip()) for v in value)
        ),
        "allowed_cloud_projects": lambda value: value is None
        or (
            isinstance(value, list)
            and all(isinstance(v, str) and bool(v.strip()) for v in value)
        ),
        "allowed_cloud_accounts": lambda value: value is None
        or (
            isinstance(value, list)
            and all(isinstance(v, str) and bool(v.strip()) for v in value)
        ),
        "require_local_integrations": lambda value: isinstance(value, bool),
        "allow_external_feedback": lambda value: isinstance(value, bool),
        "feedback_endpoint": lambda value: _clamp_https_url(value) is not None,
        "allow_local_diagnostics": lambda value: isinstance(value, bool),
        "allow_diagnostic_exports": lambda value: isinstance(value, bool),
        "diagnostic_retention_days_ceiling": lambda value: _clamp_pos_int(value)
        is not None,
        "diagnostic_log_bytes_ceiling": lambda value: _clamp_pos_int(value)
        is not None,
    }
    return all(
        key not in data or validator(data[key])
        for key, validator in validators.items()
    )


def load_enterprise_policy() -> EnterprisePolicy | None:
    """Load and validate the enterprise policy, if one is deployed.

    Never raises. Three outcomes, matching ``policy.load_policy``'s
    own three-way split:

    - No file at any resolved location: ``None`` (no enterprise layer
      applies; every combinator in this module becomes a no-op).
    - File present but unparseable / wrong shape / future version:
      the fail-closed policy (strictest representable configuration
      on every axis).
    - File present and parseable: the validated document. Missing fields mean
      "no opinion on this axis". A present malformed or unknown field fails
      closed because silently ignoring an admin typo can disable the exact
      restriction they intended to deploy.
    """
    p = enterprise_policy_path()
    if p is None:
        return None
    if not p.is_file():
        return _FAIL_CLOSED_ENTERPRISE_POLICY

    # Trust gate: a path resolved from SIFT_ENTERPRISE_POLICY carries
    # none of the fixed-location protection ``_SYSTEM_PATHS`` gets
    # "for free" from the OS -- see ``_env_path_is_trustworthy``'s
    # docstring. Without this, any researcher could point the env var
    # at a file they themselves authored (or an empty file, which
    # parses below as "legal, fully permissive") and defeat the
    # module's entire reason for existing. Resolution via the fixed
    # system paths is unaffected: those remain protected purely by
    # being outside the researcher's writable filesystem, exactly as
    # the module docstring describes.
    env_path = os.environ.get(ENV_VAR)
    resolved_via_env = env_path is not None and bool(env_path) and p == Path(env_path)
    if (
        resolved_via_env
        and not _TESTING_TRUST_ENV_PATH_UNCONDITIONALLY
        and not _env_path_is_trustworthy(p)
    ):
        return _FAIL_CLOSED_ENTERPRISE_POLICY

    try:
        import yaml
        text = p.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
    except Exception:  # noqa: BLE001 — any parse/read failure fails closed
        return _FAIL_CLOSED_ENTERPRISE_POLICY
    if data is None:
        # A present but empty admin policy is almost always a truncated or
        # half-deployed file. Treat absence as "no enterprise layer", but
        # fail closed once a policy path exists and contains no document.
        return _FAIL_CLOSED_ENTERPRISE_POLICY
    if not isinstance(data, dict):
        return _FAIL_CLOSED_ENTERPRISE_POLICY
    if "version" in data and data.get("version") != 1:
        return _FAIL_CLOSED_ENTERPRISE_POLICY
    if not _policy_fields_are_valid(data):
        return _FAIL_CLOSED_ENTERPRISE_POLICY

    return EnterprisePolicy(
        version=1,
        max_depth_ceiling=_clamp_depth(data.get("max_depth_ceiling")),
        never_expose_fields=_clamp_str_set(data.get("never_expose_fields")),
        min_privacy_profile=_clamp_profile(data.get("min_privacy_profile")),
        min_cell_suppression_threshold=_clamp_pos_int(
            data.get("min_cell_suppression_threshold")),
        min_n_regression=_clamp_pos_int(data.get("min_n_regression")),
        min_n_descriptive=_clamp_pos_int(data.get("min_n_descriptive")),
        min_n_ttest_group=_clamp_pos_int(data.get("min_n_ttest_group")),
        dp_epsilon_ceiling=_clamp_epsilon(data.get("dp_epsilon_ceiling")),
        allow_differential_privacy=_clamp_bool(
            data.get("allow_differential_privacy"), True),
        export_approval_required=_clamp_bool(
            data.get("export_approval_required"), False),
        allowed_model_providers=_clamp_optional_id_set(
            data.get("allowed_model_providers")),
        allowed_database_backends=_clamp_optional_id_set(
            data.get("allowed_database_backends")),
        allowed_cloud_sources=_clamp_optional_id_set(
            data.get("allowed_cloud_sources")),
        require_local_model=_clamp_bool(
            data.get("require_local_model"), False),
        allowed_endpoint_hosts=_clamp_optional_id_set(
            data.get("allowed_endpoint_hosts")),
        allowed_regions=_clamp_optional_id_set(data.get("allowed_regions")),
        allowed_cloud_projects=_clamp_optional_id_set(
            data.get("allowed_cloud_projects")),
        allowed_cloud_accounts=_clamp_optional_id_set(
            data.get("allowed_cloud_accounts")),
        require_local_integrations=_clamp_bool(
            data.get("require_local_integrations"), False),
        allow_external_feedback=_clamp_bool(
            data.get("allow_external_feedback"), True),
        feedback_endpoint=_clamp_https_url(data.get("feedback_endpoint")),
        allow_local_diagnostics=_clamp_bool(
            data.get("allow_local_diagnostics"), True),
        allow_diagnostic_exports=_clamp_bool(
            data.get("allow_diagnostic_exports"), True),
        diagnostic_retention_days_ceiling=_clamp_pos_int(
            data.get("diagnostic_retention_days_ceiling")),
        diagnostic_log_bytes_ceiling=_clamp_pos_int(
            data.get("diagnostic_log_bytes_ceiling")),
        source_path=str(p),
    )


# ---------------------------------------------------------------------------
# Combinators — every function here returns a value AT LEAST as
# restrictive as its session-derived input. This is the load-bearing
# security property of the whole module.
# ---------------------------------------------------------------------------

def external_feedback_allowed(enterprise: EnterprisePolicy | None) -> bool:
    """Whether researcher-authored text may leave through feedback."""
    return enterprise is None or enterprise.allow_external_feedback


def resolve_feedback_endpoint(
    default_endpoint: str, enterprise: EnterprisePolicy | None,
) -> str:
    """Use an admin-pinned HTTPS collector when one is configured."""
    if enterprise is None or enterprise.feedback_endpoint is None:
        return default_endpoint
    return enterprise.feedback_endpoint


def local_diagnostics_allowed(enterprise: EnterprisePolicy | None) -> bool:
    """Whether Sift may retain redacted diagnostic logs on this device."""
    return enterprise is None or enterprise.allow_local_diagnostics


def diagnostic_exports_allowed(enterprise: EnterprisePolicy | None) -> bool:
    """Whether a future explicit diagnostic bundle may leave local storage."""
    return enterprise is None or enterprise.allow_diagnostic_exports


def apply_diagnostic_retention_ceiling(
    requested_days: int, enterprise: EnterprisePolicy | None,
) -> int:
    """Apply the admin's maximum log lifetime without ever extending it."""
    requested = max(1, int(requested_days))
    ceiling = (
        enterprise.diagnostic_retention_days_ceiling
        if enterprise is not None else None
    )
    return min(requested, ceiling) if ceiling is not None else requested


def apply_diagnostic_bytes_ceiling(
    requested_bytes: int, enterprise: EnterprisePolicy | None,
) -> int:
    """Apply the admin's maximum local log bytes without enlarging it."""
    requested = max(1, int(requested_bytes))
    ceiling = (
        enterprise.diagnostic_log_bytes_ceiling
        if enterprise is not None else None
    )
    return min(requested, ceiling) if ceiling is not None else requested

def apply_depth_ceiling(
    session_ceiling: str, enterprise: EnterprisePolicy | None,
) -> str:
    """Combine a session-derived schema-depth ceiling with the
    enterprise ceiling, if any. Returns whichever is stricter.
    """
    if enterprise is None or enterprise.max_depth_ceiling is None:
        return session_ceiling
    session_rank = _DEPTH_ORDER.get(session_ceiling, 0)
    ent_rank = _DEPTH_ORDER.get(enterprise.max_depth_ceiling, 0)
    return session_ceiling if session_rank <= ent_rank else enterprise.max_depth_ceiling


def apply_privacy_profile_floor(
    session_profile: str, enterprise: EnterprisePolicy | None,
) -> str:
    """Combine a session-derived privacy profile with the enterprise
    floor, if any. Returns whichever is MORE restrictive (higher
    ``PRIVACY_PROFILES`` rank).
    """
    if enterprise is None or enterprise.min_privacy_profile is None:
        return session_profile
    session_rank = _PROFILE_ORDER.get(session_profile, 0)
    floor_rank = _PROFILE_ORDER.get(enterprise.min_privacy_profile, 0)
    return (
        session_profile if session_rank >= floor_rank
        else enterprise.min_privacy_profile
    )


def apply_banned_variables(
    session_banned: frozenset[str], enterprise: EnterprisePolicy | None,
) -> frozenset[str]:
    """Union a dataset's banned-variable set with the enterprise's
    global never-expose list. Union only -- this can only grow the
    banned set, never shrink it.
    """
    if enterprise is None or not enterprise.never_expose_fields:
        return session_banned
    return session_banned | enterprise.never_expose_fields


def apply_sdc_floor(
    sdc_cfg: SDCConfig, enterprise: EnterprisePolicy | None,
) -> SDCConfig:
    """Raise ``sdc_cfg``'s suppression/min-N thresholds and clamp its
    DP epsilon to the enterprise floor/ceiling, if any.

    Every field this function touches only moves in the stricter
    direction: thresholds go up (never down), epsilon goes down
    (never up, since a smaller epsilon means more injected noise —
    i.e. MORE private), and DP is only ever turned OFF by this
    function, never on (an enterprise policy cannot force a
    researcher session that never opted into ``dp_epsilon`` to
    suddenly have one — that would change what mechanism runs, not
    just how strict it is).
    """
    if enterprise is None:
        return sdc_cfg
    updates: dict[str, Any] = {}
    if enterprise.never_expose_fields:
        merged = sdc_cfg.banned_variables | enterprise.never_expose_fields
        if merged != sdc_cfg.banned_variables:
            updates["banned_variables"] = merged
    if (enterprise.min_cell_suppression_threshold is not None
            and enterprise.min_cell_suppression_threshold
            > sdc_cfg.cell_suppression_threshold):
        updates["cell_suppression_threshold"] = (
            enterprise.min_cell_suppression_threshold)
    if (enterprise.min_n_regression is not None
            and enterprise.min_n_regression > sdc_cfg.min_n_regression):
        updates["min_n_regression"] = enterprise.min_n_regression
    if (enterprise.min_n_descriptive is not None
            and enterprise.min_n_descriptive > sdc_cfg.min_n_descriptive):
        updates["min_n_descriptive"] = enterprise.min_n_descriptive
    if (enterprise.min_n_ttest_group is not None
            and enterprise.min_n_ttest_group > sdc_cfg.min_n_ttest_group):
        updates["min_n_ttest_group"] = enterprise.min_n_ttest_group
    if sdc_cfg.dp_epsilon is not None:
        if not enterprise.allow_differential_privacy:
            updates["dp_epsilon"] = None
        elif (enterprise.dp_epsilon_ceiling is not None
              and sdc_cfg.dp_epsilon > enterprise.dp_epsilon_ceiling):
            updates["dp_epsilon"] = enterprise.dp_epsilon_ceiling
    if not updates:
        return sdc_cfg
    return replace(sdc_cfg, **updates)


def export_requires_approval(enterprise: EnterprisePolicy | None) -> bool:
    return enterprise is not None and enterprise.export_approval_required


def model_provider_allowed(
    provider: str, enterprise: EnterprisePolicy | None,
) -> bool:
    """Apply the enterprise provider allowlist and local-only rule."""
    if enterprise is None:
        return True
    normalized = str(provider).casefold()
    if (enterprise.allowed_model_providers is not None
            and normalized not in enterprise.allowed_model_providers):
        return False
    if enterprise.require_local_model:
        try:
            from sift.integrations import provider_is_local
            if not provider_is_local(normalized):
                return False
        except Exception:  # noqa: BLE001 — uncertainty fails closed
            return False
    endpoint = {
        "openai": "https://api.openai.com",
        "anthropic": "https://api.anthropic.com",
        "gemini": "https://generativelanguage.googleapis.com",
        "azure_openai": os.environ.get("SIFT_AZURE_OPENAI_ENDPOINT"),
        "vertex_gemini": _vertex_endpoint(
            os.environ.get("SIFT_VERTEX_GEMINI_LOCATION")),
        "vertex_anthropic": _vertex_endpoint(
            os.environ.get("SIFT_VERTEX_ANTHROPIC_LOCATION")),
        "bedrock_anthropic": _bedrock_endpoint(
            os.environ.get("SIFT_BEDROCK_REGION")),
    }.get(normalized)
    if normalized == "openai_compatible":
        endpoint = os.environ.get("SIFT_OPENAI_COMPATIBLE_BASE_URL")
    if not integration_endpoint_allowed(endpoint, enterprise):
        return False
    return True


def _vertex_endpoint(location: str | None) -> str | None:
    if not location:
        return None
    normalized = location.strip().casefold()
    if normalized == "global":
        return "https://aiplatform.googleapis.com"
    if normalized in {"us", "eu"}:
        return f"https://aiplatform.{normalized}.rep.googleapis.com"
    return f"https://{normalized}-aiplatform.googleapis.com"


def _bedrock_endpoint(region: str | None) -> str | None:
    if not region:
        return None
    normalized = region.strip().casefold()
    suffix = "amazonaws.com.cn" if normalized.startswith("cn-") else "amazonaws.com"
    return f"https://bedrock-runtime.{normalized}.{suffix}"


def database_backend_allowed(
    backend: str, enterprise: EnterprisePolicy | None,
) -> bool:
    """Apply the enterprise database-backend allowlist."""
    if enterprise is None or enterprise.allowed_database_backends is None:
        return True
    return str(backend).casefold() in enterprise.allowed_database_backends


def cloud_source_allowed(
    source_kind: str, enterprise: EnterprisePolicy | None,
) -> bool:
    """Apply the enterprise cloud-source allowlist."""
    if enterprise is None or enterprise.allowed_cloud_sources is None:
        return True
    return str(source_kind).casefold() in enterprise.allowed_cloud_sources


def _host_matches_allowlist(host: str, allowed: frozenset[str]) -> bool:
    normalized = host.rstrip(".").casefold()
    for rule in allowed:
        candidate = rule.rstrip(".").casefold()
        if candidate.startswith("*."):
            suffix = candidate[1:]
            if normalized.endswith(suffix) and normalized != suffix[1:]:
                return True
        elif normalized == candidate:
            return True
    return False


def integration_endpoint_allowed(
    endpoint: str | None,
    enterprise: EnterprisePolicy | None,
    *,
    local_hint: bool = False,
) -> bool:
    """Apply local-only and exact/wildcard hostname restrictions.

    Ambiguous or hostless remote endpoints fail closed when either restriction
    is configured. ``local_hint`` is reserved for already-validated local file
    integrations such as SQLite and DuckDB.
    """
    if enterprise is None:
        return True
    is_local = local_hint
    host: str | None = None
    if endpoint:
        try:
            parsed = urlsplit(endpoint)
            host = parsed.hostname
        except (TypeError, ValueError):
            host = None
        if host:
            from sift.integrations import endpoint_is_local
            is_local = endpoint_is_local(endpoint)
    if enterprise.require_local_integrations and not is_local:
        return False
    allowed = enterprise.allowed_endpoint_hosts
    if allowed is None:
        return True
    if local_hint and not endpoint:
        return True
    return bool(host) and _host_matches_allowlist(host or "", allowed)


def integration_region_allowed(
    region: str | None,
    enterprise: EnterprisePolicy | None,
) -> bool:
    if enterprise is None or enterprise.allowed_regions is None:
        return True
    if not region:
        return False
    return region.strip().casefold() in enterprise.allowed_regions


def managed_project_allowed(
    project: str | None,
    enterprise: EnterprisePolicy | None,
) -> bool:
    """Apply the approved Azure/GCP project or resource allowlist."""
    if enterprise is None or enterprise.allowed_cloud_projects is None:
        return True
    if not project:
        return False
    return project.strip().casefold() in enterprise.allowed_cloud_projects


def managed_account_allowed(
    account: str | None,
    enterprise: EnterprisePolicy | None,
) -> bool:
    """Apply the approved AWS account allowlist."""
    if enterprise is None or enterprise.allowed_cloud_accounts is None:
        return True
    if not account:
        return False
    return account.strip().casefold() in enterprise.allowed_cloud_accounts


def fail_closed_policy() -> EnterprisePolicy:
    """Public accessor for the strictest representable enterprise
    policy — the same instance ``load_enterprise_policy()`` falls
    back to when a deployed file can't be trusted. Exposed so callers
    that must themselves fail closed on an unrelated error (e.g. the
    UI bridge's export gate, if the load call itself raises) can
    reach for the same conservative value without importing a
    private module attribute.
    """
    return _FAIL_CLOSED_ENTERPRISE_POLICY


# ---------------------------------------------------------------------------
# Export-approval workflow (local, file-based -- see module docstring)
# ---------------------------------------------------------------------------

_REQUESTS_DIR = Path(".sift") / "export_requests"


def _requests_dir(cwd: Path) -> Path:
    return cwd / _REQUESTS_DIR


def _session_revision(cwd: Path) -> str:
    """Fingerprint the local material an export can draw from.

    This intentionally hashes file identity/size/mtime rather than file
    contents. Export gating remains a local workflow primitive, not a
    tamper-proof authorization system, and reading multi-gigabyte datasets
    merely to ask for approval would be unusable. The revision does bind an
    approval to the ordinary observable state reviewed at request time.

    Existing exports and approval records are excluded because they are
    outputs of the workflow, not source material. Symlinks are not followed.
    Races fail conservatively by adding an explicit marker to the digest.
    """
    root = Path(cwd)
    digest = hashlib.sha256()
    try:
        for current, dirnames, filenames in os.walk(root, followlinks=False):
            current_path = Path(current)
            try:
                rel_root = current_path.relative_to(root)
            except ValueError:
                continue

            # Do not let the workflow's own records or prior exports make a
            # request stale. Sort in place for deterministic traversal.
            if rel_root == Path("."):
                dirnames[:] = [d for d in dirnames if d != "exports"]
            if rel_root == Path(".sift"):
                dirnames[:] = [d for d in dirnames if d != "export_requests"]
            dirnames.sort()

            for name in sorted(filenames):
                path = current_path / name
                try:
                    rel = path.relative_to(root).as_posix()
                    stat = path.lstat()
                    target = os.readlink(path) if path.is_symlink() else ""
                    item = (
                        f"{rel}\0{stat.st_mode}\0{stat.st_size}\0"
                        f"{stat.st_mtime_ns}\0{target}\n"
                    )
                except OSError:
                    item = f"<changed-during-scan>\0{path.name}\n"
                digest.update(item.encode("utf-8", errors="surrogateescape"))
    except OSError:
        digest.update(b"<unreadable-session>")

    # Bind approval to the deployed governance policy too. It usually lives
    # outside ``cwd`` and would otherwise be invisible to the session scan.
    # Policy files are tiny, so hash their bytes rather than metadata.
    policy_path = enterprise_policy_path()
    if policy_path is not None:
        try:
            digest.update(b"\0enterprise-policy\0")
            digest.update(policy_path.read_bytes())
        except OSError:
            digest.update(b"\0enterprise-policy-unreadable\0")
    return digest.hexdigest()


def request_export_approval(cwd: Path, export_kind: str) -> dict[str, Any]:
    """Record a pending export request instead of producing the export.

    Returns the request record (including its ``id``) so the caller
    (a ``ui.py`` bridge method) can surface it to the researcher —
    "your export is pending approval, request id X" — rather than a
    bare failure.
    """
    d = _requests_dir(cwd)
    d.mkdir(parents=True, exist_ok=True)
    from sift.file_lock import exclusive_file_lock
    from sift.reliability import atomic_write_json, clock_safe_timestamp

    with exclusive_file_lock(d / "requests.lock"):
        revision = _session_revision(cwd)
        # Recheck while holding the cross-process mutation lock: two double
        # clicks must converge on one pending request.
        for existing in list_export_requests(cwd):
            if (existing.get("export_kind") == export_kind
                    and existing.get("session_revision") == revision
                    and existing.get("status") == "pending"):
                return existing
        req_id = uuid.uuid4().hex[:12]
        record = {
            "id": req_id,
            "export_kind": export_kind,
            "requested_at": clock_safe_timestamp(),
            "status": "pending",
            "host": platform.node() or "unknown-host",
            "session_revision": revision,
        }
        atomic_write_json(d / f"{req_id}.json", record, use_lock=False)
        return record


def list_export_requests(cwd: Path) -> list[dict[str, Any]]:
    """Return every export-approval request recorded for this session,
    newest first. Never raises -- a missing/corrupt directory yields
    an empty list.
    """
    d = _requests_dir(cwd)
    if not d.is_dir():
        return []
    out: list[dict[str, Any]] = []
    try:
        files = sorted(d.glob("*.json"))
    except OSError:
        return []
    for f in files:
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(rec, dict):
            out.append(rec)
    out.sort(key=lambda r: r.get("requested_at", ""), reverse=True)
    return out


def is_export_approved(cwd: Path, export_kind: str) -> bool:
    """Return whether the most recent request for ``export_kind`` has
    been approved.

    Deliberately keyed on "most recent request for this kind", not
    "any request ever approved for this kind" — a researcher who
    changed the underlying session material after an old approval
    should not get to reuse it indefinitely; each export attempt that
    finds no pending-or-approved request of its own creates a fresh
    one via ``request_export_approval``.
    """
    revision = _session_revision(cwd)
    for rec in list_export_requests(cwd):
        if rec.get("export_kind") == export_kind:
            return (rec.get("status") == "approved"
                    and rec.get("session_revision") == revision)
    return False


def approve_export(
    cwd: Path, request_id: str, *, approver_note: str = "",
) -> dict[str, Any]:
    """Mark a pending export request as approved.

    Intended to be called by someone OTHER than the researcher whose
    session this is — see the module docstring's honesty note about
    what this workflow can and cannot guarantee about who called it.
    Returns ``{"ok": False, "reason": ...}`` for an unknown id or a
    request that isn't pending; never raises.
    """
    d = _requests_dir(cwd)
    f = d / f"{request_id}.json"
    from sift.file_lock import exclusive_file_lock
    from sift.reliability import atomic_write_json, clock_safe_timestamp

    with exclusive_file_lock(d / "requests.lock"):
        if not f.is_file():
            return {"ok": False, "reason": f"no export request with id {request_id!r}"}
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            return {"ok": False, "reason": f"request record unreadable: {e}"}
        if not isinstance(rec, dict):
            return {"ok": False, "reason": "request record malformed"}
        if rec.get("status") != "pending":
            return {"ok": False, "reason": f"request is {rec.get('status')!r}, not pending"}
        requested_revision = rec.get("session_revision")
        current_revision = _session_revision(cwd)
        if (not isinstance(requested_revision, str)
                or requested_revision != current_revision):
            rec["status"] = "stale"
            rec["stale_at"] = clock_safe_timestamp(rec.get("requested_at"))
            atomic_write_json(f, rec, use_lock=False)
            return {
                "ok": False,
                "reason": (
                    "session material changed after this export request; "
                    "create and review a new request"
                ),
            }
        rec["status"] = "approved"
        rec["approved_at"] = clock_safe_timestamp(rec.get("requested_at"))
        if approver_note:
            rec["approver_note"] = str(approver_note)[:500]
        atomic_write_json(f, rec, use_lock=False)
        return {"ok": True, "request": rec}
