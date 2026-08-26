"""Enterprise policy: the admin-authored floor over
researcher policy.

The property this whole module exists to guarantee is one-directional
combination: every function that mixes a session-derived value with
an enterprise value must return something AT LEAST as restrictive as
the session value alone. That's the load-bearing invariant tested
here, plus the absence/malformation/well-formed three-way load split
that mirrors ``policy.py``'s own established behaviour, plus the
integration points in ``tools.py`` (schema depth ceiling, banned
variables, SDC floor) and ``ui.py`` (export approval gate).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from sift import enterprise_policy as ep
from sift.sanitizer import DEFAULT_CONFIG, SDCConfig


@pytest.fixture()
def clean_env(monkeypatch):
    """Ensure no stray SIFT_ENTERPRISE_POLICY leaks between tests, and
    that the fixed system paths (which won't exist in a sandbox but
    could in principle on a real machine running this suite) don't
    accidentally get picked up.
    """
    monkeypatch.delenv(ep.ENV_VAR, raising=False)
    monkeypatch.setattr(ep, "_SYSTEM_PATHS", ())
    # Every fixture file these tests point SIFT_ENTERPRISE_POLICY at is
    # created (and therefore owned/writable) by this test process, so
    # it would otherwise always fail the real ownership/writability
    # trust gate ``load_enterprise_policy`` applies to env-var-resolved
    # paths (see enterprise_policy.py's ``_env_path_is_trustworthy``).
    # That gate exists to stop a researcher from pointing the env var
    # at a file THEY authored; it is not something this suite's own
    # parsing/clamping tests are trying to exercise, so it is disabled
    # here via the white-box-only testing flag -- the dedicated
    # "trust gate" tests below turn it back off to verify the gate
    # itself.
    monkeypatch.setattr(ep, "_TESTING_TRUST_ENV_PATH_UNCONDITIONALLY", True)
    yield


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "enterprise_policy.yaml"
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# load_enterprise_policy — absence / malformation / well-formed
# ---------------------------------------------------------------------------

def test_no_file_anywhere_returns_none(clean_env, tmp_path):
    assert ep.load_enterprise_policy() is None


def test_env_var_missing_file_fails_closed(clean_env, tmp_path, monkeypatch):
    monkeypatch.setenv(ep.ENV_VAR, str(tmp_path / "does_not_exist.yaml"))
    result = ep.load_enterprise_policy()
    assert result is not None
    assert result.export_approval_required is True
    assert result.max_depth_ceiling == "names_only"


def test_empty_file_fails_closed(clean_env, tmp_path, monkeypatch):
    p = _write(tmp_path, "")
    monkeypatch.setenv(ep.ENV_VAR, str(p))
    result = ep.load_enterprise_policy()
    assert result == ep._FAIL_CLOSED_ENTERPRISE_POLICY


def test_garbage_yaml_fails_closed(clean_env, tmp_path, monkeypatch):
    p = _write(tmp_path, "{{{ not: valid: yaml ][")
    monkeypatch.setenv(ep.ENV_VAR, str(p))
    result = ep.load_enterprise_policy()
    assert result.allow_differential_privacy is False
    assert result.export_approval_required is True


def test_yaml_list_not_dict_fails_closed(clean_env, tmp_path, monkeypatch):
    p = _write(tmp_path, "- a\n- b\n")
    monkeypatch.setenv(ep.ENV_VAR, str(p))
    result = ep.load_enterprise_policy()
    assert result.max_depth_ceiling == "names_only"


def test_future_version_fails_closed(clean_env, tmp_path, monkeypatch):
    p = _write(tmp_path, "version: 2\nmax_depth_ceiling: names_only\n")
    monkeypatch.setenv(ep.ENV_VAR, str(p))
    result = ep.load_enterprise_policy()
    assert result.export_approval_required is True  # fail-closed, not the file's own (absent) value


def test_well_formed_file_parses_every_field(clean_env, tmp_path, monkeypatch):
    p = _write(tmp_path, """
version: 1
max_depth_ceiling: names_types_labels
never_expose_fields:
  - ssn
  - national_id
min_privacy_profile: confidential
min_cell_suppression_threshold: 20
min_n_regression: 30
min_n_descriptive: 30
min_n_ttest_group: 30
dp_epsilon_ceiling: 0.5
allow_differential_privacy: true
export_approval_required: true
allowed_model_providers: [openai_compatible, anthropic]
allowed_database_backends: [sqlite, postgresql]
allowed_cloud_sources: [s3, gcs]
require_local_model: true
allowed_endpoint_hosts: [localhost, "*.approved.example"]
allowed_regions: [us-west-2, europe-west4]
require_local_integrations: true
allow_external_feedback: false
feedback_endpoint: https://feedback.example.edu/sift/submit
allow_local_diagnostics: true
allow_diagnostic_exports: false
diagnostic_retention_days_ceiling: 5
diagnostic_log_bytes_ceiling: 2097152
""")
    monkeypatch.setenv(ep.ENV_VAR, str(p))
    result = ep.load_enterprise_policy()
    assert result.max_depth_ceiling == "names_types_labels"
    assert result.never_expose_fields == frozenset({"ssn", "national_id"})
    assert result.min_privacy_profile == "confidential"
    assert result.min_cell_suppression_threshold == 20
    assert result.min_n_regression == 30
    assert result.dp_epsilon_ceiling == 0.5
    assert result.allow_differential_privacy is True
    assert result.export_approval_required is True
    assert result.allowed_model_providers == frozenset({
        "openai_compatible", "anthropic",
    })
    assert result.allowed_database_backends == frozenset({
        "sqlite", "postgresql",
    })
    assert result.allowed_cloud_sources == frozenset({"s3", "gcs"})
    assert result.require_local_model is True
    assert result.allowed_endpoint_hosts == frozenset({
        "localhost", "*.approved.example",
    })
    assert result.allowed_regions == frozenset({"us-west-2", "europe-west4"})
    assert result.require_local_integrations is True
    assert result.allow_external_feedback is False
    assert result.feedback_endpoint == "https://feedback.example.edu/sift/submit"
    assert result.allow_local_diagnostics is True
    assert result.allow_diagnostic_exports is False
    assert result.diagnostic_retention_days_ceiling == 5
    assert result.diagnostic_log_bytes_ceiling == 2_097_152
    assert result.source_path == str(p)


def test_single_bad_field_fails_document_closed(clean_env, tmp_path, monkeypatch):
    # A typo in a present admin field must never silently remove that axis.
    p = _write(tmp_path, """
version: 1
max_depth_ceiling: names_types_labels
min_privacy_profile: super-duper-secret
export_approval_required: false
""")
    monkeypatch.setenv(ep.ENV_VAR, str(p))
    result = ep.load_enterprise_policy()
    assert result.max_depth_ceiling == "names_only"
    assert result.min_privacy_profile == "regulated"
    assert result.export_approval_required is True


def test_negative_and_zero_thresholds_fail_closed(clean_env, tmp_path, monkeypatch):
    p = _write(tmp_path, """
version: 1
min_cell_suppression_threshold: -5
min_n_regression: 0
dp_epsilon_ceiling: -1.0
""")
    monkeypatch.setenv(ep.ENV_VAR, str(p))
    result = ep.load_enterprise_policy()
    assert result.min_cell_suppression_threshold == 25
    assert result.min_n_regression == 25
    assert result.dp_epsilon_ceiling == 0.1


def test_unknown_field_typo_fails_closed(clean_env, tmp_path, monkeypatch):
    p = _write(
        tmp_path,
        "version: 1\nexport_approval_requird: true\n",
    )
    monkeypatch.setenv(ep.ENV_VAR, str(p))
    result = ep.load_enterprise_policy()
    assert result.export_approval_required is True
    assert result.require_local_model is True
    assert result.require_local_integrations is True
    assert result.allow_external_feedback is False
    assert result.allow_local_diagnostics is False
    assert result.allow_diagnostic_exports is False


def test_windows_programdata_policy_is_discovered(
    clean_env, tmp_path, monkeypatch,
) -> None:
    program_data = tmp_path / "ProgramData"
    policy_path = program_data / "Sift" / "enterprise_policy.yaml"
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text(
        "version: 1\nallow_external_feedback: false\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ep.platform, "system", lambda: "Windows")
    # Production Windows deliberately ignores mutable PROGRAMDATA and asks
    # the shell known-folder API. Stub that trusted boundary directly so this
    # remains deterministic on both Windows and POSIX test hosts.
    monkeypatch.setattr(
        ep, "_windows_program_data_directory", lambda: program_data,
    )
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path / "attacker-controlled"))

    assert ep.enterprise_policy_path() == policy_path
    policy = ep.load_enterprise_policy()
    assert policy is not None
    assert policy.source_path == str(policy_path)
    assert policy.allow_external_feedback is False


def test_local_governance_controls_only_tighten() -> None:
    policy = ep.EnterprisePolicy(
        allow_external_feedback=False,
        allow_local_diagnostics=False,
        allow_diagnostic_exports=False,
        diagnostic_retention_days_ceiling=3,
        diagnostic_log_bytes_ceiling=2048,
    )
    assert ep.external_feedback_allowed(policy) is False
    assert ep.local_diagnostics_allowed(policy) is False
    assert ep.diagnostic_exports_allowed(policy) is False
    assert ep.apply_diagnostic_retention_ceiling(7, policy) == 3
    assert ep.apply_diagnostic_retention_ceiling(1, policy) == 1
    assert ep.apply_diagnostic_bytes_ceiling(8192, policy) == 2048
    assert ep.apply_diagnostic_bytes_ceiling(1024, policy) == 1024


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://feedback.example.edu/submit",
        "https://user:secret@feedback.example.edu/submit",
        "https://feedback.example.edu/submit#fragment",
    ],
)
def test_unsafe_feedback_endpoint_fails_document_closed(
    clean_env, tmp_path, monkeypatch, endpoint,
) -> None:
    p = _write(
        tmp_path,
        f"version: 1\nfeedback_endpoint: {endpoint}\n",
    )
    monkeypatch.setenv(ep.ENV_VAR, str(p))
    result = ep.load_enterprise_policy()
    assert result == ep._FAIL_CLOSED_ENTERPRISE_POLICY


# ---------------------------------------------------------------------------
# Combinators — one-directional guarantee
# ---------------------------------------------------------------------------

def test_apply_depth_ceiling_none_enterprise_is_noop():
    assert ep.apply_depth_ceiling("names_types_labels_summary", None) == \
        "names_types_labels_summary"


def test_apply_depth_ceiling_takes_stricter():
    ent = ep.EnterprisePolicy(max_depth_ceiling="names_types")
    assert ep.apply_depth_ceiling("names_types_labels_summary", ent) == "names_types"
    # Enterprise ceiling looser than session's own — session's stricter value wins.
    ent2 = ep.EnterprisePolicy(max_depth_ceiling="names_types_labels_summary")
    assert ep.apply_depth_ceiling("names_only", ent2) == "names_only"


def test_apply_banned_variables_unions_never_shrinks():
    session = frozenset({"a"})
    ent = ep.EnterprisePolicy(never_expose_fields=frozenset({"b", "c"}))
    assert ep.apply_banned_variables(session, ent) == frozenset({"a", "b", "c"})
    assert ep.apply_banned_variables(session, None) == session


def test_apply_privacy_profile_floor_takes_stricter():
    ent = ep.EnterprisePolicy(min_privacy_profile="confidential")
    assert ep.apply_privacy_profile_floor("public", ent) == "confidential"
    assert ep.apply_privacy_profile_floor("regulated", ent) == "regulated"


def test_apply_sdc_floor_raises_thresholds_only():
    ent = ep.EnterprisePolicy(
        min_cell_suppression_threshold=50,
        min_n_regression=5,  # LOWER than default (10) -- must not lower it
    )
    out = ep.apply_sdc_floor(DEFAULT_CONFIG, ent)
    assert out.cell_suppression_threshold == 50  # raised
    assert out.min_n_regression == DEFAULT_CONFIG.min_n_regression  # untouched, not lowered


def test_apply_sdc_floor_clamps_epsilon_down_never_up():
    cfg = SDCConfig(dp_epsilon=2.0)
    ent = ep.EnterprisePolicy(dp_epsilon_ceiling=0.5)
    out = ep.apply_sdc_floor(cfg, ent)
    assert out.dp_epsilon == 0.5

    # A ceiling looser than the session's own choice must not raise it.
    cfg2 = SDCConfig(dp_epsilon=0.1)
    ent2 = ep.EnterprisePolicy(dp_epsilon_ceiling=2.0)
    out2 = ep.apply_sdc_floor(cfg2, ent2)
    assert out2.dp_epsilon == 0.1


def test_apply_sdc_floor_kills_dp_when_disallowed():
    cfg = SDCConfig(dp_epsilon=0.5)
    ent = ep.EnterprisePolicy(allow_differential_privacy=False)
    out = ep.apply_sdc_floor(cfg, ent)
    assert out.dp_epsilon is None


def test_apply_sdc_floor_cannot_turn_dp_on():
    # Session never opted into DP (dp_epsilon is None). An enterprise
    # ceiling must not turn it on -- that would change the mechanism
    # that runs, not just how strict it is.
    cfg = DEFAULT_CONFIG
    assert cfg.dp_epsilon is None
    ent = ep.EnterprisePolicy(dp_epsilon_ceiling=0.5, allow_differential_privacy=True)
    out = ep.apply_sdc_floor(cfg, ent)
    assert out.dp_epsilon is None


def test_apply_sdc_floor_unions_never_expose_into_banned():
    cfg = SDCConfig(banned_variables=frozenset({"x"}))
    ent = ep.EnterprisePolicy(never_expose_fields=frozenset({"y"}))
    out = ep.apply_sdc_floor(cfg, ent)
    assert out.banned_variables == frozenset({"x", "y"})


def test_apply_sdc_floor_none_enterprise_is_true_noop():
    cfg = SDCConfig(min_n_regression=17, dp_epsilon=0.3)
    out = ep.apply_sdc_floor(cfg, None)
    assert out is cfg  # identity, not just equality -- confirms zero-op fast path


def test_integration_allowlists_are_restrictive() -> None:
    policy = ep.EnterprisePolicy(
        allowed_model_providers=frozenset({"openai"}),
        allowed_database_backends=frozenset({"sqlite"}),
    )
    assert ep.model_provider_allowed("openai", policy) is True
    assert ep.model_provider_allowed("gemini", policy) is False
    assert ep.database_backend_allowed("sqlite", policy) is True
    assert ep.database_backend_allowed("snowflake", policy) is False
    assert ep.model_provider_allowed("gemini", None) is True
    assert ep.database_backend_allowed("snowflake", None) is True


def test_endpoint_and_region_restrictions_fail_closed() -> None:
    policy = ep.EnterprisePolicy(
        allowed_endpoint_hosts=frozenset({"api.approved.example", "*.lab.example"}),
        allowed_regions=frozenset({"us-west-2"}),
    )
    assert ep.integration_endpoint_allowed(
        "https://api.approved.example/v1", policy
    ) is True
    assert ep.integration_endpoint_allowed(
        "https://node.lab.example/v1", policy
    ) is True
    assert ep.integration_endpoint_allowed(
        "https://lab.example/v1", policy
    ) is False
    assert ep.integration_endpoint_allowed(
        "https://evil.example/v1", policy
    ) is False
    assert ep.integration_endpoint_allowed("not-a-url", policy) is False
    assert ep.integration_region_allowed("us-west-2", policy) is True
    assert ep.integration_region_allowed("us-east-1", policy) is False
    assert ep.integration_region_allowed(None, policy) is False


def test_local_only_integrations_allow_only_validated_local_targets(monkeypatch) -> None:
    policy = ep.EnterprisePolicy(require_local_integrations=True)
    assert ep.integration_endpoint_allowed(None, policy, local_hint=True) is True
    assert ep.integration_endpoint_allowed(
        "http://127.0.0.1:11434/v1", policy
    ) is True
    assert ep.integration_endpoint_allowed(
        "https://gateway.example/v1", policy
    ) is False

    monkeypatch.setenv(
        "SIFT_OPENAI_COMPATIBLE_BASE_URL", "http://127.0.0.1:11434/v1"
    )
    assert ep.model_provider_allowed("openai_compatible", policy) is True
    assert ep.model_provider_allowed("openai", policy) is False


# Property-based: for ANY session config and ANY enterprise policy,
# apply_sdc_floor must never produce a config that is LOOSER than the
# input on any threshold, and must never turn DP on if it was off.
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    min_n_regression=st.integers(min_value=1, max_value=500),
    min_n_descriptive=st.integers(min_value=1, max_value=500),
    min_n_ttest_group=st.integers(min_value=1, max_value=500),
    cell_suppression_threshold=st.integers(min_value=1, max_value=500),
    dp_epsilon=st.one_of(st.none(), st.floats(min_value=0.01, max_value=10, allow_nan=False)),
    ent_min_n_regression=st.one_of(st.none(), st.integers(min_value=1, max_value=500)),
    ent_cell_threshold=st.one_of(st.none(), st.integers(min_value=1, max_value=500)),
    ent_epsilon_ceiling=st.one_of(st.none(), st.floats(min_value=0.01, max_value=10, allow_nan=False)),
    ent_allow_dp=st.booleans(),
)
def test_apply_sdc_floor_never_loosens(
    min_n_regression, min_n_descriptive, min_n_ttest_group,
    cell_suppression_threshold, dp_epsilon,
    ent_min_n_regression, ent_cell_threshold, ent_epsilon_ceiling, ent_allow_dp,
):
    cfg = SDCConfig(
        min_n_regression=min_n_regression,
        min_n_descriptive=min_n_descriptive,
        min_n_ttest_group=min_n_ttest_group,
        cell_suppression_threshold=cell_suppression_threshold,
        dp_epsilon=dp_epsilon,
    )
    ent = ep.EnterprisePolicy(
        min_n_regression=ent_min_n_regression,
        min_cell_suppression_threshold=ent_cell_threshold,
        dp_epsilon_ceiling=ent_epsilon_ceiling,
        allow_differential_privacy=ent_allow_dp,
    )
    out = ep.apply_sdc_floor(cfg, ent)
    assert out.min_n_regression >= cfg.min_n_regression
    assert out.min_n_descriptive >= cfg.min_n_descriptive
    assert out.min_n_ttest_group >= cfg.min_n_ttest_group
    assert out.cell_suppression_threshold >= cfg.cell_suppression_threshold
    if cfg.dp_epsilon is None:
        assert out.dp_epsilon is None  # never turned on
    elif not ent_allow_dp:
        assert out.dp_epsilon is None  # killed
    else:
        assert out.dp_epsilon <= cfg.dp_epsilon  # never raised


# ---------------------------------------------------------------------------
# tools.py integration — schema depth ceiling + banned variables
# ---------------------------------------------------------------------------

def test_get_schema_respects_enterprise_depth_ceiling(clean_env, tmp_path, monkeypatch):
    import asyncio
    import json
    from sift.config import set_cwd
    from sift.tools import get_schema

    set_cwd(tmp_path)
    (tmp_path / "d.csv").write_text("age,income\n20,1\n30,2\n40,3\n")

    ent_file = _write(tmp_path, """
version: 1
max_depth_ceiling: names_only
""")
    monkeypatch.setenv(ep.ENV_VAR, str(ent_file))

    result = asyncio.run(get_schema.handler({
        "dataset": "d.csv", "depth": "names_types_labels_summary",
    }))
    body = json.loads(result["content"][0]["text"])
    assert body["status"] == "denied"
    assert "names_only" in body["reason"]


def test_get_schema_unaffected_when_no_enterprise_policy(clean_env, tmp_path):
    import asyncio
    import json
    from sift.config import set_cwd
    from sift.tools import get_schema

    set_cwd(tmp_path)
    (tmp_path / "d.csv").write_text("age\n20\n30\n40\n")

    result = asyncio.run(get_schema.handler({
        "dataset": "d.csv", "depth": "names_types",
    }))
    body = json.loads(result["content"][0]["text"])
    assert body["status"] != "denied"


def test_get_schema_strips_enterprise_never_expose_field(clean_env, tmp_path, monkeypatch):
    import asyncio
    import json
    from sift.config import set_cwd
    from sift.tools import get_schema

    set_cwd(tmp_path)
    (tmp_path / "d.csv").write_text("age,ssn\n20,1\n30,2\n40,3\n")

    ent_file = _write(tmp_path, """
version: 1
never_expose_fields:
  - ssn
""")
    monkeypatch.setenv(ep.ENV_VAR, str(ent_file))

    result = asyncio.run(get_schema.handler({
        "dataset": "d.csv", "depth": "names_types",
    }))
    body = json.loads(result["content"][0]["text"])
    var_names = [v.get("name") for v in body.get("variables", [])]
    assert "age" in var_names
    assert "ssn" not in var_names


def test_get_schema_strips_enterprise_never_expose_field_despite_case_mismatch(
    clean_env, tmp_path, monkeypatch,
):
    """Audit pass 2 finding: the admin's YAML floor (``never_expose_
    fields``) is documented as impossible for a session to loosen --
    but before this fix, a case mismatch between the YAML entry and
    the real dataset column silently loosened it completely, with no
    error anywhere. Real dataset column is uppercase "SSN"; the
    admin's YAML (as an admin copying a data dictionary might type
    it) is lowercase "ssn"."""
    import asyncio
    import json
    from sift.config import set_cwd
    from sift.tools import get_schema

    set_cwd(tmp_path)
    (tmp_path / "d.csv").write_text("AGE,SSN\n20,1\n30,2\n40,3\n")

    ent_file = _write(tmp_path, """
version: 1
never_expose_fields:
  - ssn
""")
    monkeypatch.setenv(ep.ENV_VAR, str(ent_file))

    result = asyncio.run(get_schema.handler({
        "dataset": "d.csv", "depth": "names_types",
    }))
    body = json.loads(result["content"][0]["text"])
    var_names = [v.get("name") for v in body.get("variables", [])]
    assert "AGE" in var_names
    assert "SSN" not in var_names, (
        "enterprise never_expose_fields listed \"ssn\" but the real "
        "column \"SSN\" (different case) was NOT dropped -- the "
        "admin-controlled floor was silently defeated by the case "
        "mismatch"
    )


# ---------------------------------------------------------------------------
# tools.py integration — SDC floor applied to _resolve_sdc_and_source_n
# ---------------------------------------------------------------------------

def test_enterprise_never_expose_override_is_stable_with_trailing_newline(
    clean_env, tmp_path, monkeypatch,
):
    """The admin-controlled floor must win over a researcher's own
    per-dataset opt-in, not just over the defaults: a dataset policy
    can opt a variable into ``non_disclosive_variables`` (real min/max
    via request_data's numeric_bounds -- see data_request.py), but if
    the SAME field is also on the enterprise's ``never_expose_fields``
    list, the request must still be denied outright. This is not a
    new code path -- ``_check_not_banned`` already runs before
    ``_numeric_bounds`` dispatch, and ``apply_sdc_floor`` already
    unions ``never_expose_fields`` into ``banned_variables``
    unconditionally -- this test exists to PROVE that composition
    holds end-to-end through the real request_data tool, now that
    non_disclosive_variables actually does something for the first
    time."""
    import asyncio
    import json
    from sift.config import set_cwd
    from sift.policy import DatasetPolicy, SiftPolicy, save_policy
    from sift.tools import request_data

    set_cwd(tmp_path)
    (tmp_path / "d.csv").write_text(
        "age\n" + "\n".join(str(20 + i) for i in range(40)) + "\n"
    )
    save_policy(tmp_path, SiftPolicy(datasets={
        "d.csv": DatasetPolicy(non_disclosive_variables=("age",)),
    }))
    ent_file = _write(tmp_path, """
version: 1
never_expose_fields:
  - age
""")
    monkeypatch.setenv(ep.ENV_VAR, str(ent_file))

    result = asyncio.run(request_data.handler({
        "dataset": "d.csv", "request_type": "numeric_bounds",
        "variable": "age",
    }))
    body = json.loads(result["content"][0]["text"])
    assert body["status"] == "denied", (
        "dataset policy opted \"age\" into non_disclosive_variables, "
        "but the enterprise floor also lists it in never_expose_fields "
        "-- the admin-controlled ban must win outright, not just "
        "suppress the exact-bounds fields"
    )
    assert "banned" in body["reason"].lower()


def test_enterprise_never_expose_overrides_dataset_non_disclosive_opt_in(
    clean_env, tmp_path, monkeypatch,
):
    """The admin-controlled floor must win over a researcher's own
    per-dataset opt-in, not just over the defaults: a dataset policy
    can opt a variable into ``non_disclosive_variables`` (real min/max
    via request_data's numeric_bounds -- see data_request.py), but if
    the SAME field is also on the enterprise's ``never_expose_fields``
    list, the request must still be denied outright. This is not a
    new code path -- ``_check_not_banned`` already runs before
    ``_numeric_bounds`` dispatch, and ``apply_sdc_floor`` already
    unions ``never_expose_fields`` into ``banned_variables``
    unconditionally -- this test exists to PROVE that composition
    holds end-to-end through the real request_data tool, now that
    non_disclosive_variables actually does something for the first
    time."""
    import asyncio
    import json
    from sift.config import set_cwd
    from sift.policy import DatasetPolicy, SiftPolicy, save_policy
    from sift.tools import request_data

    set_cwd(tmp_path)
    (tmp_path / "d.csv").write_text(
        "age\n" + "\n".join(str(20 + i) for i in range(40)) + "\n",
    )
    save_policy(tmp_path, SiftPolicy(datasets={
        "d.csv": DatasetPolicy(non_disclosive_variables=("age",)),
    }))
    ent_file = _write(tmp_path, """
version: 1
never_expose_fields:
  - age
""")
    monkeypatch.setenv(ep.ENV_VAR, str(ent_file))

    result = asyncio.run(request_data.handler({
        "dataset": "d.csv", "request_type": "numeric_bounds",
        "variable": "age",
    }))
    body = json.loads(result["content"][0]["text"])
    assert body["status"] == "denied", (
        'dataset policy opted "age" into non_disclosive_variables, '
        'but the enterprise floor also lists it in never_expose_fields '
        '-- the admin-controlled ban must win outright, not just '
        'suppress the exact-bounds fields'
    )
    assert "banned" in body["reason"].lower()


def test_resolve_sdc_and_source_n_applies_enterprise_floor(clean_env, tmp_path, monkeypatch):
    from sift.config import set_cwd
    from sift.tools import _resolve_sdc_and_source_n

    set_cwd(tmp_path)
    ent_file = _write(tmp_path, """
version: 1
min_cell_suppression_threshold: 77
never_expose_fields:
  - secret_col
""")
    monkeypatch.setenv(ep.ENV_VAR, str(ent_file))

    sdc_cfg, source_n, audit_seconds, budget_status = _resolve_sdc_and_source_n(
        tmp_path, None,
    )
    assert sdc_cfg.cell_suppression_threshold == 77
    assert "secret_col" in sdc_cfg.banned_variables


def test_resolve_sdc_and_source_n_unaffected_without_enterprise_policy(
    clean_env, tmp_path,
):
    from sift.config import set_cwd
    from sift.tools import _resolve_sdc_and_source_n
    from sift.sanitizer import DEFAULT_CONFIG

    set_cwd(tmp_path)
    sdc_cfg, source_n, audit_seconds, budget_status = _resolve_sdc_and_source_n(
        tmp_path, None,
    )
    assert sdc_cfg.cell_suppression_threshold == DEFAULT_CONFIG.cell_suppression_threshold


def test_resolve_sdc_and_source_n_applies_enterprise_privacy_profile_floor(
    clean_env, tmp_path, monkeypatch,
):
    """Regression test for architecture-audit finding F:
    ``apply_privacy_profile_floor`` was implemented and unit-tested in
    isolation but never actually called from a production code path,
    so an enterprise ``min_privacy_profile`` setting had zero real
    effect -- ``apply_sdc_floor`` (the function that IS wired in)
    never reads that field either. This exercises the real
    ``_resolve_sdc_and_source_n`` call with a genuine dataset and
    confirms an enterprise floor of "regulated" actually narrows the
    adaptive-suppression budget to the "regulated" allowance (15),
    not the session-default "internal" allowance (150) a dataset with
    no explicit policy entry would otherwise get.
    """
    from sift.config import set_cwd
    from sift.policy import PRIVACY_BUDGET_BY_PROFILE
    from sift.tools import _resolve_sdc_and_source_n

    set_cwd(tmp_path)
    (tmp_path / "d.csv").write_text("age,income\n20,1\n30,2\n40,3\n")
    ent_file = _write(tmp_path, "version: 1\nmin_privacy_profile: regulated\n")
    monkeypatch.setenv(ep.ENV_VAR, str(ent_file))

    sdc_cfg, source_n, audit_seconds, budget_status = _resolve_sdc_and_source_n(
        tmp_path, "d.csv",
    )
    assert budget_status is not None
    assert budget_status.privacy_profile == "regulated"
    assert budget_status.budget == PRIVACY_BUDGET_BY_PROFILE["regulated"]
    assert budget_status.budget != PRIVACY_BUDGET_BY_PROFILE["internal"]


def test_resolve_sdc_and_source_n_privacy_profile_floor_never_loosens(
    clean_env, tmp_path, monkeypatch,
):
    """A dataset already under a stricter session-derived profile than
    the enterprise floor must keep its own (stricter) profile --
    ``apply_privacy_profile_floor`` only ever raises, never lowers."""
    from sift.config import set_cwd
    from sift.policy import PRIVACY_BUDGET_BY_PROFILE, SiftPolicy, DatasetPolicy, save_policy
    from sift.tools import _resolve_sdc_and_source_n

    set_cwd(tmp_path)
    (tmp_path / "d.csv").write_text("age,income\n20,1\n30,2\n40,3\n")
    save_policy(tmp_path, SiftPolicy(
        datasets={"d.csv": DatasetPolicy(privacy_profile="regulated")},
    ))
    ent_file = _write(tmp_path, "version: 1\nmin_privacy_profile: internal\n")
    monkeypatch.setenv(ep.ENV_VAR, str(ent_file))

    sdc_cfg, source_n, audit_seconds, budget_status = _resolve_sdc_and_source_n(
        tmp_path, "d.csv",
    )
    assert budget_status is not None
    assert budget_status.privacy_profile == "regulated"
    assert budget_status.budget == PRIVACY_BUDGET_BY_PROFILE["regulated"]


# ---------------------------------------------------------------------------
# ui.py integration — export approval gate
# ---------------------------------------------------------------------------

def test_export_blocked_and_request_recorded_when_approval_required(
    clean_env, tmp_path, monkeypatch,
):
    from sift.ui import SiftBridge

    ent_file = _write(tmp_path, """
version: 1
export_approval_required: true
""")
    monkeypatch.setenv(ep.ENV_VAR, str(ent_file))

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    bridge = SiftBridge()
    bridge.cwd = session_dir

    result = bridge.export_codebook()
    assert result["ok"] is False
    assert result.get("pending_approval") is True
    assert "request_id" in result
    # The export file must NOT have been written.
    assert not (session_dir / "exports").exists() or not any(
        (session_dir / "exports").glob("codebook_*")
    )

    requests = ep.list_export_requests(session_dir)
    assert len(requests) == 1
    assert requests[0]["export_kind"] == "codebook"
    assert requests[0]["status"] == "pending"


def test_export_proceeds_after_approval(clean_env, tmp_path, monkeypatch):
    from sift.ui import SiftBridge

    ent_file = _write(tmp_path, """
version: 1
export_approval_required: true
""")
    monkeypatch.setenv(ep.ENV_VAR, str(ent_file))

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    bridge = SiftBridge()
    bridge.cwd = session_dir

    first = bridge.export_codebook()
    assert first["ok"] is False
    req_id = first["request_id"]

    approval = bridge.approve_export(req_id)
    assert approval["ok"] is True

    second = bridge.export_codebook()
    assert second.get("ok", True) is not False or "display_path" in second
    # More directly: no longer pending_approval, and it actually wrote a file.
    assert second.get("pending_approval") is not True
    assert "display_path" in second
    assert Path(second["display_path"]).is_file()


def test_export_unaffected_when_no_enterprise_policy(clean_env, tmp_path):
    from sift.ui import SiftBridge

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    bridge = SiftBridge()
    bridge.cwd = session_dir

    result = bridge.export_codebook()
    assert result.get("pending_approval") is not True


def test_export_unaffected_when_enterprise_policy_present_but_not_requiring_approval(
    clean_env, tmp_path, monkeypatch,
):
    from sift.ui import SiftBridge

    ent_file = _write(tmp_path, """
version: 1
max_depth_ceiling: names_types_labels
export_approval_required: false
""")
    monkeypatch.setenv(ep.ENV_VAR, str(ent_file))

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    bridge = SiftBridge()
    bridge.cwd = session_dir

    result = bridge.export_codebook()
    assert result.get("pending_approval") is not True


def test_approve_export_rejects_unknown_id(clean_env, tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    result = ep.approve_export(session_dir, "not-a-real-id")
    assert result["ok"] is False


def test_approve_export_rejects_double_approval(clean_env, tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    record = ep.request_export_approval(session_dir, "codebook")
    first = ep.approve_export(session_dir, record["id"])
    assert first["ok"] is True
    second = ep.approve_export(session_dir, record["id"])
    assert second["ok"] is False


def test_pending_export_request_is_reused_for_unchanged_session(
    clean_env, tmp_path,
):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    first = ep.request_export_approval(session_dir, "codebook")
    second = ep.request_export_approval(session_dir, "codebook")
    assert second["id"] == first["id"]
    assert len(ep.list_export_requests(session_dir)) == 1


def test_export_approval_becomes_stale_when_session_material_changes(
    clean_env, tmp_path,
):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    dataset = session_dir / "d.csv"
    dataset.write_text("x\n1\n", encoding="utf-8")

    record = ep.request_export_approval(session_dir, "codebook")
    dataset.write_text("x\n1\n2\n", encoding="utf-8")

    approval = ep.approve_export(session_dir, record["id"])
    assert approval["ok"] is False
    assert "changed" in approval["reason"]
    assert ep.is_export_approved(session_dir, "codebook") is False


def test_approved_export_cannot_be_reused_after_later_session_change(
    clean_env, tmp_path,
):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    dataset = session_dir / "d.csv"
    dataset.write_text("x\n1\n", encoding="utf-8")

    record = ep.request_export_approval(session_dir, "codebook")
    assert ep.approve_export(session_dir, record["id"])["ok"] is True
    assert ep.is_export_approved(session_dir, "codebook") is True

    dataset.write_text("x\n1\n2\n", encoding="utf-8")
    assert ep.is_export_approved(session_dir, "codebook") is False


# ---------------------------------------------------------------------------
# env-var trust gate — the fix for the researcher-authored-floor bypass
# ---------------------------------------------------------------------------

def test_researcher_owned_env_file_is_rejected_fail_closed(
    clean_env, tmp_path, monkeypatch,
):
    """The bug this closes: without the trust gate, a researcher could
    set SIFT_ENTERPRISE_POLICY to point at an empty file THEY wrote
    (e.g. in their own home directory) and get a fully-permissive
    EnterprisePolicy back -- defeating the module's entire "floor a
    researcher cannot lower" guarantee. With the gate active (turned
    back on here, undoing the ``clean_env`` fixture's testing bypass),
    a file the current process owns and can write must fail closed
    exactly like a corrupted/unparseable file does, even though its
    content parses as a perfectly well-formed, permissive policy.
    """
    monkeypatch.setattr(ep, "_TESTING_TRUST_ENV_PATH_UNCONDITIONALLY", False)
    p = _write(tmp_path, "")  # well-formed, empty -> would be fully permissive
    monkeypatch.setenv(ep.ENV_VAR, str(p))

    result = ep.load_enterprise_policy()

    assert result is not None
    assert result.export_approval_required is True
    assert result.max_depth_ceiling == "names_only"
    assert result.source_path == "<unreadable enterprise policy file>"


def test_researcher_owned_env_file_with_real_content_still_rejected(
    clean_env, tmp_path, monkeypatch,
):
    """Same gate, but confirms it fires on content-bearing files too,
    not just empty ones -- a researcher authoring a *plausible-looking*
    permissive policy (rather than just an empty file) must not fare
    any better."""
    monkeypatch.setattr(ep, "_TESTING_TRUST_ENV_PATH_UNCONDITIONALLY", False)
    p = _write(tmp_path, """
version: 1
max_depth_ceiling: full
export_approval_required: false
""")
    monkeypatch.setenv(ep.ENV_VAR, str(p))

    result = ep.load_enterprise_policy()

    assert result is not None
    assert result.export_approval_required is True


def test_env_path_is_trustworthy_false_for_own_file(tmp_path):
    p = tmp_path / "mine.yaml"
    p.write_text("", encoding="utf-8")
    assert ep._env_path_is_trustworthy(p) is False


def test_env_path_is_trustworthy_false_for_missing_file(tmp_path):
    assert ep._env_path_is_trustworthy(tmp_path / "nope.yaml") is False


def test_env_path_is_trustworthy_true_for_non_writable_non_owned_file(
    tmp_path, monkeypatch,
):
    """Simulates a genuinely admin-authored file: not owned by the
    current uid, and not writable by the current process. Since tests
    can't actually create a root-owned file in a sandbox, this
    monkeypatches ``os.getuid``/``os.access`` at the point ``_env_path_
    is_trustworthy`` calls them, rather than faking filesystem
    ownership directly."""
    p = tmp_path / "admin.yaml"
    p.write_text("version: 1\n", encoding="utf-8")
    file_owner = p.stat().st_uid
    monkeypatch.setattr(
        ep.os, "getuid", lambda: file_owner + 1, raising=False,
    )
    monkeypatch.setattr(ep.os, "access", lambda *a, **k: False)
    assert ep._env_path_is_trustworthy(p) is True


def test_system_path_resolution_unaffected_by_trust_gate(
    clean_env, tmp_path, monkeypatch,
):
    """The trust gate only applies to env-var resolution -- a file
    found via ``_SYSTEM_PATHS`` (which this test simulates by pointing
    the fixed-path tuple at a test file the process owns) must load
    normally, since ``_SYSTEM_PATHS`` protection comes from the OS,
    not from this gate, and the gate must not double-restrict it."""
    monkeypatch.setattr(ep, "_TESTING_TRUST_ENV_PATH_UNCONDITIONALLY", False)
    p = _write(tmp_path, "version: 1\nexport_approval_required: false\n")
    monkeypatch.setattr(ep, "_SYSTEM_PATHS", (p,))

    result = ep.load_enterprise_policy()

    assert result is not None
    assert result.export_approval_required is False
    assert result.source_path == str(p)
