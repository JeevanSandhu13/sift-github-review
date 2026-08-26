"""Per-dataset privacy profiles (public/internal/confidential/regulated).

A profile is a human-scale classification that maps to its own
schema-depth ceiling. The authoritative enforced ceiling —
``effective_max_depth()`` — is always the STRICTER of a dataset's
``max_depth`` field and its profile's ceiling, never either alone.
These tests cover: the combination logic itself, the fail-open-on-
absence / fail-closed-on-corruption split for the new field (a real
bug caught during development — see below), the tool-layer wiring
(``get_schema`` actually enforces the combined ceiling), and the UI
bridge methods that let a researcher set a profile without
disturbing the other, independent policy axes.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from sift.config import set_cwd
from sift.policy import (
    DEFAULT_PRIVACY_PROFILE,
    FAIL_CLOSED_PRIVACY_PROFILE,
    PRIVACY_PROFILES,
    PROFILE_MAX_DEPTH_CEILING,
    DatasetPolicy,
    SiftPolicy,
    effective_max_depth,
    get_privacy_profile,
    load_policy,
    save_policy,
)
from sift.tools import get_schema
from sift.ui import SiftBridge


def _call_get_schema(args: dict) -> dict:
    envelope = asyncio.run(get_schema.handler(args))
    text_block = envelope["content"][0]["text"]
    return json.loads(text_block)


# ---------------------------------------------------------------------------
# effective_max_depth() — the combination logic
# ---------------------------------------------------------------------------

def test_all_four_profiles_have_a_ceiling_defined():
    for p in PRIVACY_PROFILES:
        assert p in PROFILE_MAX_DEPTH_CEILING


def test_profile_stricter_than_max_depth_wins():
    policy = SiftPolicy(datasets={
        "d.csv": DatasetPolicy(
            max_depth="names_types_labels_summary",  # richest
            privacy_profile="regulated",               # strictest
        ),
    })
    assert effective_max_depth(policy, "d.csv") == "names_only"


def test_max_depth_stricter_than_profile_wins():
    policy = SiftPolicy(datasets={
        "d.csv": DatasetPolicy(
            max_depth="names_only",       # strictest
            privacy_profile="public",      # loosest
        ),
    })
    assert effective_max_depth(policy, "d.csv") == "names_only"


def test_public_and_internal_do_not_add_restriction():
    for profile in ("public", "internal"):
        policy = SiftPolicy(datasets={
            "d.csv": DatasetPolicy(
                max_depth="names_types_labels_summary",
                privacy_profile=profile,
            ),
        })
        assert effective_max_depth(policy, "d.csv") == \
            "names_types_labels_summary"


def test_confidential_blocks_summary_tier():
    policy = SiftPolicy(datasets={
        "d.csv": DatasetPolicy(
            max_depth="names_types_labels_summary",
            privacy_profile="confidential",
        ),
    })
    assert effective_max_depth(policy, "d.csv") == "names_types_labels"


def test_dataset_with_no_entry_uses_default_profile():
    policy = SiftPolicy()  # no datasets at all
    assert get_privacy_profile(policy, "unknown.csv") == DEFAULT_PRIVACY_PROFILE
    assert effective_max_depth(policy, "unknown.csv") == \
        policy.default_max_depth


# ---------------------------------------------------------------------------
# load_policy(): absence-vs-corruption split — the bug caught during dev
# ---------------------------------------------------------------------------

def test_missing_privacy_profile_key_defaults_to_internal_not_fail_closed(
    tmp_path: Path,
):
    """A policy.json entry with NO privacy_profile key at all (every
    entry written before this feature existed, or by a researcher who
    only ever touched the depth chip) must default to
    DEFAULT_PRIVACY_PROFILE, not fail closed. This was a real bug:
    the first implementation used the fail-closed default for
    absence, which would have silently clamped every pre-existing
    policy.json entry to "regulated" (names_only) the moment this
    code shipped."""
    p = tmp_path / ".sift"
    p.mkdir()
    (p / "policy.json").write_text(json.dumps({
        "version": 1,
        "default_max_depth": "names_types_labels_summary",
        "datasets": {
            "old.csv": {
                "max_depth": "names_types_labels",
                "set_at": "2026-01-01T00:00:00+00:00",
                # deliberately no "privacy_profile" key
            },
        },
    }))
    policy = load_policy(tmp_path)
    assert get_privacy_profile(policy, "old.csv") == "internal"
    assert effective_max_depth(policy, "old.csv") == "names_types_labels"


def test_invalid_privacy_profile_value_fails_closed(tmp_path: Path):
    """A PRESENT but unrecognised profile string (a future version's
    profile name, a hand-edit typo) is a genuine corruption signal
    and must clamp to the strictest tier — the opposite handling
    from mere absence, and the important distinction the previous
    test pins."""
    p = tmp_path / ".sift"
    p.mkdir()
    (p / "policy.json").write_text(json.dumps({
        "version": 1,
        "default_max_depth": "names_types_labels_summary",
        "datasets": {
            "weird.csv": {
                "max_depth": "names_types_labels_summary",
                "set_at": "2026-01-01T00:00:00+00:00",
                "privacy_profile": "top-secret",  # not a real profile
            },
        },
    }))
    policy = load_policy(tmp_path)
    assert get_privacy_profile(policy, "weird.csv") == FAIL_CLOSED_PRIVACY_PROFILE
    assert effective_max_depth(policy, "weird.csv") == "names_only"


def test_save_then_load_round_trips_a_non_default_profile(tmp_path: Path):
    save_policy(tmp_path, SiftPolicy(datasets={
        "d.csv": DatasetPolicy(privacy_profile="confidential"),
    }))
    reloaded = load_policy(tmp_path)
    assert get_privacy_profile(reloaded, "d.csv") == "confidential"


def test_save_omits_the_field_when_default_for_file_tidiness(tmp_path: Path):
    save_policy(tmp_path, SiftPolicy(datasets={
        "d.csv": DatasetPolicy(privacy_profile=DEFAULT_PRIVACY_PROFILE),
    }))
    raw = json.loads((tmp_path / ".sift" / "policy.json").read_text(encoding="utf-8"))
    assert "privacy_profile" not in raw["datasets"]["d.csv"]


# ---------------------------------------------------------------------------
# Tool-layer wiring: get_schema actually enforces the combined ceiling
# ---------------------------------------------------------------------------

def test_get_schema_enforces_regulated_profile_even_with_loose_max_depth(
    tmp_path: Path,
):
    set_cwd(tmp_path)
    csv = tmp_path / "patients.csv"
    csv.write_text("age,diagnosis\n40,x\n50,y\n60,z\n")
    save_policy(tmp_path, SiftPolicy(datasets={
        "patients.csv": DatasetPolicy(
            max_depth="names_types_labels_summary",  # would otherwise allow
            privacy_profile="regulated",
        ),
    }))
    resp = _call_get_schema({
        "dataset": "patients.csv", "depth": "names_types_labels_summary",
    })
    assert resp["status"] == "denied"

    resp_ok = _call_get_schema({
        "dataset": "patients.csv", "depth": "names_only",
    })
    assert resp_ok["status"] == "ok"


def test_get_schema_public_profile_does_not_restrict(tmp_path: Path):
    set_cwd(tmp_path)
    csv = tmp_path / "open.csv"
    csv.write_text("x,y\n1,2\n3,4\n")
    save_policy(tmp_path, SiftPolicy(datasets={
        "open.csv": DatasetPolicy(
            max_depth="names_types_labels_summary",
            privacy_profile="public",
        ),
    }))
    resp = _call_get_schema({
        "dataset": "open.csv", "depth": "names_types_labels_summary",
    })
    assert resp["status"] == "ok"


# ---------------------------------------------------------------------------
# UI bridge: set_dataset_privacy_profile + cross-axis preservation
# ---------------------------------------------------------------------------

def test_set_dataset_privacy_profile_persists(tmp_path: Path):
    bridge = SiftBridge()
    bridge.cwd = tmp_path
    result = bridge.set_dataset_privacy_profile("d.csv", "confidential")
    assert result["ok"] is True
    policy = load_policy(tmp_path)
    assert get_privacy_profile(policy, "d.csv") == "confidential"


def test_set_dataset_privacy_profile_rejects_unknown_value(tmp_path: Path):
    bridge = SiftBridge()
    bridge.cwd = tmp_path
    result = bridge.set_dataset_privacy_profile("d.csv", "ultra-classified")
    assert result["ok"] is False


def test_set_dataset_privacy_profile_preserves_max_depth(tmp_path: Path):
    """Setting the privacy profile must not disturb an existing,
    independently-set max_depth — the two axes are orthogonal, same
    principle already pinned for non_disclosive_variables."""
    save_policy(tmp_path, SiftPolicy(datasets={
        "d.csv": DatasetPolicy(max_depth="names_types_labels"),
    }))
    bridge = SiftBridge()
    bridge.cwd = tmp_path
    bridge.set_dataset_privacy_profile("d.csv", "confidential")
    policy = load_policy(tmp_path)
    entry = policy.datasets["d.csv"]
    assert entry.max_depth == "names_types_labels"
    assert entry.privacy_profile == "confidential"


def test_set_dataset_policy_preserves_privacy_profile_on_depth_change(
    tmp_path: Path,
):
    """The reverse direction of the previous test: changing the depth
    chip must not silently reset a previously-set privacy profile
    back to the default — the exact regression class this module's
    existing non_disclosive_variables preservation logic already
    guards against, now extended to the new field."""
    save_policy(tmp_path, SiftPolicy(
        default_max_depth="names_types_labels_summary",
        datasets={
            "d.csv": DatasetPolicy(
                max_depth="names_types_labels",
                privacy_profile="regulated",
            ),
        },
    ))
    bridge = SiftBridge()
    bridge.cwd = tmp_path

    # Tighten further — profile must survive.
    bridge.set_dataset_policy("d.csv", "names_only")
    after = load_policy(tmp_path)
    assert after.datasets["d.csv"].privacy_profile == "regulated"

    # Revert depth to the file-wide default — profile must STILL
    # survive (and keep the entry alive) even though pre-fix the
    # entry would have been dropped entirely at this point.
    bridge.set_dataset_policy(
        "d.csv", after.default_max_depth,
    )
    after_revert = load_policy(tmp_path)
    assert "d.csv" in after_revert.datasets
    assert after_revert.datasets["d.csv"].privacy_profile == "regulated"
    # And the effective ceiling reflects the surviving profile, not
    # just the reverted-to-default max_depth.
    assert effective_max_depth(after_revert, "d.csv") == "names_only"


def test_set_dataset_privacy_profile_collapses_entry_at_full_default(
    tmp_path: Path,
):
    """Setting the profile back to 'internal' with nothing else
    explicit should drop the entry entirely — same tidiness rule
    set_dataset_policy already applies."""
    save_policy(tmp_path, SiftPolicy(datasets={
        "d.csv": DatasetPolicy(privacy_profile="confidential"),
    }))
    bridge = SiftBridge()
    bridge.cwd = tmp_path
    bridge.set_dataset_privacy_profile("d.csv", DEFAULT_PRIVACY_PROFILE)
    policy = load_policy(tmp_path)
    assert "d.csv" not in policy.datasets


def test_policy_summary_reports_effective_ceiling_not_raw_max_depth(
    tmp_path: Path,
):
    """_policy_summary's 'ceiling' field must reflect the ACTUAL
    enforced ceiling (profile-combined), not just the raw max_depth —
    otherwise the UI footer would show the researcher a looser number
    than what get_schema actually enforces."""
    (tmp_path / "d.csv").write_text("x,y\n1,2\n3,4\n")
    save_policy(tmp_path, SiftPolicy(datasets={
        "d.csv": DatasetPolicy(
            max_depth="names_types_labels_summary",
            privacy_profile="regulated",
        ),
    }))
    bridge = SiftBridge()
    bridge.cwd = tmp_path
    summary = bridge._policy_summary()
    entry = next(d for d in summary["datasets"] if d["name"] == "d.csv")
    assert entry["ceiling"] == "names_only"
    assert entry["max_depth"] == "names_types_labels_summary"
    assert entry["privacy_profile"] == "regulated"
