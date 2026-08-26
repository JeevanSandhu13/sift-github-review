"""Load the build-embedded desktop update policy, if one is configured."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from sift.release_manifest import CHANNELS, ReleaseManifestError, load_trusted_json
from sift.update_service import UpdateError, _origin


POLICY_FILENAME = "update-policy.json"
TRUST_FILENAME = "release-trust-store.json"
POLICY_FIELDS = frozenset({
    "format", "schema_version", "enabled", "manifest_url", "channel",
    "trust_store_filename", "trust_store_sha256",
})


def _resource_root() -> Path | None:
    override = os.environ.get("SIFT_UPDATE_POLICY_DIR")
    if override:
        return Path(override).expanduser()
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        bundled = Path(frozen_root) / "sift" / "update"
        # Development and local qualification builds intentionally omit the
        # publisher-controlled update endpoint.  An absent bundle is therefore
        # "not configured"; a present but incomplete/tampered bundle remains a
        # hard failure in load_update_policy below.
        return bundled if bundled.is_dir() else None
    generated = Path(__file__).resolve().parents[2] / "packaging" / "generated" / "update"
    return generated if generated.is_dir() else None


def load_update_policy() -> dict[str, Any]:
    root = _resource_root()
    if root is None:
        return {
            "ok": True,
            "configured": False,
            "reason": "Updates are not configured in this development build.",
        }
    policy_path = root / POLICY_FILENAME
    if policy_path.is_symlink() or not policy_path.is_file():
        raise UpdateError("embedded update policy is missing or unsafe")
    try:
        document = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UpdateError("embedded update policy is unreadable") from exc
    if not isinstance(document, dict) or set(document) != POLICY_FIELDS:
        raise UpdateError("embedded update policy fields do not match its schema")
    if document["format"] != "sift-update-policy" or document["schema_version"] != 1:
        raise UpdateError("embedded update policy format is unsupported")
    if document["enabled"] is not True:
        return {"ok": True, "configured": False, "reason": "Updates are disabled."}
    if document["channel"] not in CHANNELS:
        raise UpdateError("embedded update channel is invalid")
    _origin(document["manifest_url"])
    if document["trust_store_filename"] != TRUST_FILENAME:
        raise UpdateError("embedded update trust-store filename is invalid")
    trust_path = root / TRUST_FILENAME
    try:
        load_trusted_json(trust_path, document["trust_store_sha256"])
    except ReleaseManifestError as exc:
        raise UpdateError(str(exc)) from exc
    return {
        "ok": True,
        "configured": True,
        "manifest_url": document["manifest_url"],
        "channel": document["channel"],
        "trust_store_path": str(trust_path),
        "trust_store_sha256": document["trust_store_sha256"],
    }


__all__ = ["load_update_policy"]
