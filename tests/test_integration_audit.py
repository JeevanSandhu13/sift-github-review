from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from sift.integration_audit import (
    AUDIT_RELATIVE_PATH,
    read_and_verify,
    record_integration_event,
)


def test_audit_allowlist_drops_credentials_queries_and_remote_paths(tmp_path) -> None:
    assert record_integration_event(
        tmp_path,
        integration_id="postgresql",
        kind="database",
        action="materialize",
        outcome="success",
        metadata={
            "rows": 20,
            "password": "secret-canary",
            "connection": "postgresql://alice:secret-canary@host/db",
            "sql": "select secret_canary from people",
            "object": "private/patient.csv",
        },
    )
    text = (tmp_path / AUDIT_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "secret-canary" not in text
    assert "patient.csv" not in text
    assert "select" not in text
    ok, rows = read_and_verify(tmp_path)
    assert ok is True
    assert rows[0]["metadata"] == {"rows": 20}


def test_audit_chain_detects_mutation(tmp_path) -> None:
    for index in range(3):
        assert record_integration_event(
            tmp_path,
            integration_id="s3",
            kind="object_storage",
            action="materialize",
            outcome="success",
            metadata={"bytes": index + 1},
        )
    assert read_and_verify(tmp_path)[0] is True
    path = tmp_path / AUDIT_RELATIVE_PATH
    rows = path.read_text(encoding="utf-8").splitlines()
    event = json.loads(rows[1])
    event["metadata"]["bytes"] = 999
    rows[1] = json.dumps(event)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert read_and_verify(tmp_path)[0] is False


def test_concurrent_audit_appends_preserve_every_event(tmp_path) -> None:
    def append(index: int) -> bool:
        return record_integration_event(
            tmp_path,
            integration_id="openai",
            kind="model",
            action="conversation_turn",
            outcome="success",
            metadata={"duration_ms": index},
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(append, range(40)))
    ok, rows = read_and_verify(tmp_path)
    assert ok is True
    assert len(rows) == 40
    assert {row["metadata"]["duration_ms"] for row in rows} == set(range(40))


def test_malformed_existing_tail_fails_closed(tmp_path) -> None:
    path = tmp_path / AUDIT_RELATIVE_PATH
    path.parent.mkdir(parents=True)
    path.write_text("not json\n", encoding="utf-8")
    assert record_integration_event(
        tmp_path,
        integration_id="openai",
        kind="model",
        action="conversation_turn",
        outcome="failure",
    ) is False
    assert path.read_text(encoding="utf-8") == "not json\n"


def test_integration_audit_refuses_symlink_without_modifying_target(tmp_path) -> None:
    path = tmp_path / AUDIT_RELATIVE_PATH
    path.parent.mkdir(parents=True)
    target = tmp_path / "sensitive.txt"
    target.write_text("preserve", encoding="utf-8")
    path.symlink_to(target)
    assert record_integration_event(
        tmp_path,
        integration_id="openai",
        kind="model",
        action="conversation_turn",
        outcome="success",
    ) is False
    assert target.read_text(encoding="utf-8") == "preserve"
    assert read_and_verify(tmp_path) == (False, [])


def test_integration_audit_malformed_utf8_fails_closed(tmp_path) -> None:
    path = tmp_path / AUDIT_RELATIVE_PATH
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xff\xfe\n")
    assert read_and_verify(tmp_path) == (False, [])
