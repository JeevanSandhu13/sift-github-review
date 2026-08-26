"""Opt-in compatibility certification against disposable vendor instances.

The suite contains one independently reported scenario for every Stage 4
remote-database requirement. It is not a Sift product-release prerequisite.
In ordinary test runs absent services skip. With
``SIFT_REQUIRE_LIVE_DATABASES=1``, any absent input fails and every scenario
must execute. Synthetic fixtures and disposable identities are mandatory.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import hashlib
import re
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import sift.connectors as connectors

from sift.connectors import (
    ConnectionInput,
    ConnectorError,
    _create_bounded_engine,
    _configure_connection_read_only,
    _prepare_connection,
    check_connection,
    inspect_database,
    preview_query,
    normalize_sql,
    run_extract,
    validate_connection_security,
    databricks_oauth_connection,
    snowflake_key_pair_connection,
)
from sift.database_certification import (
    AUTHENTICATION_PROOF_CONTRACTS,
    DATABASE_CERTIFICATION_SCENARIOS,
    DatabaseCertificationScenario,
    authentication_proof_contract,
    validate_authentication_context,
)
from sift.integration_core import CancellationToken


def _strict() -> bool:
    return os.environ.get("SIFT_REQUIRE_LIVE_DATABASES") == "1"


def _require_disposable() -> None:
    if os.environ.get("SIFT_LIVE_DATABASES_DISPOSABLE") != "1":
        pytest.fail(
            "SIFT_LIVE_DATABASES_DISPOSABLE=1 is required; live certification "
            "must use synthetic fixtures and disposable identities"
        )


def _scenario_values(scenario: DatabaseCertificationScenario) -> dict[str, str]:
    missing = [name for name in scenario.required_environment() if not os.environ.get(name)]
    if missing:
        message = f"{scenario.step_id} missing live inputs: {', '.join(missing)}"
        if _strict():
            pytest.fail(message)
        pytest.skip(message)
    _require_disposable()
    return {name: os.environ[name] for name in scenario.required_environment()}


AUTHENTICATION_VARIANTS = tuple(
    (scenario, variant)
    for scenario in DATABASE_CERTIFICATION_SCENARIOS
    if scenario.mode == "auth"
    for variant in scenario.variants
)

AUTHENTICATION_PROOF_REQUIRED_FIELDS = (
    "authenticated_identity",
    "authentication_context",
)

_NATIVE_TYPE_TOKENS: dict[str, dict[str, tuple[str, ...]]] = {
    "postgresql": {
        "integer_value": ("INT",), "bigint_value": ("BIGINT",),
        "numeric_value": ("NUMERIC", "DECIMAL"), "uuid_value": ("UUID",),
        "jsonb_value": ("JSONB",), "array_value": ("ARRAY",),
        "timestamptz_value": ("TIMESTAMP",), "interval_value": ("INTERVAL",),
        "bytea_value": ("BYTEA", "BINARY"),
    },
    "mysql": {
        "unsigned_integer": ("UNSIGNED",), "enum_value": ("ENUM",),
        "set_value": ("SET",), "json_value": ("JSON",), "zero_date": ("DATE",),
    },
    "mssql": {
        "datetimeoffset_value": ("DATETIMEOFFSET",), "money_value": ("MONEY",),
        "guid_value": ("UNIQUEIDENTIFIER",), "xml_value": ("XML",),
        "spatial_value": ("GEOGRAPHY", "GEOMETRY"),
    },
    "oracle": {
        "decimal_value": ("NUMBER", "NUMERIC", "DECIMAL"),
        "timestamp_value": ("TIMESTAMP",), "interval_value": ("INTERVAL",),
        "clob_value": ("CLOB",), "blob_value": ("BLOB",),
    },
    "snowflake": {
        "variant_value": ("VARIANT",), "array_value": ("ARRAY",),
        "object_value": ("OBJECT",), "geography_value": ("GEOGRAPHY",),
        "timestamp_value": ("TIMESTAMP",),
    },
    "bigquery": {
        "record_value": ("RECORD", "STRUCT"),
        "repeated_value": ("REPEATED", "ARRAY"),
        "geography_value": ("GEOGRAPHY",), "numeric_value": ("NUMERIC",),
        "bignumeric_value": ("BIGNUMERIC",),
    },
    "redshift": {
        "decimal_value": ("DECIMAL", "NUMERIC"), "super_value": ("SUPER",),
    },
    "databricks": {
        "array_value": ("ARRAY",), "map_value": ("MAP",),
        "struct_value": ("STRUCT",), "decimal_value": ("DECIMAL",),
        "timestamp_value": ("TIMESTAMP",),
    },
}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _read_first_row(path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path).slice(0, 1).to_pylist()
    assert rows, "certification query returned no rows"
    return _jsonable(rows[0])


def _parse_expected_row(
    value: str,
    *,
    required_fields: tuple[str, ...],
    label: str,
) -> dict[str, Any]:
    expected = json.loads(value)
    assert isinstance(expected, dict) and expected, f"{label} must be a non-empty object"
    missing = set(required_fields) - set(expected)
    assert not missing, f"{label} omits required fields: {missing}"
    return expected


def _assert_exact_single_row(
    path: Path,
    expected: dict[str, Any],
    *,
    label: str,
) -> None:
    import pyarrow.parquet as pq

    rows = pq.read_table(path).slice(0, 2).to_pylist()
    assert len(rows) == 1, f"{label} must return exactly one row"
    assert _jsonable(rows[0]) == expected


def _parse_authentication_expected_row(value: str) -> dict[str, Any]:
    expected = _parse_expected_row(
        value,
        required_fields=AUTHENTICATION_PROOF_REQUIRED_FIELDS,
        label="authentication proof",
    )
    assert all(
        isinstance(expected[field], str) and expected[field].strip()
        for field in AUTHENTICATION_PROOF_REQUIRED_FIELDS
    ), "authentication identity and context must be non-empty strings"
    return expected


def _quoted_fixture_identifier(backend: str, value: str) -> str:
    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]{0,127}", value), (
        "fixture identifiers must use a portable reviewed spelling", value,
    )
    if backend == "mssql":
        return f"[{value}]"
    if backend in {"mysql", "bigquery"}:
        return f"`{value}`"
    return f'"{value}"'


def _native_fixture_query_and_manifest(
    scenario: DatabaseCertificationScenario,
    manifest_json: str,
    uri: ConnectionInput,
    tmp_path: Path,
) -> str:
    """Bind a versioned manifest to provider-reflected native columns.

    The executable query is generated here from the reviewed field list. An
    operator cannot certify casts or constants by supplying arbitrary SQL.
    """
    manifest = json.loads(manifest_json)
    assert isinstance(manifest, dict)
    assert manifest.get("schema_version") == 1
    assert manifest.get("synthetic") is True
    source_schema = manifest.get("source_schema")
    source_object = manifest.get("source_object")
    assert source_schema is None or isinstance(source_schema, str)
    assert isinstance(source_object, str) and source_object
    expected_definition = {
        "schema_version": 1,
        "backend": scenario.backend,
        "synthetic": True,
        "source_schema": source_schema,
        "source_object": source_object,
        "required_native_types": {
            field: list(_NATIVE_TYPE_TOKENS[scenario.backend][field])
            for field in scenario.required_fields
        },
    }
    reviewed_hash = hashlib.sha256(json.dumps(
        expected_definition, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    assert manifest.get("reviewed_manifest_sha256") == reviewed_hash, (
        "fixture manifest is not bound to Sift's reviewed native-type contract",
        reviewed_hash,
    )
    assert set(manifest) == {
        "schema_version", "synthetic", "source_schema", "source_object",
        "reviewed_manifest_sha256",
    }
    catalog = inspect_database(
        tmp_path, connection=uri, schema=source_schema,
        object_name=source_object,
    )
    assert len(catalog.objects) == 1
    reflected = {
        str(column["name"]): str(column["type"]).upper()
        for column in catalog.objects[0]["columns"]
    }
    assert set(scenario.required_fields) <= set(reflected)
    for field, accepted in _NATIVE_TYPE_TOKENS[scenario.backend].items():
        assert any(token in reflected[field] for token in accepted), (
            field, reflected[field], accepted,
        )
    qualified = [_quoted_fixture_identifier(scenario.backend, source_object)]
    if source_schema:
        qualified.insert(0, _quoted_fixture_identifier(
            scenario.backend, source_schema,
        ))
    columns = ", ".join(
        _quoted_fixture_identifier(scenario.backend, field)
        for field in scenario.required_fields
    )
    return f"SELECT {columns} FROM {'.'.join(qualified)}"


def _cancellation_proof_query(backend: str, tag: str, uri: str) -> str:
    """Return a code-owned terminal-state query for one unguessable operation."""
    assert re.fullmatch(r"sift_cancel_[0-9a-f]{32}", tag)
    if backend == "mssql":
        return f"""WITH sift_events AS (
SELECT CAST(target.target_data AS xml) AS target_data
FROM sys.dm_xe_session_targets AS target
JOIN sys.dm_xe_sessions AS session
  ON session.address = target.event_session_address
WHERE session.name = 'sift_cancellation_certification'
  AND target.target_name = 'ring_buffer')
SELECT CASE WHEN EXISTS (
  SELECT 1 FROM sift_events
  CROSS APPLY target_data.nodes('//RingBufferTarget/event[@name="attention"]') AS item(event)
  WHERE item.event.value('(action[@name="sql_text"]/value)[1]', 'nvarchar(max)')
    LIKE '%{tag}%'
) THEN 'cancelled' ELSE 'not_cancelled' END AS cancellation_state"""
    if backend == "snowflake":
        from sqlalchemy.engine import make_url

        warehouse = str(make_url(uri).query.get("warehouse", ""))
        assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]{0,127}", warehouse)
        return f"""SELECT IFF(COUNT_IF(
  (execution_status = 'FAILED_WITH_ERROR' AND error_code = '000604')
  OR execution_status = 'CANCELED') = 1, 'cancelled', 'not_cancelled')
  AS "cancellation_state"
FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY(
  END_TIME_RANGE_START => DATEADD('minute', -10, CURRENT_TIMESTAMP()),
  RESULT_LIMIT => 1000))
WHERE QUERY_TEXT ILIKE '%{tag}%'
  AND WAREHOUSE_NAME = '{warehouse}'"""
    if backend == "bigquery":
        from sqlalchemy.engine import make_url

        location = str(make_url(uri).query.get("location", "us")).casefold()
        assert re.fullmatch(r"[a-z0-9_-]{1,32}", location)
        return f"""SELECT IF(COUNTIF(
  state = 'DONE' AND error_result.reason = 'stopped') = 1,
  'cancelled', 'not_cancelled') AS cancellation_state
FROM `region-{location}`.INFORMATION_SCHEMA.JOBS_BY_USER
WHERE CONTAINS_SUBSTR(query, '{tag}')
  AND creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 10 MINUTE)"""
    if backend == "redshift":
        return f"""SELECT CASE WHEN SUM(
  CASE WHEN status = 'canceled' THEN 1 ELSE 0 END) = 1
  THEN 'cancelled' ELSE 'not_cancelled' END
  AS cancellation_state
FROM sys_query_history
WHERE query_text LIKE '%{tag}%'
  AND start_time >= DATEADD(minute, -10, GETDATE())"""
    if backend == "databricks":
        return f"""SELECT IF(COUNT_IF(execution_status = 'CANCELED') = 1,
  'cancelled', 'not_cancelled') AS cancellation_state
FROM system.query.history
WHERE statement_text LIKE '%{tag}%'
  AND start_time >= CURRENT_TIMESTAMP() - INTERVAL 10 MINUTES"""
    raise AssertionError(f"no cancellation proof contract for {backend}")


def _assert_backend(uri: ConnectionInput, backend: str, tmp_path: Path) -> None:
    health = check_connection(tmp_path, connection=uri)
    assert health.backend == backend


def _assert_authentication_configuration(
    connection: ConnectionInput,
    backend: str,
    variant: str,
) -> None:
    """Prove Sift selected the intended driver credential path."""
    from sqlalchemy.engine import make_url
    from sift.connectors import ConnectionSpec

    if backend == "snowflake":
        if variant == "key_pair":
            assert isinstance(connection, ConnectionSpec)
            assert connection.authentication == "key_pair"
            return
        url = make_url(connection if isinstance(connection, str) else connection.uri)
        authenticator = str(url.query.get("authenticator", "")).casefold()
        if variant == "oauth":
            assert authenticator == "oauth" and bool(url.query.get("token"))
        elif variant == "sso":
            assert authenticator and authenticator not in {"oauth", "snowflake"}
        elif variant == "password":
            assert bool(url.password) and authenticator in {"", "snowflake"}
    elif backend == "databricks":
        if variant in {"oauth_u2m", "oauth_m2m"}:
            assert isinstance(connection, ConnectionSpec)
            assert connection.authentication == variant
        elif variant == "token":
            url = make_url(connection if isinstance(connection, str) else connection.uri)
            assert bool(url.password)


def _assert_authentication_variant(
    uri: ConnectionInput,
    expected_row_json: str,
    backend: str,
    variant: str,
    tmp_path: Path,
    *,
    dataset_name: str,
) -> dict[str, Any]:
    """Prove the expected identity/context, not merely connectivity."""
    _assert_authentication_configuration(uri, backend, variant)
    _assert_backend(uri, backend, tmp_path)
    expected = _parse_authentication_expected_row(expected_row_json)
    contract = authentication_proof_contract(backend)
    result = run_extract(
        tmp_path,
        connection=uri,
        sql=contract.query,
        dataset_name=dataset_name,
        row_limit=2,
    )
    assert result.rows == 1, "authentication proof query must return exactly one row"
    _assert_exact_single_row(
        result.dataset_path, expected, label="authentication proof query",
    )
    validate_authentication_context(
        backend, variant, str(expected["authentication_context"]),
    )
    return expected


def _authentication_connection(
    scenario: DatabaseCertificationScenario,
    variant: str,
    values: dict[str, str],
) -> ConnectionInput:
    prefix = f"{scenario.env_prefix}_{variant.upper()}"
    uri = values[f"{prefix}_URI"]
    if (scenario.step_id, variant) == ("S04-082", "key_pair"):
        return snowflake_key_pair_connection(
            uri,
            private_key_pem=values[f"{prefix}_PRIVATE_KEY_PEM"],
            passphrase=os.environ.get(f"{prefix}_PRIVATE_KEY_PASSPHRASE"),
        )
    if (scenario.step_id, variant) == ("S04-099", "oauth_u2m"):
        return databricks_oauth_connection(uri, mode="oauth_u2m")
    if (scenario.step_id, variant) == ("S04-100", "oauth_m2m"):
        return databricks_oauth_connection(
            uri,
            mode="oauth_m2m",
            client_id=values[f"{prefix}_CLIENT_ID"],
            client_secret=values[f"{prefix}_CLIENT_SECRET"],
        )
    return uri


def _assert_negative_hostname_is_same_connection(uri: str, negative: str) -> None:
    """Prevent an auth failure on an unrelated endpoint from proving TLS rejection."""
    from sqlalchemy.engine import make_url

    good = make_url(uri)
    bad = make_url(negative)
    assert good.host and bad.host and good.host != bad.host
    assert bad.set(host=good.host) == good, (
        "negative TLS URI must differ from the working URI only by hostname"
    )


def _assert_cancellation(
    uri: str,
    query: str,
    backend: str,
    tmp_path: Path,
    *,
    suffix: str,
    required_transport: str | None = None,
) -> None:
    tag = f"sift_cancel_{uuid.uuid4().hex}"
    normalized = normalize_sql(query, backend=backend)
    assert normalized is not None, "cancellation fixture must be read-only SQL"
    tagged_query = f"{normalized} /* {tag} */"
    proof_query = _cancellation_proof_query(backend, tag, uri)
    token = CancellationToken()
    outcome: dict[str, Any] = {}
    transport_evidence: list[Any] = []

    def execute() -> None:
        try:
            run_extract(
                tmp_path,
                connection=uri,
                sql=tagged_query,
                dataset_name=f"{backend}_{suffix}",
                cancellation=token,
                transport_probe=(
                    transport_evidence.append if required_transport else None
                ),
            )
        except BaseException as exc:  # recorded and asserted in the parent thread
            outcome["exception"] = exc

    worker = threading.Thread(target=execute, daemon=True)
    worker.start()
    time.sleep(0.5)
    token.cancel()
    worker.join(timeout=20)
    assert not worker.is_alive(), "driver did not terminate within 20 seconds of cancellation"
    error = outcome.get("exception")
    assert isinstance(error, ConnectorError), outcome
    assert error.code == "cancelled"
    if required_transport is not None:
        assert transport_evidence, (
            "the cancelled operation never reached the required transport"
        )
        assert transport_evidence[-1].transport == required_transport
    proof_deadline = time.monotonic() + 60
    last_row: dict[str, Any] | None = None
    while time.monotonic() < proof_deadline:
        proof = run_extract(
            tmp_path,
            connection=uri,
            sql=proof_query,
            dataset_name=f"{backend}_{suffix}_cancel_proof",
            row_limit=2,
        )
        last_row = _read_first_row(proof.dataset_path)
        if last_row == {"cancellation_state": "cancelled"}:
            break
        time.sleep(1)
    assert last_row == {"cancellation_state": "cancelled"}, last_row


@pytest.fixture()
def live_sqlite(tmp_path: Path) -> str:
    path = tmp_path / "certification.sqlite"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE certification ("
        "id INTEGER, exact_value NUMERIC, observed_at TEXT, active BOOLEAN, payload BLOB)"
    )
    connection.execute(
        "INSERT INTO certification VALUES (1, 12.25, '2026-08-21T12:00:00Z', 1, X'00FF')"
    )
    connection.commit()
    connection.close()
    return str(path)


def test_sqlite_live_disposable_certification(
    tmp_path: Path, live_sqlite: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Resource-guard behavior has dedicated tests. This local functional proof
    # must remain runnable on a nearly full qualification host without changing
    # the production 512 MiB reserve.
    monkeypatch.setattr(connectors, "_MIN_FREE_DISK_RESERVE", 0)
    health = check_connection(tmp_path, connection=live_sqlite)
    catalog = inspect_database(tmp_path, connection=live_sqlite, object_name="certification")
    extract = run_extract(
        tmp_path, connection=live_sqlite, sql="SELECT * FROM certification",
        dataset_name="sqlite_certification",
    )
    assert health.backend == "sqlite"
    assert catalog.objects[0]["columns"]
    assert extract.rows == 1 and extract.dataset_path.is_file()


def test_duckdb_live_disposable_certification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(connectors, "_MIN_FREE_DISK_RESERVE", 0)
    duckdb = pytest.importorskip("duckdb")
    path = tmp_path / "certification.duckdb"
    connection = duckdb.connect(str(path))
    connection.execute(
        "CREATE TABLE certification AS SELECT 1 AS id, "
        "DECIMAL '1234567890.123456789' AS exact_value, "
        "TIMESTAMPTZ '2026-08-21 12:00:00-07:00' AS observed_at, "
        "TRUE AS active, [1, 2, 3] AS nested"
    )
    connection.close()
    health = check_connection(tmp_path, connection=str(path))
    extract = run_extract(
        tmp_path, connection=str(path), sql="SELECT * FROM certification",
        dataset_name="duckdb_certification",
    )
    assert health.backend == "duckdb"
    assert extract.rows == 1 and extract.columns == 5


def test_exact_type_value_proof_checks_values_and_cardinality(
    tmp_path: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    one_row = tmp_path / "one-row.parquet"
    pq.write_table(pa.table({
        "integer_value": [7],
        "numeric_value": [Decimal("12.340")],
        "bytea_value": [b"\x00\xff"],
        "array_value": [[1, 2]],
    }), one_row)
    expected = _parse_expected_row(
        json.dumps({
            "integer_value": 7,
            "numeric_value": "12.340",
            "bytea_value": "00ff",
            "array_value": [1, 2],
        }),
        required_fields=(
            "integer_value", "numeric_value", "bytea_value", "array_value",
        ),
        label="type value fixture",
    )
    _assert_exact_single_row(one_row, expected, label="type fidelity query")

    with pytest.raises(AssertionError):
        _assert_exact_single_row(
            one_row,
            {**expected, "integer_value": 8},
            label="type fidelity query",
        )

    two_rows = tmp_path / "two-rows.parquet"
    pq.write_table(pa.table({"integer_value": [7, 7]}), two_rows)
    with pytest.raises(AssertionError, match="exactly one row"):
        _assert_exact_single_row(
            two_rows, {"integer_value": 7}, label="type fidelity query",
        )


def test_authentication_proof_requires_exact_identity_context_shape() -> None:
    expected = _parse_authentication_expected_row(json.dumps({
        "authenticated_identity": "researcher@example.test",
        "authentication_context": "certificate:subject=researcher",
    }))
    assert expected == {
        "authenticated_identity": "researcher@example.test",
        "authentication_context": "certificate:subject=researcher",
    }

    with pytest.raises(AssertionError, match="omits required fields"):
        _parse_authentication_expected_row(json.dumps({
            "authenticated_identity": "researcher@example.test",
        }))
    with pytest.raises(AssertionError, match="non-empty strings"):
        _parse_authentication_expected_row(json.dumps({
            "authenticated_identity": "researcher@example.test",
            "authentication_context": " ",
        }))


@pytest.mark.parametrize("scenario", DATABASE_CERTIFICATION_SCENARIOS, ids=lambda row: row.step_id)
def test_remote_database_requirement(scenario: DatabaseCertificationScenario, tmp_path: Path) -> None:
    values = _scenario_values(scenario)
    prefix = scenario.env_prefix
    backend = scenario.backend
    mode = scenario.mode

    if mode == "auth":
        expected_proofs: list[dict[str, Any]] = []
        for variant in scenario.variants:
            variant_prefix = f"{prefix}_{variant.upper()}"
            expected_proofs.append(_assert_authentication_variant(
                _authentication_connection(scenario, variant, values),
                values[f"{variant_prefix}_EXPECTED_ROW_JSON"],
                backend,
                variant,
                tmp_path,
                dataset_name=f"{backend}_{scenario.step_id}_{variant}_auth_proof",
            ))
        assert len({row["authenticated_identity"].strip() for row in expected_proofs}) == len(
            expected_proofs
        ), "each authentication variant must use a distinct disposable identity"
        return

    uri = values[f"{prefix}_URI"]
    if mode == "core":
        _assert_backend(uri, backend, tmp_path)
        catalog = inspect_database(tmp_path, connection=uri)
        result = run_extract(
            tmp_path, connection=uri,
            sql="SELECT 1 AS sift_live_probe" if backend != "oracle" else "SELECT 1 AS sift_live_probe FROM DUAL",
            dataset_name=f"{backend}_{scenario.step_id.lower().replace('-', '_')}", row_limit=10,
        )
        assert catalog.backend == backend and result.rows == 1
    elif mode == "tls_hostname":
        validate_connection_security(uri, backend)
        _assert_backend(uri, backend, tmp_path)
        negative = values[f"{prefix}_NEGATIVE_HOSTNAME_URI"]
        validate_connection_security(negative, backend)
        _assert_negative_hostname_is_same_connection(uri, negative)
        with pytest.raises(ConnectorError) as caught:
            check_connection(tmp_path, connection=negative)
        diagnostic = str(caught.value).casefold()
        assert any(
            marker in diagnostic
            for marker in ("certificate", "hostname", "ssl", "tls", "server identity")
        ), "negative endpoint failed for a reason other than TLS hostname verification"
    elif mode == "secure_policy":
        validate_connection_security(uri, backend)
        _assert_backend(uri, backend, tmp_path)
        with pytest.raises(ConnectorError):
            validate_connection_security(values[f"{prefix}_INSECURE_URI"], backend)
        if scenario.step_id == "S04-081":
            # A query parameter proves only local policy. The second endpoint
            # must exercise the connector's actual fail-closed OCSP path (for
            # example through a disposable revocation-responder fault).
            failure_uri = values[f"{prefix}_OCSP_FAILURE_URI"]
            validate_connection_security(failure_uri, backend)
            with pytest.raises(ConnectorError) as caught:
                check_connection(tmp_path, connection=failure_uri)
            diagnostic = str(caught.value).casefold()
            assert any(marker in diagnostic for marker in (
                "ocsp", "revocation", "certificate",
            )), "OCSP fault failed for a reason unrelated to certificate revocation"
    elif mode == "types":
        assert set(scenario.required_fields) == set(_NATIVE_TYPE_TOKENS[backend])
        fixture_query = _native_fixture_query_and_manifest(
            scenario, values[f"{prefix}_FIXTURE_MANIFEST_JSON"],
            uri, tmp_path,
        )
        result = run_extract(
            tmp_path, connection=uri, sql=fixture_query,
            dataset_name=f"{backend}_{scenario.step_id.lower().replace('-', '_')}", row_limit=100,
        )
        import pyarrow.parquet as pq

        actual = {field.name: str(field.type) for field in pq.read_schema(result.dataset_path)}
        expected = json.loads(values[f"{prefix}_EXPECTED_SCHEMA_JSON"])
        assert isinstance(expected, dict) and expected
        assert set(scenario.required_fields) <= set(expected), (
            f"type fixture omits required fields: {set(scenario.required_fields) - set(expected)}"
        )
        assert actual == expected
        expected_row = _parse_expected_row(
            values[f"{prefix}_EXPECTED_ROW_JSON"],
            required_fields=scenario.required_fields,
            label="type value fixture",
        )
        assert result.rows == 1, "type fidelity query must return exactly one row"
        _assert_exact_single_row(
            result.dataset_path, expected_row, label="type fidelity query",
        )
    elif mode == "read_only":
        if os.environ.get("SIFT_LIVE_DATABASE_WRITE_PROBE_ACK") != "1":
            pytest.fail("read-only probes require SIFT_LIVE_DATABASE_WRITE_PROBE_ACK=1")
        from sqlalchemy import create_engine, text

        prepared_backend, effective = _prepare_connection(tmp_path, uri)
        assert prepared_backend == backend
        engine = _create_bounded_engine(effective, backend)
        read_probe = "SELECT probe_value FROM sift_certification_readonly_probe"
        write_probe = (
            "UPDATE sift_certification_readonly_probe "
            "SET probe_value = probe_value"
        )
        try:
            with engine.connect() as connection:
                assert connection.execute(text(read_probe)).fetchone() is not None
                connection.rollback()
                # Prove the write probe itself is valid against this disposable
                # fixture. A missing table or syntax error must not masquerade as
                # successful read-only enforcement.
                connection.execute(text(write_probe))
                connection.rollback()
                _configure_connection_read_only(connection, engine.dialect.name)
                with pytest.raises(Exception):
                    connection.execute(text(write_probe))
                connection.rollback()
        finally:
            engine.dispose()
    elif mode == "cancellation":
        _assert_cancellation(
            uri, values[f"{prefix}_CANCEL_QUERY"], backend, tmp_path,
            suffix=scenario.step_id,
        )
    elif mode == "driver_18":
        _assert_backend(uri, backend, tmp_path)
        import pyodbc

        prepared_backend, effective = _prepare_connection(tmp_path, uri)
        assert prepared_backend == backend
        engine = _create_bounded_engine(effective, backend)
        try:
            raw = engine.raw_connection()
            try:
                version = str(raw.driver_connection.getinfo(pyodbc.SQL_DRIVER_VER))
                name = str(raw.driver_connection.getinfo(pyodbc.SQL_DRIVER_NAME))
                assert version.split(".", 1)[0] == "18", (name, version)
                assert "msodbcsql18" in name.casefold(), (name, version)
            finally:
                raw.close()
        finally:
            engine.dispose()
    elif mode == "result":
        result = run_extract(
            tmp_path, connection=uri,
            sql=scenario.proof_query or values[f"{prefix}_QUERY"],
            dataset_name=f"{backend}_{scenario.step_id}", row_limit=10,
        )
        expected = json.loads(values[f"{prefix}_EXPECTED_ROW_JSON"])
        assert set(scenario.required_fields) <= set(expected)
        if scenario.step_id == "S04-089":
            from sqlalchemy.engine import make_url

            url = make_url(uri)
            assert expected["project"] == (url.host or url.database)
            assert expected["billing_project"] == url.query.get("billing_project_id")
            assert _read_first_row(result.dataset_path) == {
                "billing_project": expected["billing_project"],
            }
        else:
            assert _read_first_row(result.dataset_path) == expected
    elif mode == "preview":
        preview = preview_query(tmp_path, connection=uri, sql=values[f"{prefix}_QUERY"])
        assert preview.executes_query is False
        assert preview.dry_run_supported is True
        assert preview.estimated_bytes is not None and preview.estimated_bytes >= 0
        assert preview.estimate_source == "bigquery_dry_run"
    elif mode == "metadata":
        expected = json.loads(values[f"{prefix}_EXPECTED_METADATA_JSON"])
        catalog = inspect_database(
            tmp_path, connection=uri, schema=expected.get("schema"),
            object_name=expected.get("object"),
        )
        assert expected["schema"] in catalog.schemas
        assert any(row["name"] == expected["object"] for row in catalog.objects)
        denied = json.loads(values[f"{prefix}_DENIED_METADATA_JSON"])
        assert isinstance(denied, dict) and denied.get("schema") and denied.get("object")
        assert set(denied) == {"schema", "object"}
        denied_source = ".".join(
            _quoted_fixture_identifier(backend, value)
            for value in (denied["schema"], denied["object"])
        )
        with pytest.raises(ConnectorError) as caught:
            run_extract(
                tmp_path, connection=uri,
                sql=f"SELECT 1 AS permission_probe FROM {denied_source}",
                dataset_name="databricks_denied_permission_probe", row_limit=1,
            )
        diagnostic = str(caught.value).casefold()
        assert any(marker in diagnostic for marker in (
            "permission", "not authorized", "access denied", "unauthorized",
            "insufficient privileges", "insufficient_permissions",
            "does not have",
        )), (
            "a nonexistent object or unrelated failure cannot prove a denied "
            "Unity Catalog permission",
            diagnostic,
        )
    elif mode == "warehouse_cancellation":
        _assert_cancellation(
            uri, values[f"{prefix}_CANCEL_QUERY"], backend, tmp_path,
            suffix="warehouse",
        )
        from sqlalchemy.engine import make_url

        warehouse = str(make_url(uri).query.get("warehouse", ""))
        assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]{0,127}", warehouse)
        prepared_backend, effective = _prepare_connection(tmp_path, uri)
        assert prepared_backend == "snowflake"
        engine = _create_bounded_engine(effective, backend)
        try:
            deadline = time.monotonic() + 30
            state = ""
            while time.monotonic() < deadline:
                with engine.connect() as connection:
                    rows = connection.exec_driver_sql(
                        f"SHOW WAREHOUSES LIKE '{warehouse}'"
                    ).mappings().all()
                matching = [
                    row for row in rows
                    if str(row.get("name", "")).casefold() == warehouse.casefold()
                ]
                assert len(matching) == 1
                state = str(matching[0].get("state", "")).upper()
                if state == "SUSPENDED":
                    break
                time.sleep(1)
            assert state == "SUSPENDED", (warehouse, state)
        finally:
            engine.dispose()
    elif mode == "cloudfetch":
        minimum = int(values[f"{prefix}_EXPECTED_MIN_ROWS"])
        assert minimum >= 10_000, "CloudFetch certification needs a meaningfully large fixture"
        transport_evidence: list[Any] = []
        result = run_extract(
            tmp_path, connection=uri, sql=values[f"{prefix}_QUERY"],
            dataset_name="databricks_cloudfetch", row_limit=minimum + 1,
            transport_probe=transport_evidence.append,
        )
        assert result.rows >= minimum
        assert len(transport_evidence) == 1
        assert transport_evidence[0].backend == "databricks"
        assert transport_evidence[0].transport == "cloudfetch", (
            "the actual Sift extraction did not use CloudFetch"
        )
        _assert_cancellation(
            uri, values[f"{prefix}_CANCEL_QUERY"], backend, tmp_path,
            suffix="cloudfetch",
            required_transport="cloudfetch",
        )
    else:  # pragma: no cover - registry mode exhaustiveness guard
        raise AssertionError(f"unhandled certification mode: {mode}")


@pytest.mark.parametrize(
    ("scenario", "variant"),
    AUTHENTICATION_VARIANTS,
    ids=lambda value: value.step_id if isinstance(value, DatabaseCertificationScenario) else value,
)
def test_remote_database_auth_variant(
    scenario: DatabaseCertificationScenario,
    variant: str,
    tmp_path: Path,
) -> None:
    """Certify one authentication path without pretending its peers ran."""
    variant_prefix = f"{scenario.env_prefix}_{variant.upper()}"
    required = scenario.required_authentication_environment(variant)
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        message = (
            f"{scenario.step_id}:{variant} missing live inputs: "
            f"{', '.join(missing)}"
        )
        if _strict():
            pytest.fail(message)
        pytest.skip(message)
    _require_disposable()
    _assert_authentication_variant(
        _authentication_connection(
            scenario,
            variant,
            {name: os.environ[name] for name in required},
        ),
        os.environ[required[1]],
        scenario.backend,
        variant,
        tmp_path,
        dataset_name=f"{scenario.backend}_{scenario.step_id}_{variant}_auth_proof",
    )


def test_every_auth_backend_has_one_code_owned_proof_contract() -> None:
    auth_backends = {
        scenario.backend for scenario in DATABASE_CERTIFICATION_SCENARIOS
        if scenario.mode == "auth"
    }
    assert set(AUTHENTICATION_PROOF_CONTRACTS) == auth_backends
    for backend, contract in AUTHENTICATION_PROOF_CONTRACTS.items():
        assert contract.backend == backend
        assert set(contract.mechanism_variants).issubset({
            variant
            for scenario in DATABASE_CERTIFICATION_SCENARIOS
            if scenario.backend == backend and scenario.mode == "auth"
            for variant in scenario.variants
        })
        assert "authenticated_identity" in contract.query
        assert "authentication_context" in contract.query
        assert "expected" not in contract.query.lower()
    assert {
        backend: contract.mechanism_variants
        for backend, contract in AUTHENTICATION_PROOF_CONTRACTS.items()
    } == {
        "postgresql": ("certificate",),
        "mssql": ("entra", "windows"),
        "oracle": ("wallet_mtls",),
        "snowflake": (),
        "bigquery": (),
        "redshift": ("iam",),
        "databricks": (),
    }


@pytest.mark.parametrize(
    ("backend", "uri", "provider_surface"),
    (
        ("mssql", "mssql+pyodbc://user@host/db", "dm_xe_session_targets"),
        (
            "snowflake",
            "snowflake://user@account/db/schema?warehouse=SIFT_CERT",
            "QUERY_HISTORY",
        ),
        ("bigquery", "bigquery://project/dataset?location=us", "JOBS_BY_USER"),
        ("redshift", "redshift+redshift_connector://user@host/db", "sys_query_history"),
        ("databricks", "databricks://user@host?http_path=/sql/x", "system.query.history"),
    ),
)
def test_cancellation_proof_is_code_owned_and_bound_to_one_operation(
    backend: str, uri: str, provider_surface: str,
) -> None:
    tag = "sift_cancel_0123456789abcdef0123456789abcdef"
    query = _cancellation_proof_query(backend, tag, uri)
    assert tag in query
    assert provider_surface.casefold() in query.casefold()
    assert "cancellation_state" in query
    assert "not_cancelled" in query


def test_native_fixture_manifest_generates_direct_provider_bound_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = next(
        value for value in DATABASE_CERTIFICATION_SCENARIOS
        if value.step_id == "S04-096"
    )
    definition = {
        "schema_version": 1,
        "backend": "redshift",
        "synthetic": True,
        "source_schema": "sift_cert",
        "source_object": "native_types",
        "required_native_types": {
            field: list(_NATIVE_TYPE_TOKENS["redshift"][field])
            for field in scenario.required_fields
        },
    }
    manifest = {
        key: definition[key] for key in (
            "schema_version", "synthetic", "source_schema", "source_object",
        )
    }
    manifest["reviewed_manifest_sha256"] = hashlib.sha256(json.dumps(
        definition, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    catalog = type("Catalog", (), {"objects": ({
        "columns": [
            {"name": "decimal_value", "type": "DECIMAL(38, 9)"},
            {"name": "super_value", "type": "SUPER"},
        ],
    },)})()
    monkeypatch.setitem(
        globals(), "inspect_database", lambda *_args, **_kwargs: catalog,
    )
    query = _native_fixture_query_and_manifest(
        scenario, json.dumps(manifest),
        "redshift+redshift_connector://user@host/db", Path("."),
    )
    assert query == (
        'SELECT "decimal_value", "super_value" '
        'FROM "sift_cert"."native_types"'
    )
    poisoned = dict(manifest)
    poisoned["native_types"] = {"super_value": "SUPER"}
    with pytest.raises(AssertionError):
        _native_fixture_query_and_manifest(
            scenario, json.dumps(poisoned),
            "redshift+redshift_connector://user@host/db", Path("."),
        )


@pytest.mark.parametrize(
    ("backend", "variant", "context"),
    (
        ("postgresql", "certificate", "tls-client-certificate:CN=fixture"),
        ("mssql", "entra", "FEDAUTH"),
        ("mssql", "windows", "KERBEROS"),
        ("mssql", "windows", "NTLM"),
        ("oracle", "wallet_mtls", "SSL"),
        ("redshift", "iam", "arn:aws:iam::123456789012:role/fixture"),
        ("redshift", "iam", "temporary-iam-user:IAMA:fixture"),
        ("snowflake", "oauth", "principal_type:USER_PERSON"),
        ("bigquery", "adc", "query_project:fixture"),
        ("databricks", "token", "session_principal:fixture"),
    ),
)
def test_authentication_context_fixture_contract_accepts_provider_evidence(
    backend: str, variant: str, context: str,
) -> None:
    validate_authentication_context(backend, variant, context)


@pytest.mark.parametrize(
    ("backend", "variant", "context"),
    (
        ("postgresql", "certificate", "tls:no-client-certificate"),
        ("mssql", "entra", "SQL"),
        ("mssql", "windows", "SQL"),
        ("oracle", "wallet_mtls", "PASSWORD"),
        ("redshift", "iam", "local-user"),
    ),
)
def test_authentication_context_fixture_contract_rejects_wrong_mechanism(
    backend: str, variant: str, context: str,
) -> None:
    with pytest.raises(ValueError):
        validate_authentication_context(backend, variant, context)


def test_every_remote_requirement_has_complete_live_configuration() -> None:
    if not _strict():
        pytest.skip("release live-database gate is opt-in")
    _require_disposable()
    missing = {
        scenario.step_id: [
            name for name in scenario.required_environment() if not os.environ.get(name)
        ]
        for scenario in DATABASE_CERTIFICATION_SCENARIOS
    }
    missing = {step: names for step, names in missing.items() if names}
    assert not missing, f"incomplete Stage 4 live certification inputs: {missing}"
