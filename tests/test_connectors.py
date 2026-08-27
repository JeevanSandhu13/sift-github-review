"""Data connectors — databases as local datasets, without a hole in the sandbox.

The architectural claim: a connector can exist without weakening the
privacy boundary, because the query runs host-side (like
``install_packages``), the result is materialized locally, and
everything downstream is an ordinary Sift dataset.

The tests that matter most are the negative ones: the model must not
be able to issue a query, credentials must never survive into any
visible surface, and no statement that can modify a database may run.
Sift users point this at production research databases.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from sift import connectors
from sift.connectors import (
    ConnectorError,
    check_connection,
    describe_backend,
    inspect_database,
    normalize_sql,
    preview_query,
    redact_connection,
    run_extract,
    validate_connection_security,
)
from sift.integrations import DATABASE_ADAPTERS


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    path = tmp_path / "hospital.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE patients (id INTEGER, age INTEGER, dx TEXT)")
    con.executemany(
        "INSERT INTO patients VALUES (?,?,?)",
        [(i, 30 + i % 50, f"dx{i % 7}") for i in range(1000)],
    )
    con.commit()
    con.close()
    return path


# --------------------------------------------------------------------
# The model must not be able to query a database
# --------------------------------------------------------------------


def test_connector_is_not_a_model_capability() -> None:
    """A connector the model could drive is a data-exfiltration
    primitive. It must be bridge-only."""
    from sift.tools import ALLOWED_TOOL_NAMES, HANDLERS

    joined = " ".join(ALLOWED_TOOL_NAMES) + " " + " ".join(HANDLERS)
    for word in ("connector", "database", "sql_query", "extract"):
        assert word not in joined.lower()
    assert "connectors" not in Path("src/sift/tools.py").read_text(encoding="utf-8")


def test_sandbox_network_denial_is_untouched() -> None:
    """The connector must not have loosened the executor profile."""
    executor = Path("src/sift/executor.py").read_text(encoding="utf-8")
    assert "(deny network*)" in executor


# --------------------------------------------------------------------
# Read-only enforcement
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    "statement",
    [
        "DROP TABLE patients",
        "INSERT INTO patients VALUES (1,2,'x')",
        "UPDATE patients SET age = 0",
        "DELETE FROM patients",
        "SELECT 1; DROP TABLE patients",  # stacked
        "SELECT * INTO copied_patients FROM patients",
        "SELECT * FROM patients INTO OUTFILE '/tmp/patients.csv'",
        "SELECT * FROM patients -- \n; DELETE FROM patients",  # comment-hidden
        "/* hide */ DELETE FROM patients",  # block comment
        "ALTER TABLE patients ADD COLUMN x INT",
        "CREATE TABLE evil (a INT)",
        "ATTACH DATABASE '/tmp/x.db' AS x",
        "PRAGMA journal_mode=WAL",
        "PRAGMA writable_schema=ON",
        "EXPLAIN ANALYZE DELETE FROM patients",
        "SHOW TABLES",
        "DESCRIBE patients",
        "",
    ],
)
def test_write_statements_refused(tmp_path: Path, db: Path, statement) -> None:
    with pytest.raises(ConnectorError):
        run_extract(
            tmp_path, connection=f"sqlite:///{db}", sql=statement, dataset_name="evil"
        )
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM patients").fetchone()[0] == 1000
    con.close()


@pytest.mark.parametrize(
    "statement",
    [
        "SELECT * FROM patients",
        "WITH x AS (SELECT * FROM patients) SELECT * FROM x",
        "SELECT dx, COUNT(*) AS n FROM patients GROUP BY dx",
        "SELECT * FROM patients;",  # trailing semicolon is fine
    ],
)
def test_read_statements_allowed(tmp_path: Path, db: Path, statement) -> None:
    result = run_extract(
        tmp_path, connection=f"sqlite:///{db}", sql=statement, dataset_name="ok"
    )
    assert result.rows > 0


# --------------------------------------------------------------------
# normalize_sql: syntactically read-only SELECTs that call a known
# dangerous function (filesystem / network / admin escape, or a
# resource-exhaustion primitive) entirely server-side. No top-level
# write KEYWORD appears in this text, so ``_WRITE_TOKENS`` alone
# never caught these -- the gap this closes.
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    "statement",
    [
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT pg_read_binary_file('/etc/shadow')",
        "SELECT * FROM pg_ls_dir('/tmp')",
        "SELECT lo_export(loid, '/tmp/exfil') FROM objects",
        "SELECT lo_import('/etc/passwd')",
        "SELECT dblink_exec('dbname=x', 'DELETE FROM t')",
        "SELECT dblink_connect('host=evil.example')",
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity",
        "SELECT pg_advisory_lock(42)",
        "SELECT pg_try_advisory_lock_shared(42)",
        "SELECT pg_notify('research', 'payload')",
        "SELECT nextval('study_sequence')",
        "SELECT set_config('log_statement', 'none', false)",
        "SELECT pg_sleep(999999)",
        "SELECT LOAD_FILE('/etc/passwd')",
        "SELECT * FROM t WHERE 1=1 OR sys_exec('id') = 0",
        "SELECT * FROM OPENROWSET('SQLNCLI', 'exec xp_cmdshell('dir')')",
        "SELECT SLEEP(30)",
        "SELECT BENCHMARK(50000000, MD5('x'))",
    ],
)
def test_normalize_sql_refuses_dangerous_function_calls(statement) -> None:
    assert normalize_sql(statement) is None


@pytest.mark.parametrize(
    "statement",
    [
        "SELECT * FROM patients",
        "SELECT dx, COUNT(*) FROM patients GROUP BY dx",
        # A column merely named after a dangerous function, with no call
        # (no immediately-following paren), must not false-positive.
        "SELECT sleep_duration, benchmark_score FROM patients",
        # A string literal that happens to contain a dangerous-function-
        # looking substring, but not as a real call, stays allowed --
        # this function only rejects an actual `name(` call shape.
        "SELECT * FROM patients WHERE notes = 'see sleep study results'",
    ],
)
def test_normalize_sql_does_not_false_positive_on_lookalikes(statement) -> None:
    assert normalize_sql(statement) is not None


@pytest.mark.parametrize(
    "statement",
    [
        "SELECT * FROM patients FOR UPDATE",
        "SELECT * FROM patients FOR NO KEY UPDATE",
        "SELECT * FROM patients FOR SHARE",
        "SELECT * FROM patients FOR KEY SHARE",
        "SELECT * FROM patients LOCK IN SHARE MODE",
    ],
)
def test_normalize_sql_refuses_locking_reads(statement: str) -> None:
    assert normalize_sql(statement) is None


@pytest.mark.parametrize(
    "statement",
    [
        "SELECT '-- not a comment; DROP TABLE x' AS note",
        "SELECT 'sleep(5); /* still data */' AS note",
        'SELECT "delete" FROM "research results"',
        "SELECT [update] FROM [clinical data]",
        "SELECT `drop` FROM `lab results`",
        "SELECT $$; DROP TABLE x; sleep(5)$$ AS note",
        "SELECT $study$-- comment-looking data$study$ AS note",
        "SELECT q'[; DROP TABLE x; sleep(5)]' AS note FROM dual",
        "SELECT q'!/* not a comment */; DELETE!' AS note FROM dual",
    ],
)
def test_normalize_sql_understands_quoted_regions(statement: str) -> None:
    """Scanner controls apply to code, never values or identifiers."""
    assert normalize_sql(statement) == statement


@pytest.mark.parametrize(
    "statement",
    [
        "SELECT 'unterminated",
        'SELECT "unterminated',
        "SELECT [unterminated",
        "SELECT /* unterminated",
        "SELECT $study$unterminated",
        "SELECT 1; -- trailing text changes if semicolon is removed",
    ],
)
def test_normalize_sql_refuses_ambiguous_lexical_input(statement: str) -> None:
    assert normalize_sql(statement) is None


def test_normalize_sql_preserves_comments_and_ignores_their_tokens() -> None:
    statement = "/* DROP; */ SELECT 1 AS value -- sleep(5); DELETE"
    assert normalize_sql(statement) == statement


def test_quoted_sql_text_round_trips_through_sqlite(
    tmp_path: Path,
    db: Path,
) -> None:
    value = "-- literal; /* literal */ DROP TABLE patients; sleep(5)"
    result = run_extract(
        tmp_path,
        connection=f"sqlite:///{db}",
        sql=f"SELECT '{value}' AS note",
        dataset_name="quoted",
    )
    assert pd.read_parquet(result.dataset_path).iloc[0]["note"] == value


def test_run_extract_refuses_dangerous_function_call_before_touching_db(
    tmp_path: Path,
    db: Path,
) -> None:
    """End-to-end: the SAME gate ``run_extract`` uses must reject a
    dangerous-function SELECT with the "read-only statement" message
    -- confirming this is the SQL-normalization gate rejecting it,
    not a downstream database error (which would carry a different
    message and, more importantly, would mean the call attempt
    reached the database at all)."""
    with pytest.raises(ConnectorError) as exc:
        run_extract(
            tmp_path,
            connection=f"sqlite:///{db}",
            sql="SELECT load_extension('/tmp/evil.so')",
            dataset_name="evil",
        )
    assert "read-only statement" in str(exc.value)


# --------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------


def test_password_redacted_everywhere() -> None:
    uri = "postgresql://alice:hunter2@db.uni.edu:5432/research"
    shown = redact_connection(uri)
    assert "hunter2" not in shown
    assert "alice" in shown and "db.uni.edu" in shown


def test_credentials_never_reach_the_ledger(tmp_path: Path, db: Path) -> None:
    from sift import release_ledger

    (tmp_path / ".sift").mkdir()
    run_extract(
        tmp_path,
        connection=f"sqlite:///{db}",
        sql="SELECT * FROM patients",
        dataset_name="cohort",
    )
    blob = json.dumps(release_ledger.read_ledger(tmp_path))
    assert "hunter2" not in blob
    records = release_ledger.read_ledger(tmp_path)
    assert records[0]["kind"] == "local_ingestion"
    assert records[0]["extra"]["backend"] == "sqlite"


# --------------------------------------------------------------------
# Materialization and downstream behaviour
# --------------------------------------------------------------------


def test_extract_becomes_a_first_class_dataset(tmp_path: Path, db: Path) -> None:
    from sift.dataset_profile import profile_dataset
    from sift.schema import extract as schema_extract
    from sift.schema import row_count

    (tmp_path / ".sift").mkdir()
    result = run_extract(
        tmp_path,
        connection=f"sqlite:///{db}",
        sql="SELECT * FROM patients",
        dataset_name="cohort",
    )
    ds = result.dataset_path
    assert ds.suffix == ".parquet" and ds.parent == tmp_path
    assert [v["name"] for v in schema_extract(ds, "names_types")["variables"]] == [
        "id",
        "age",
        "dx",
    ]
    assert row_count(ds) == 1000
    assert profile_dataset(ds)["ok"] is True


def test_connection_check_reads_no_dataset_rows(tmp_path: Path, db: Path) -> None:
    result = check_connection(tmp_path, connection=f"sqlite:///{db}")
    assert result.backend == "sqlite"
    assert result.latency_ms >= 0
    assert result.read_only_enforcement == "database_session_and_query_gate"
    assert result.connection_display.endswith("hospital.db")


def test_catalog_is_bounded_metadata_only_and_columns_are_opt_in(
    tmp_path: Path,
    db: Path,
) -> None:
    shallow = inspect_database(tmp_path, connection=f"sqlite:///{db}")
    patients = next(obj for obj in shallow.objects if obj["name"] == "patients")
    assert patients == {
        "schema": "main", "name": "patients", "kind": "table", "columns": [],
    }

    detailed = inspect_database(
        tmp_path,
        connection=f"sqlite:///{db}",
        schema="main",
        object_name="patients",
    )
    patients = next(obj for obj in detailed.objects if obj["name"] == "patients")
    assert [column["name"] for column in patients["columns"]] == ["id", "age", "dx"]
    assert all(set(column) == {"name", "type", "nullable"} for column in patients["columns"])

    with pytest.raises(ConnectorError, match="no database object"):
        inspect_database(
            tmp_path,
            connection=f"sqlite:///{db}",
            object_name="not_a_real_table",
        )


def test_duckdb_file_has_simple_source_view_and_relative_path(
    tmp_path: Path,
) -> None:
    pq = tmp_path / "source-data.parquet"
    pd.DataFrame({"amount": [1, 2, 3]}).to_parquet(pq)
    result = run_extract(
        tmp_path,
        connection=pq.name,
        sql="SELECT SUM(amount) AS total FROM source",
        dataset_name="source_rollup",
    )
    assert pd.read_parquet(result.dataset_path).iloc[0]["total"] == 6
    catalog = inspect_database(
        tmp_path,
        connection=pq.name,
        object_name="source",
    )
    assert catalog.objects[0]["columns"][0]["name"] == "amount"


def test_extract_provenance_hashes_are_reproducible_and_local(
    tmp_path: Path,
    db: Path,
) -> None:
    import hashlib

    from sift import release_ledger

    (tmp_path / ".sift").mkdir()
    result = run_extract(
        tmp_path,
        connection=f"sqlite:///{db}",
        sql="SELECT id, age FROM patients",
        dataset_name="audit",
    )
    assert result.query_sha256 == hashlib.sha256(
        b"SELECT id, age FROM patients"
    ).hexdigest()
    assert result.dataset_sha256 == hashlib.sha256(
        result.dataset_path.read_bytes()
    ).hexdigest()
    extra = release_ledger.read_ledger(tmp_path)[0]["extra"]
    assert extra["query_sha256"] == result.query_sha256
    assert extra["dataset_sha256"] == result.dataset_sha256
    assert extra["canonical_fingerprint"] == result.canonical_fingerprint
    assert "SELECT id" not in json.dumps(extra)

    # An ordinary later analysis must retain the connector's approved-query
    # identity instead of replacing the current manifest with selection={}. 
    from sift.tools import _canonicalize_analysis_sources
    sources, error = _canonicalize_analysis_sources(
        tmp_path, (result.dataset_path.relative_to(tmp_path).as_posix(),),
    )
    assert error is None
    assert sources[0]["selection"]["query_sha256"] == result.query_sha256
    assert sources[0]["selection"]["extraction_scope"] == "bounded_read_only_query"


def test_duplicate_result_columns_are_canonicalized_and_recorded(
    tmp_path: Path,
    db: Path,
) -> None:
    from sift import release_ledger

    (tmp_path / ".sift").mkdir()
    result = run_extract(
        tmp_path,
        connection=f"sqlite:///{db}",
        sql=(
            "SELECT a.id, b.id, a.id AS id__2 FROM patients a "
            "JOIN patients b ON a.id = b.id LIMIT 1"
        ),
        dataset_name="duplicate_columns",
    )
    assert list(pd.read_parquet(result.dataset_path).columns) == [
        "id", "id__2", "id__2__2",
    ]
    assert result.column_renames == (
        {"position": 1, "original": "id", "materialized": "id__2"},
        {"position": 2, "original": "id__2", "materialized": "id__2__2"},
    )
    from sift.canonical_dataset import manifest_path
    assert result.canonical_fingerprint is not None
    manifest = json.loads(
        manifest_path(tmp_path, result.canonical_fingerprint).read_text(encoding="utf-8")
    )
    assert [row["original_name"] for row in manifest["columns"]] == [
        "id", "id", "id__2",
    ]
    assert [row["materialized_name"] for row in manifest["columns"]] == [
        "id", "id__2", "id__2__2",
    ]
    extra = release_ledger.read_ledger(tmp_path)[0]["extra"]
    assert extra["column_renames"] == list(result.column_renames)


def test_failed_required_provenance_removes_materialized_extract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Path,
) -> None:
    from sift import release_ledger

    monkeypatch.setattr(release_ledger, "record_release", lambda *a, **k: False)
    with pytest.raises(ConnectorError, match="record extract provenance"):
        run_extract(
            tmp_path,
            connection=f"sqlite:///{db}",
            sql="SELECT * FROM patients",
            dataset_name="must_be_tracked",
        )
    assert not list(tmp_path.glob("must_be_tracked*.parquet"))
    canonical = tmp_path / ".sift" / "datasets"
    assert not list((canonical / "manifests").glob("*.json"))
    assert not list((canonical / "paths").glob("*.json"))
    assert not list((canonical / "snapshots").rglob("*.parquet"))


def test_failed_provenance_surfaces_incomplete_confidential_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Path,
) -> None:
    from sift import canonical_dataset, release_ledger

    monkeypatch.setattr(release_ledger, "record_release", lambda *a, **k: False)
    monkeypatch.setattr(
        canonical_dataset, "discard_uncommitted_manifest", lambda *a, **k: False,
    )
    with pytest.raises(ConnectorError, match="cleanup was incomplete"):
        run_extract(
            tmp_path,
            connection=f"sqlite:///{db}",
            sql="SELECT * FROM patients",
            dataset_name="cleanup_failure",
        )


def test_audit_failure_preserves_release_committed_extract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Path,
) -> None:
    from sift import integration_audit

    monkeypatch.setattr(
        integration_audit, "record_integration_event", lambda *a, **k: False,
    )
    with pytest.raises(ConnectorError, match="record extract provenance"):
        run_extract(
            tmp_path,
            connection=f"sqlite:///{db}",
            sql="SELECT * FROM patients",
            dataset_name="audit_failed",
        )
    extracts = list(tmp_path.glob("audit_failed*.parquet"))
    assert len(extracts) == 1
    assert extracts[0].with_suffix(".parquet.metadata.json").is_file()
    assert list((tmp_path / ".sift" / "datasets" / "manifests").glob("*.json"))


def test_duckdb_extract_honors_byte_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from sift import connectors

    source = tmp_path / "wide.parquet"
    pd.DataFrame({"text": ["x" * 4096 for _ in range(100)]}).to_parquet(source)
    monkeypatch.setattr(connectors, "DEFAULT_BYTE_LIMIT", 1024)
    with pytest.raises(ConnectorError, match="in-memory safety limit"):
        connectors.run_extract(
            tmp_path,
            connection=str(source),
            sql="SELECT * FROM source",
            dataset_name="too_wide",
        )
    assert not list(tmp_path.glob("too_wide*.parquet"))


def test_database_timeout_configuration_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sift.connectors import database_query_timeout_seconds

    for raw, expected in [
        ("", 300), ("not-a-number", 300), ("0", 1), ("-10", 1),
        ("999999", 3600), ("42", 42),
    ]:
        monkeypatch.setenv("SIFT_DATABASE_QUERY_TIMEOUT_SECONDS", raw)
        assert database_query_timeout_seconds() == expected


def test_database_connect_timeout_configuration_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for raw, expected in [
        ("", 30), ("not-a-number", 30), ("0", 1), ("-10", 1),
        ("999999", 120), ("42", 42),
    ]:
        monkeypatch.setenv("SIFT_DATABASE_CONNECT_TIMEOUT_SECONDS", raw)
        assert connectors.database_connect_timeout_seconds() == expected


@pytest.mark.parametrize(
    ("connection", "backend", "expected"),
    [
        ("postgresql+psycopg://u:p@host/db", "postgresql", {
            "connect_timeout": 17,
        }),
        ("mysql+pymysql://u:p@host/db", "mysql", {
            "connect_timeout": 17, "read_timeout": 17, "write_timeout": 17,
        }),
        ("mssql+pyodbc://u:p@host/db", "mssql", {"timeout": 17}),
        ("oracle+oracledb://u:p@host/db", "oracle", {
            "tcp_connect_timeout": 17.0,
        }),
        ("redshift+redshift_connector://u:p@host/db", "redshift", {
            "timeout": 17,
        }),
        ("snowflake://u:p@account/db", "snowflake", {
            "login_timeout": 17, "network_timeout": 17, "socket_timeout": 17,
        }),
        ("databricks://token:p@host/db", "databricks", {
            "_socket_timeout": 17.0,
        }),
    ],
)
def test_reviewed_database_drivers_receive_native_network_timeouts(
    connection: str,
    backend: str,
    expected: dict[str, object],
) -> None:
    assert connectors._connection_timeout_args(connection, backend, 17) == expected


def test_unbounded_sql_server_driver_is_refused() -> None:
    with pytest.raises(ConnectorError, match="pyodbc"):
        connectors._connection_timeout_args(
            "mssql+unknown://u:p@host/db", "mssql", 17,
        )


@pytest.mark.parametrize(
    ("connection", "backend", "guidance"),
    [
        ("postgresql+pg8000://u:p@host/db", "postgresql", "psycopg"),
        ("mysql+mysqlconnector://u:p@host/db", "mysql", "PyMySQL"),
        ("mssql+pymssql://u:p@host/db", "mssql", "pyodbc"),
        ("oracle+cx_oracle://u:p@host/db", "oracle", "oracledb"),
        ("redshift+psycopg2://u:p@host/db", "redshift", "redshift_connector"),
        ("databricks+unknown://u:p@host/db", "databricks", "databricks-sqlalchemy"),
    ],
)
def test_unreviewed_driver_without_proven_deadline_is_refused(
    connection: str,
    backend: str,
    guidance: str,
) -> None:
    with pytest.raises(ConnectorError, match=guidance):
        connectors._connection_timeout_args(connection, backend, 17)


def test_engine_native_timeout_is_capped_by_outer_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sqlalchemy

    observed: dict[str, object] = {}
    sentinel = object()

    def create_engine(_connection: str, **kwargs):
        observed.update(kwargs)
        return sentinel

    monkeypatch.setattr(sqlalchemy, "create_engine", create_engine)
    monkeypatch.setenv("SIFT_DATABASE_CONNECT_TIMEOUT_SECONDS", "30")
    assert connectors._create_bounded_engine(
        "postgresql+psycopg://u:private@host/db",
        "postgresql",
        timeout_cap_seconds=7.9,
    ) is sentinel
    assert observed["connect_args"] == {"connect_timeout": 7}
    # Only the engine sees the real URI; policy kwargs contain no secrets.
    assert "private" not in repr(observed)


def test_bigquery_billing_project_is_forwarded_to_current_dialect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sqlalchemy

    observed: dict[str, object] = {}
    sentinel = object()
    monkeypatch.setattr(
        sqlalchemy,
        "create_engine",
        lambda uri, **kwargs: observed.update(uri=uri, **kwargs) or sentinel,
    )
    connection = "bigquery://data-project/dataset?billing_project_id=billing-project"
    assert connectors._create_bounded_engine(connection, "bigquery") is sentinel
    assert observed == {
        "uri": "bigquery://data-project/dataset",
        "billing_project_id": "billing-project",
    }


def test_bigquery_sdk_calls_receive_remaining_deadline_and_no_retries() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class Job:
        def result(self, **kwargs):
            calls.append(("result", kwargs))
            return object()

        def cancel(self, **kwargs):
            calls.append(("cancel", kwargs))
            return True

    class Client:
        project = "research"

        def query(self, *_args, **kwargs):
            calls.append(("query", kwargs))
            return Job()

        def list_tables(self, *_args, **kwargs):
            calls.append(("list_tables", kwargs))
            return []

    bounded = connectors._BoundedBigQueryClient(Client(), 4.5)
    bounded.query("SELECT 1", timeout=20)
    bounded.query_and_wait("SELECT 1")
    bounded.list_tables("dataset")
    assert calls == [
        ("query", {"timeout": 4.5, "retry": None, "job_retry": None}),
        ("query", {
            "timeout": 4.5, "retry": None, "job_retry": None,
        }),
        ("result", {
            "page_size": None, "max_results": None, "timeout": 4.5,
            "retry": None, "job_retry": None,
        }),
        ("list_tables", {"timeout": 4.5, "retry": None}),
    ]


def test_bigquery_interrupt_requests_provider_side_job_cancellation() -> None:
    class Client:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel_active_job(self) -> bool:
            self.cancelled = True
            return True

    client = Client()
    raw = type("Raw", (), {"_client": client})()
    assert connectors._interrupt_driver_operation(None, raw) == "bigquery_job_cancel"
    assert client.cancelled is True


def test_structured_authentication_connect_args_are_merged_without_uri_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sqlalchemy

    observed: dict[str, object] = {}
    sentinel = object()
    monkeypatch.setattr(
        sqlalchemy,
        "create_engine",
        lambda uri, **kwargs: observed.update(uri=uri, **kwargs) or sentinel,
    )
    spec = connectors.databricks_oauth_connection(
        "databricks://user@host:443?http_path=/sql/warehouse",
        mode="oauth_m2m",
        client_id="client-id",
        client_secret="super-secret",
    )
    assert connectors._create_bounded_engine(spec, "databricks") is sentinel
    assert observed["uri"] == spec.uri
    connect_args = observed["connect_args"]
    assert connect_args["_socket_timeout"] == 15.0
    provider = connect_args["credentials_provider"]
    assert provider.oauth_client_id == "client-id"
    assert provider.timeout_seconds == 15
    assert "super-secret" not in repr(spec)
    assert "super-secret" in connectors._connection_secrets(spec)
    with pytest.raises(TypeError):
        spec.connect_args["credentials_provider"] = object()  # type: ignore[index]


def test_databricks_m2m_provider_bounds_sdk_discovery_and_hides_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from databricks.sdk import core

    observed: dict[str, object] = {}
    monkeypatch.setattr(
        core, "Config",
        lambda **kwargs: observed.update(kwargs) or object(),
    )
    sentinel = object()
    monkeypatch.setattr(
        core, "oauth_service_principal", lambda config: (config, sentinel),
    )
    spec = connectors.databricks_oauth_connection(
        "databricks://user@workspace:443?http_path=/sql/warehouse",
        mode="oauth_m2m", client_id="client-id", client_secret="client-secret",
    )
    provider = spec.connect_args["credentials_provider"].with_timeout(7)
    assert provider()[1] is sentinel
    assert observed == {
        "host": "https://workspace",
        "client_id": "client-id",
        "client_secret": "client-secret",
        "http_timeout_seconds": 7.0,
        "retry_timeout_seconds": 7,
    }
    assert "client-secret" not in repr(provider)


@pytest.mark.parametrize(
    "builder",
    (
        lambda: connectors.databricks_oauth_connection(
            "databricks://user@host?http_path=/sql/x&access_token=token",
            mode="oauth_u2m",
        ),
        lambda: connectors.databricks_oauth_connection(
            "databricks://user:token@host?http_path=/sql/x",
            mode="oauth_m2m", client_id="id", client_secret="secret",
        ),
        lambda: connectors.snowflake_key_pair_connection(
            "snowflake://user:password@account/db",
            private_key_pem="not reached",
        ),
    ),
)
def test_structured_authentication_rejects_ambiguous_uri_credentials(builder) -> None:
    with pytest.raises(ConnectorError, match="cannot be combined"):
        builder()


def test_snowflake_private_key_uri_secrets_are_redacted_and_rejected() -> None:
    uri = (
        "snowflake://user@account/db?private_key_file=/tmp/key.p8&"
        "private_key_file_pwd=hunter2&oauth_refresh_token=refresh-secret"
    )
    displayed = redact_connection(uri)
    assert "hunter2" not in displayed
    assert "refresh-secret" not in displayed
    with pytest.raises(ConnectorError, match="cannot be combined"):
        connectors.snowflake_key_pair_connection(
            uri, private_key_pem="not reached",
        )


def test_bounded_remote_catalog_places_limits_in_provider_queries() -> None:
    statements: list[tuple[str, dict[str, object]]] = []

    class Result:
        def __init__(self, rows: list[tuple[object, ...]]) -> None:
            self.rows = rows

        def fetchall(self):
            return self.rows

    class Connection:
        def execute(self, statement, params):
            rendered = str(statement)
            statements.append((rendered, params))
            if "schemata" in rendered:
                return Result([("public",)])
            return Result([("public", "fixture", "BASE TABLE")])

    schemas, objects, warnings = connectors._bounded_remote_catalog(
        Connection(), "postgresql",
        default_schema="public", selected_schema="public",
    )
    assert schemas == ["public"]
    assert objects == [("public", "fixture", "table")]
    assert warnings == []
    assert all("LIMIT" in statement for statement, _ in statements)
    assert statements[0][1]["limit"] == connectors.MAX_CATALOG_SCHEMAS + 1
    assert statements[1][1]["limit"] == connectors.MAX_CATALOG_OBJECTS + 1


def test_oracle_catalog_limits_use_portable_bound_rownum() -> None:
    statements: list[tuple[str, dict[str, object]]] = []

    class Result:
        def __init__(self, rows):
            self.rows = rows

        def fetchall(self):
            return self.rows

    class Connection:
        def execute(self, statement, params):
            rendered = str(statement)
            statements.append((rendered, params))
            if "all_users" in rendered:
                return Result([("RESEARCH",)])
            return Result([("RESEARCH", "FIXTURE", "TABLE")])

    schemas, objects, warnings = connectors._bounded_remote_catalog(
        Connection(), "oracle",
        default_schema="RESEARCH", selected_schema="RESEARCH",
    )
    assert schemas == ["RESEARCH"]
    assert objects == [("RESEARCH", "FIXTURE", "table")]
    assert warnings == []
    assert all("ROWNUM <= :limit" in sql for sql, _ in statements)
    assert all("FETCH FIRST" not in sql for sql, _ in statements)


def test_duckdb_file_column_discovery_is_bounded_at_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class Cursor:
        def fetchmany(self, count: int):
            observed["fetchmany"] = count
            return [
                (f"column_{index}", "VARCHAR", "YES")
                for index in range(count)
            ]

    class Connection:
        closed = False

        def execute(self, sql: str):
            assert sql == "DESCRIBE source"
            return Cursor()

        def close(self) -> None:
            self.closed = True

    connection = Connection()
    monkeypatch.setattr(connectors, "_open_duckdb", lambda *_: connection)
    _default, _schemas, objects, warnings = connectors._duckdb_catalog(
        "fixture.csv", "duckdb-file", schema=None, object_name="source",
    )
    assert observed["fetchmany"] == connectors.MAX_CATALOG_COLUMNS + 1
    assert len(objects[0]["columns"]) == connectors.MAX_CATALOG_COLUMNS
    assert warnings == ["column list truncated"]
    assert connection.closed is True


def test_bigquery_broad_catalog_fails_closed_without_identifier_interpolation() -> None:
    class Connection:
        def execute(self, *_args, **_kwargs):
            raise AssertionError("BigQuery broad discovery must not execute SQL")

    schemas, objects, warnings = connectors._bounded_remote_catalog(
        Connection(), "bigquery",
        default_schema=None, selected_schema="confidential-dataset",
    )
    assert schemas == ["confidential-dataset"]
    assert objects == []
    assert "disabled" in warnings[0]


def test_transport_evidence_attests_actual_databricks_cloudfetch_queue() -> None:
    cloudfetch_type = type(
        "CloudFetchQueue", (), {"__module__": "databricks.sql.utils"},
    )
    cursor = type("Cursor", (), {"active_result_set": type(
        "ResultSet", (), {"results": cloudfetch_type()},
    )()})()
    evidence = connectors._query_transport_evidence(cursor, "databricks")
    assert evidence.backend == "databricks"
    assert evidence.transport == "cloudfetch"
    assert connectors._query_transport_evidence(
        object(), "databricks",
    ).transport == "inline"


def test_bigquery_deadline_disables_and_closes_separate_storage_transport() -> None:
    class Storage:
        closed = False

        def close(self) -> None:
            self.closed = True

    class Client:
        pass

    storage = Storage()
    raw = type("Raw", (), {
        "_client": Client(),
        "_bqstorage_client": storage,
        "_owns_bqstorage_client": True,
    })()
    wrapper = type("Wrapper", (), {"driver_connection": raw})()
    conn = type("Connection", (), {"connection": wrapper})()
    connectors._configure_sdk_request_deadline(conn, "bigquery", 3.0)
    assert isinstance(raw._client, connectors._BoundedBigQueryClient)
    assert storage.closed is True
    assert raw._bqstorage_client is None
    assert raw._owns_bqstorage_client is False


def test_explicit_mysqlconnector_is_outside_reviewed_driver_set() -> None:
    with pytest.raises(ConnectorError, match="reviewed"):
        connectors._create_bounded_engine(
            "mysql+mysqlconnector://u:p@host/db", "mysql",
        )


def test_engine_helper_cannot_bypass_reviewed_driver_gate() -> None:
    with pytest.raises(ConnectorError, match="reviewed"):
        connectors._create_bounded_engine(
            "postgresql+asyncpg://u:p@host/db", "postgresql",
        )


def test_connect_timeout_is_typed_and_never_exposes_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Engine:
        def __init__(self) -> None:
            self.disposed = False

        def connect(self):
            raise TimeoutError("login timeout for hunter2")

        def dispose(self) -> None:
            self.disposed = True

    engine = Engine()
    monkeypatch.setattr(
        connectors, "_create_bounded_engine", lambda *_, **__: engine,
    )
    with pytest.raises(ConnectorError) as raised:
        check_connection(
            tmp_path,
            connection="postgresql://alice:hunter2@localhost/research",
        )
    assert raised.value.code == "deadline_exceeded"
    assert "hunter2" not in str(raised.value)
    assert engine.disposed is True


def test_cancellation_during_connect_waits_for_native_bound_then_stops_before_sql(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Sift does not abandon a credentialed connect attempt in a worker."""
    import threading

    from sift.integration_core import CancellationToken

    token = CancellationToken()
    connect_started = threading.Event()
    cancellation_sent = threading.Event()
    connect_finished = threading.Event()

    class Connection:
        def __enter__(self):
            connect_started.set()
            # Hold the simulated native connect until cancellation is known
            # to have been delivered. Wall-clock sleeps made this test depend
            # on hosted-runner thread scheduling and could let the foreground
            # thread reach SQL first even though the production boundary was
            # correct.
            assert cancellation_sent.wait(timeout=2)
            connect_finished.set()
            return self

        def __exit__(self, *_args) -> None:
            return None

        def execute(self, *_args):  # pragma: no cover - must not be reached
            raise AssertionError("SQL ran after cancellation")

    class Engine:
        dialect = type("Dialect", (), {"name": "postgresql"})()
        disposed = False

        def connect(self):
            return Connection()

        def dispose(self) -> None:
            self.disposed = True

    engine = Engine()
    monkeypatch.setattr(
        connectors, "_create_bounded_engine", lambda *_, **__: engine,
    )
    def cancel_during_connect() -> None:
        assert connect_started.wait(timeout=2)
        token.cancel()
        cancellation_sent.set()

    canceller = threading.Thread(target=cancel_during_connect)
    canceller.start()
    try:
        with pytest.raises(ConnectorError) as raised:
            check_connection(
                tmp_path,
                connection="postgresql://alice:secret@localhost/research",
                cancellation=token,
            )
    finally:
        canceller.join()
    assert raised.value.code == "cancelled"
    # The foreground call did not return while a hidden connect continued.
    assert connect_started.is_set()
    assert connect_finished.is_set()
    assert engine.disposed is True


def test_connection_check_deadline_interrupts_initial_driver_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import threading

    closed = threading.Event()

    class Raw:
        def close(self) -> None:
            closed.set()

    raw = Raw()

    class Connection:
        connection = type("Wrapper", (), {"driver_connection": raw})()

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def exec_driver_sql(self, *_args):
            return None

        def execute(self, *_args):
            assert closed.wait(0.5)
            raise TimeoutError("interrupted")

    class Engine:
        dialect = type("Dialect", (), {
            "name": "postgresql", "server_version_info": None,
        })()
        disposed = False

        def connect(self):
            return Connection()

        def dispose(self) -> None:
            self.disposed = True

    engine = Engine()
    monkeypatch.setattr(
        connectors, "_create_bounded_engine", lambda *_, **__: engine,
    )
    monkeypatch.setattr(connectors, "database_query_timeout_seconds", lambda: 0.02)
    with pytest.raises(ConnectorError) as raised:
        check_connection(
            tmp_path,
            connection="postgresql://alice:secret@localhost/research",
        )
    assert raised.value.code == "deadline_exceeded"
    assert closed.is_set()
    assert engine.disposed is True


def test_sqlite_query_deadline_interrupts_runaway_extract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Path,
) -> None:
    from sift import connectors

    monkeypatch.setattr(connectors, "database_query_timeout_seconds", lambda: 0.001)
    with pytest.raises(ConnectorError, match="deadline") as raised:
        connectors.run_extract(
            tmp_path,
            connection=f"sqlite:///{db}",
            sql=(
                "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL "
                "SELECT x + 1 FROM n WHERE x < 100000000) "
                "SELECT SUM(x) FROM n"
            ),
            dataset_name="runaway",
        )
    assert raised.value.code == "deadline_exceeded"
    assert not list(tmp_path.glob("runaway*.parquet"))


def test_row_limit_truncates_and_says_so(tmp_path: Path, db: Path) -> None:
    result = run_extract(
        tmp_path,
        connection=f"sqlite:///{db}",
        sql="SELECT * FROM patients",
        dataset_name="capped",
        row_limit=100,
    )
    assert result.rows == 100 and result.truncated is True


def test_large_sql_extract_streams_without_dataframe_concat(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """More than one fetch batch must reach Parquet without ever building a
    full-result dataframe in memory."""
    from sift import connectors

    db_path = tmp_path / "many.sqlite"
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE x (n INTEGER, value TEXT)")
    con.executemany(
        "INSERT INTO x VALUES (?, ?)",
        ((i, f"value-{i}") for i in range(connectors.FETCH_BATCH_ROWS * 2 + 17)),
    )
    con.commit()
    con.close()

    monkeypatch.setattr(
        pd, "concat",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("database extraction must not concatenate batches")
        ),
    )
    result = connectors.run_extract(
        tmp_path,
        connection=f"sqlite:///{db_path}",
        sql="SELECT * FROM x ORDER BY n",
        dataset_name="streamed",
    )
    extracted = pd.read_parquet(result.dataset_path)
    assert result.rows == connectors.FETCH_BATCH_ROWS * 2 + 17
    assert extracted.iloc[0]["n"] == 0
    assert extracted.iloc[-1]["n"] == result.rows - 1


def test_streaming_schema_handles_null_only_first_batch(tmp_path: Path) -> None:
    """A null-only early batch must not force a later numeric column to the
    Arrow null type or lose its values during the final merge."""
    from sift import connectors

    db_path = tmp_path / "late_type.sqlite"
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE x (position INTEGER, late_value INTEGER)")
    total = connectors.FETCH_BATCH_ROWS + 25
    con.executemany(
        "INSERT INTO x VALUES (?, ?)",
        ((i, None if i < connectors.FETCH_BATCH_ROWS else i) for i in range(total)),
    )
    con.commit()
    con.close()

    result = connectors.run_extract(
        tmp_path,
        connection=f"sqlite:///{db_path}",
        sql="SELECT * FROM x ORDER BY position",
        dataset_name="late_type",
    )
    extracted = pd.read_parquet(result.dataset_path)
    assert len(extracted) == total
    assert extracted["late_value"].iloc[:connectors.FETCH_BATCH_ROWS].isna().all()
    assert extracted["late_value"].iloc[-1] == total - 1


def test_streaming_extract_preserves_disk_reserve_and_cleans_parts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Path,
) -> None:
    from types import SimpleNamespace
    from sift import connectors

    monkeypatch.setattr(
        connectors.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=0),
    )
    with pytest.raises(ConnectorError, match="free-space safety reserve"):
        connectors.run_extract(
            tmp_path,
            connection=f"sqlite:///{db}",
            sql="SELECT * FROM patients",
            dataset_name="no_disk",
        )
    assert not list(tmp_path.glob("no_disk*.parquet"))
    assert not list(tmp_path.glob(".sift-extract-*.parquet"))
    assert not list(tmp_path.glob(".sift-query-parts-*"))


def test_sqlalchemy_extract_executes_original_query_without_limit_wrapper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The connector must not append dialect-specific LIMIT syntax.

    SQL Server, among others, does not accept LIMIT.  Row bounding is a
    client-side fetch concern now, so every backend receives the exact
    normalized statement the researcher approved.
    """
    from sift import connectors
    from sqlalchemy.engine import Connection

    seen: list[str] = []
    real_exec_driver_sql = Connection.exec_driver_sql

    def capture_driver_sql(self, statement: str, *args, **kwargs):
        if statement.lstrip().upper().startswith("SELECT"):
            seen.append(statement)
        return real_exec_driver_sql(self, statement, *args, **kwargs)

    monkeypatch.setattr(Connection, "exec_driver_sql", capture_driver_sql)
    db_path = tmp_path / "portable.sqlite"
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE x (n INTEGER)")
    con.executemany("INSERT INTO x VALUES (?)", [(i,) for i in range(5)])
    con.commit()
    con.close()

    result = connectors.run_extract(
        tmp_path,
        connection=f"sqlite:///{db_path}",
        sql="SELECT * FROM x;",
        dataset_name="portable",
        row_limit=2,
    )
    assert result.rows == 2 and result.truncated is True
    assert seen == ["SELECT * FROM x"]


def test_duckdb_over_a_parquet_file(tmp_path: Path) -> None:
    """The warehouse-shaped path: SQL over files, no server."""
    pq = tmp_path / "claims.parquet"
    pd.DataFrame({"amt": range(500), "grp": ["a", "b"] * 250}).to_parquet(pq)
    result = run_extract(
        tmp_path,
        connection=str(pq),
        sql="SELECT grp, SUM(amt) AS total FROM source GROUP BY grp",
        dataset_name="rollup",
    )
    assert result.rows == 2 and result.backend == "duckdb-file"


def test_name_collisions_do_not_overwrite(tmp_path: Path, db: Path) -> None:
    a = run_extract(
        tmp_path,
        connection=f"sqlite:///{db}",
        sql="SELECT * FROM patients",
        dataset_name="cohort",
    )
    b = run_extract(
        tmp_path,
        connection=f"sqlite:///{db}",
        sql="SELECT * FROM patients",
        dataset_name="cohort",
    )
    assert a.dataset_path != b.dataset_path
    assert a.dataset_path.exists() and b.dataset_path.exists()


def test_concurrent_name_collisions_do_not_overwrite(
    tmp_path: Path,
    db: Path,
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    def extract_one(_index: int):
        return run_extract(
            tmp_path,
            connection=f"sqlite:///{db}",
            sql="SELECT * FROM patients",
            dataset_name="simultaneous",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(extract_one, range(16)))
    paths = [result.dataset_path for result in results]
    assert len(set(paths)) == 16
    assert all(path.exists() for path in paths)
    assert all(len(pd.read_parquet(path)) == 1000 for path in paths)


def test_dataset_name_cannot_traverse(tmp_path: Path, db: Path) -> None:
    result = run_extract(
        tmp_path,
        connection=f"sqlite:///{db}",
        sql="SELECT * FROM patients",
        dataset_name="../../escape",
    )
    assert result.dataset_path.parent == tmp_path
    assert ".." not in result.dataset_path.name


# --------------------------------------------------------------------
# Backends and failure messages
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    "conn,expected",
    [
        ("postgresql://u@h/db", "postgresql"),
        ("postgres://u@h/db", "postgresql"),
        ("mysql+pymysql://u@h/db", "mysql"),
        ("mariadb+pymysql://u@h/db", "mariadb"),
        ("mssql+pyodbc://u@h/db", "mssql"),
        ("snowflake://u@acct/db", "snowflake"),
        ("/data/w.duckdb", "duckdb"),
        ("/data/x.sqlite", "sqlite"),
        ("/data/y.parquet", "duckdb-file"),
    ],
)
def test_backend_detection(conn, expected) -> None:
    assert describe_backend(conn) == expected


def test_unknown_connection_gets_an_actionable_message() -> None:
    with pytest.raises(ConnectorError) as exc:
        describe_backend("just some text")
    assert "SQLAlchemy URI" in str(exc.value)


def test_unknown_sqlalchemy_dialect_fails_closed() -> None:
    with pytest.raises(ConnectorError, match="not supported"):
        describe_backend("customdb://user:password@host/research")


@pytest.mark.parametrize(
    "connection",
    [
        "postgresql://user@trusted.example/research\n@other.example/db",
        "sqlite:////tmp/research\t.db",
        "postgresql://user@host/db\x7f",
    ],
)
def test_connection_inputs_reject_parser_normalized_controls(connection: str) -> None:
    with pytest.raises(ConnectorError, match="control"):
        describe_backend(connection)
    with pytest.raises(ConnectorError, match="control"):
        validate_connection_security(connection, "postgresql")


def test_driver_errors_redact_exact_connection_secrets() -> None:
    from sift.connectors import _safe_connector_error

    connection = (
        "postgresql://alice:novel-secret@db.example/research"
        "?sslmode=verify-full&access_token=token-secret"
    )
    shown = _safe_connector_error(
        RuntimeError(
            "authentication failed for novel-secret; reflected token-secret"
        ),
        connection,
    )
    assert "novel-secret" not in shown
    assert "token-secret" not in shown


def test_driver_errors_redact_structured_private_key_bytes() -> None:
    from sift.connectors import ConnectionSpec, _safe_connector_error

    private_key = b"private-key-material-that-must-never-be-displayed"
    connection = ConnectionSpec(
        "snowflake://u@account/db?ocsp_fail_open=false",
        authentication="key_pair",
        connect_args={"private_key": private_key},
    )
    shown = _safe_connector_error(
        RuntimeError(f"driver echoed native argument {private_key!r}"),
        connection,
    )
    assert "private-key-material" not in shown
    assert "***" in shown


@pytest.mark.parametrize(
    "operation",
    (
        lambda path, connection: check_connection(path, connection=connection),
        lambda path, connection: inspect_database(path, connection=connection),
        lambda path, connection: run_extract(
            path,
            connection=connection,
            sql="SELECT 1",
            dataset_name="x",
        ),
    ),
)
def test_missing_driver_message_names_exact_sift_extra(
    tmp_path: Path,
    operation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every database entry point gives the same safe recovery instruction."""
    def missing_driver(*_args, **_kwargs):
        raise connectors._engine_creation_error(
            ModuleNotFoundError(), "postgresql"
        )

    def start_query(batches, *_args, **_kwargs):
        # ``_run_sqlalchemy`` is a generator; advance it before the Parquet
        # materializer imports Arrow so this remains a driver-guidance test
        # even in a deliberately minimal test environment.
        next(iter(batches))
        raise AssertionError("missing driver did not stop query setup")

    monkeypatch.setattr(connectors, "_create_bounded_engine", missing_driver)
    monkeypatch.setattr(connectors, "_materialize_query_batches", start_query)
    connection = "postgresql+psycopg://u:p@localhost/db"
    with pytest.raises(ConnectorError) as exc:
        operation(tmp_path, connection)
    msg = str(exc.value)
    assert 'pip install "sift[postgres]"' in msg
    assert "p@localhost" not in msg  # no credentials in the error


@pytest.mark.parametrize(
    "backend",
    tuple(
        adapter.id
        for adapter in DATABASE_ADAPTERS
        if adapter.install_extra != "built-in"
    ),
)
def test_engine_setup_guidance_uses_catalog_extra(backend: str) -> None:
    adapter = next(item for item in DATABASE_ADAPTERS if item.id == backend)
    message = str(connectors._engine_creation_error(ModuleNotFoundError(), backend))
    assert f'sift[{adapter.install_extra}]' in message
    assert adapter.label in message


@pytest.mark.parametrize(
    ("connection", "backend", "expected"),
    [
        ("postgresql://u:p@localhost/db", "postgresql", "postgresql+psycopg://u:p@localhost/db"),
        ("postgres://u:p@localhost/db", "postgresql", "postgresql+psycopg://u:p@localhost/db"),
        (
            "postgresql+psycopg://u:p@localhost/db",
            "postgresql",
            "postgresql+psycopg://u:p@localhost/db",
        ),
        ("mysql://u:p@localhost/db", "mysql", "mysql+pymysql://u:p@localhost/db"),
        ("mariadb://u:p@localhost/db", "mariadb", "mariadb+pymysql://u:p@localhost/db"),
        ("oracle://u:p@localhost/db", "oracle", "oracle+oracledb://u:p@localhost/db"),
        ("oracle+oracledb://u:p@localhost/db", "oracle", "oracle+oracledb://u:p@localhost/db"),
        (
            "redshift://u:p@localhost/db",
            "redshift",
            "redshift+redshift_connector://u:p@localhost/db",
        ),
    ],
)
def test_plain_database_uris_use_declared_reviewed_driver(
    tmp_path: Path, connection: str, backend: str, expected: str,
) -> None:
    observed_backend, effective = connectors._prepare_connection(tmp_path, connection)
    assert observed_backend == backend
    assert effective == expected


@pytest.mark.parametrize(
    ("connection", "backend"),
    [
        ("postgresql+psycopg2://u:p@localhost/db", "postgresql"),
        ("mysql+mysqlconnector://u:p@localhost/db", "mysql"),
        ("mariadb+mysqlconnector://u:p@localhost/db", "mariadb"),
        ("mssql+pymssql://u:p@localhost/db", "mssql"),
        ("oracle+cx_oracle://u:p@localhost/db", "oracle"),
        ("snowflake+unknown://u:p@localhost/db", "snowflake"),
        ("bigquery+unknown://project", "bigquery"),
        ("redshift+psycopg2://u:p@localhost/db", "redshift"),
        ("databricks+unknown://u:p@localhost/db", "databricks"),
    ],
)
def test_explicit_unreviewed_database_drivers_are_refused(
    tmp_path: Path,
    connection: str,
    backend: str,
) -> None:
    with pytest.raises(ConnectorError, match="reviewed"):
        connectors._prepare_connection(tmp_path, connection)


@pytest.mark.parametrize("backend", ["mysql", "mariadb"])
def test_pymysql_custom_ca_flags_are_normalized_without_losing_verification(
    tmp_path: Path,
    backend: str,
) -> None:
    from sqlalchemy import create_engine

    connection = (
        f"{backend}+pymysql://u:p@db.example/research?"
        "ssl_ca=/private/ca.pem&ssl_verify_cert=true&"
        "ssl_verify_identity=true"
    )
    observed_backend, effective = connectors._prepare_connection(
        tmp_path, connection
    )

    assert observed_backend == backend
    assert "ssl_verify_cert" not in effective
    assert "ssl_verify_identity" not in effective
    assert "ssl_check_hostname=true" in effective
    engine = create_engine(effective)
    try:
        _args, kwargs = engine.dialect.create_connect_args(engine.url)
        assert kwargs["ssl"] == {
            "ca": "/private/ca.pem",
            "check_hostname": True,
        }
        assert "ssl_verify_cert" not in kwargs
        assert "ssl_verify_identity" not in kwargs
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    ("connection", "expected_dialect", "expected_driver"),
    [
        ("postgresql://u:p@localhost/db", "postgresql", "psycopg"),
        ("mysql://u:p@localhost/db", "mysql", "pymysql"),
        ("mariadb://u:p@localhost/db", "mariadb", "pymysql"),
        ("oracle://u:p@localhost/db", "oracle", "oracledb"),
        ("redshift://u:p@localhost/db", "redshift", "redshift_connector"),
    ],
)
def test_plain_reviewed_uri_constructs_engine_with_declared_driver(
    tmp_path: Path, connection: str, expected_dialect: str, expected_driver: str,
) -> None:
    from sqlalchemy import create_engine

    _, effective = connectors._prepare_connection(tmp_path, connection)
    engine = create_engine(effective)
    try:
        assert engine.dialect.name == expected_dialect
        assert engine.dialect.driver == expected_driver
    finally:
        engine.dispose()


# --------------------------------------------------------------------
# Redaction regressions (found by fuzzing, not review)
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    "uri,secret",
    [
        ("postgresql://alice:hunter2@host/db", "hunter2"),
        ("mysql+pymysql://u:pa/ss@host/db", "pa/ss"),  # slash in password
        ("postgresql://u:p@ss:w/rd@host:5432/db", "p@ss:w/rd"),  # @ and : and /
        ("postgresql://a b:hunter2@host/db", "hunter2"),  # space in user
        ("mssql+pyodbc://u:::@host/db", "::"),
    ],
)
def test_passwords_never_survive_redaction(uri, secret) -> None:
    """The first regex-based version leaked every one of these."""
    assert secret not in redact_connection(uri)
    assert "***" in redact_connection(uri)


def test_userless_and_odd_uris_are_left_readable() -> None:
    assert redact_connection("postgresql://user@host/db") == "postgresql://user@host/db"
    assert redact_connection("/local/file.duckdb") == "/local/file.duckdb"
    assert redact_connection(None) == ""  # type: ignore[arg-type]


# --------------------------------------------------------------------
# Schemeless credential-bearing strings (audit pass 2 finding): Oracle's
# EZConnect syntax (``user/password@host:port/service``) and its JDBC
# form (``jdbc:oracle:thin:user/password@host:port:sid``) contain
# neither "://" (so the scheme branch never runs) nor a ``pwd=``/
# ``password=`` key (so ``_KV_PASSWORD_PATTERN`` never runs either) --
# the password used to sail straight through unredacted.
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    "uri,secret",
    [
        ("scott/tiger@myhost:1521/orcl", "tiger"),
        ("jdbc:oracle:thin:scott/tiger@myhost:1521:orcl", "tiger"),
        ("scott/hunter2@//myhost:1521/orcl", "hunter2"),
    ],
)
def test_schemeless_ezconnect_style_passwords_are_redacted(uri, secret) -> None:
    """The gap this closes: no '://' and no pwd=/password= key, so
    neither existing redaction branch used to fire at all."""
    assert "://" not in uri
    redacted = redact_connection(uri)
    assert secret not in redacted
    assert "***" in redacted
    assert "myhost" in redacted  # host still useful for diagnosis


def test_schemeless_bare_user_at_host_is_left_readable() -> None:
    """Negative control: an '@' with no ':' or '/' before it (nothing
    that looks like a user<sep>password pairing) must not be mangled
    -- e.g. a bare hostname-only string someone pasted."""
    assert redact_connection("admin@myhost") == "admin@myhost"


# --------------------------------------------------------------------
# ODBC / key=value style connection strings (never contain "://", so
# the URI-only redaction logic used to let them straight through
# unredacted — a real path: pyodbc's SQL Server driver, among others,
# echoes its own constructed connection string, credentials included,
# into a connection-failure exception's message, and that message is
# exactly what flows through redact_connection at the
# _run_sqlalchemy/_run_duckdb error sites.
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    "conn,secret",
    [
        (
            "DRIVER={ODBC Driver 17};SERVER=host;UID=admin;PWD=secret123;DATABASE=db",
            "secret123",
        ),
        ("Server=host;Database=db;User Id=admin;Password=hunter2;", "hunter2"),
        ("server=host;uid=admin;pwd=lowercase_pw;", "lowercase_pw"),
    ],
)
def test_odbc_style_passwords_are_redacted(conn, secret) -> None:
    """The gap this closes: no '://' anywhere in these strings, so
    the URI-parsing branch never even runs."""
    assert "://" not in conn
    redacted = redact_connection(conn)
    assert secret not in redacted
    assert "***" in redacted
    # Non-secret structural info stays visible, same posture as the
    # URI case — this is a display string, not a full scrub.
    assert "host" in redacted
    assert "admin" in redacted or "UID" in redacted or "User Id" in redacted


def test_driver_error_message_echoing_a_connection_string_is_redacted() -> None:
    """Simulates the actual reachable path: an exception's ``str(e)``
    (what ``_run_sqlalchemy``/``_run_duckdb`` pass through
    ``redact_connection``) that embeds a raw key=value connection
    string rather than (or in addition to) a URI."""
    msg = (
        "('08001', '[08001] [Microsoft][ODBC Driver 17 for SQL Server]"
        "Login timeout expired (0) (SQLDriverConnect); [08001] "
        "[Microsoft][ODBC Driver 17 for SQL Server]... Connection "
        "string: SERVER=db.uni.edu;UID=alice;PWD=hunter2;DATABASE=research"
        "')"
    )
    redacted = redact_connection(msg)
    assert "hunter2" not in redacted
    assert "***" in redacted
    assert "db.uni.edu" in redacted  # host still useful for diagnosis


def test_odbc_redaction_does_not_false_positive_on_similar_words() -> None:
    """Word-boundary matching: a longer identifier that merely
    CONTAINS "password" as a substring (not a standalone key) must
    not be mangled — this function redacts credentials, not any text
    that happens to include the word."""
    conn = "some_password_policy_note=irrelevant;SERVER=host"
    redacted = redact_connection(conn)
    assert redacted == conn  # unchanged: no standalone pwd/password key


def test_kv_redaction_applies_even_when_a_uri_is_also_present() -> None:
    """A message containing BOTH a URI and a separately-echoed
    key=value connection string must get both forms scrubbed."""
    msg = (
        "could not connect via postgresql://alice:hunter2@host/db "
        "-- driver fallback attempted with PWD=hunter2;UID=alice;"
    )
    redacted = redact_connection(msg)
    assert "hunter2" not in redacted


@pytest.mark.parametrize(
    "uri,secret",
    [
        ("snowflake://u@acct/db?token=oauth-secret&warehouse=w", "oauth-secret"),
        ("postgresql://u@h/db?access_token=bearer-secret", "bearer-secret"),
        ("mssql://h/db?api_key=key-secret;Encrypt=yes", "key-secret"),
        ("https://h/path?client_secret=client-secret&x=1", "client-secret"),
        ("bigquery://project?credentials_base64=encoded-secret", "encoded-secret"),
        ("bigquery://project?credentials_info=json-secret", "json-secret"),
    ],
)
def test_query_string_credentials_are_redacted(uri: str, secret: str) -> None:
    shown = redact_connection(uri)
    assert secret not in shown
    assert "***" in shown


@pytest.mark.parametrize(
    ("uri", "secret", "secret_parts"),
    [
        (
            "snowflake://u@acct/db?token=alpha-secret%26omega-secret&warehouse=w",
            "alpha-secret&omega-secret",
            ("alpha-secret", "omega-secret"),
        ),
        (
            "https://h/path?client_secret=alpha-secret%3Bomega-secret&x=1",
            "alpha-secret;omega-secret",
            ("alpha-secret", "omega-secret"),
        ),
        (
            "https://h/path?api_key=alpha-secret%20omega-secret&x=1",
            "alpha-secret omega-secret",
            ("alpha-secret", "omega-secret"),
        ),
    ],
)
def test_percent_encoded_query_secret_delimiters_are_fully_redacted(
    uri: str,
    secret: str,
    secret_parts: tuple[str, ...],
) -> None:
    shown = redact_connection(uri)
    assert secret not in shown
    assert all(part not in shown for part in secret_parts)
    assert "***" in shown


def test_braced_odbc_password_with_semicolon_is_fully_redacted() -> None:
    shown = redact_connection(
        "DRIVER={ODBC Driver 18 for SQL Server};SERVER=db.example;"
        "UID=alice;PWD={part-one;part-two};Encrypt=yes"
    )
    assert "part-one" not in shown
    assert "part-two" not in shown
    assert "PWD=***" in shown


def test_percent_encoded_odbc_credentials_are_redacted() -> None:
    uri = (
        "mssql+pyodbc:///?odbc_connect="
        "DRIVER%3DODBC%2BDriver%2B18%3BSERVER%3Ddb.example%3B"
        "UID%3Dalice%3BPWD%3Dhunter2%3BEncrypt%3Dyes"
    )
    shown = redact_connection(uri)
    assert "hunter2" not in shown
    assert "PWD=***" in shown


@pytest.mark.parametrize(
    "connection,backend",
    [
        ("postgresql://u:p@db.example/research", "postgresql"),
        ("postgresql://u:p@db.example/research?sslmode=prefer", "postgresql"),
        ("postgresql://u:p@db.example/research?sslmode=verify-ca", "postgresql"),
        ("redshift://u:p@db.example/research?sslmode=verify-ca", "redshift"),
        (
            "redshift://u:p@db.example/research?sslmode=verify-full&ssl=false",
            "redshift",
        ),
        (
            "redshift://u:p@db.example/research?sslmode=verify-full&ssl_insecure=true",
            "redshift",
        ),
        (
            "redshift://u:p@db.example/research?sslmode=verify-full&auto_create=true",
            "redshift",
        ),
        ("mysql+pymysql://u:p@db.example/research?ssl_verify_cert=false", "mysql"),
        ("mariadb+pymysql://u:p@db.example/research?ssl_verify_cert=false", "mariadb"),
        (
            "mysql+pymysql://u:p@db.example/research?ssl_verify_cert=true&"
            "ssl_verify_identity=true&ssl_disabled=true",
            "mysql",
        ),
        (
            "mysql+pymysql://u:p@db.example/research?ssl_verify_cert=true&"
            "ssl_verify_identity=true&local_infile=true",
            "mysql",
        ),
        ("mssql+pyodbc://u:p@db.example/research?Encrypt=no", "mssql"),
        (
            "mssql+pyodbc://u:p@db.example/research?"
            "driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&"
            "TrustServerCertificate=yes",
            "mssql",
        ),
        ("snowflake://u:p@acct/db?insecure_mode=true", "snowflake"),
        ("snowflake://u:p@acct/db", "snowflake"),
        ("oracle+oracledb://u:p@db.example/research", "oracle"),
        (
            "oracle+oracledb://u:p@db.example:1522?service_name=research&"
            "protocol=tcps&ssl_server_dn_match=false",
            "oracle",
        ),
    ],
)
def test_insecure_remote_database_connections_are_refused(
    connection: str,
    backend: str,
) -> None:
    with pytest.raises(ConnectorError):
        validate_connection_security(connection, backend)


@pytest.mark.parametrize(
    "connection,backend",
    [
        ("postgresql://u:p@db.example/research?sslmode=verify-full", "postgresql"),
        (
            "mysql+pymysql://u:p@db.example/research?"
            "ssl_verify_cert=true&ssl_verify_identity=true",
            "mysql",
        ),
        (
            "mariadb+pymysql://u:p@db.example/research?"
            "ssl_verify_cert=true&ssl_verify_identity=true",
            "mariadb",
        ),
        (
            "mssql+pyodbc://u:p@db.example/research?"
            "driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&"
            "TrustServerCertificate=no",
            "mssql",
        ),
        ("snowflake://u:p@acct/db?ocsp_fail_open=false", "snowflake"),
        (
            "oracle+oracledb://u:p@db.example:1522?service_name=research&"
            "protocol=tcps&ssl_server_dn_match=true",
            "oracle",
        ),
        ("postgresql://u:p@localhost/research", "postgresql"),
    ],
)
def test_secure_or_loopback_database_connections_are_accepted(
    connection: str,
    backend: str,
) -> None:
    validate_connection_security(connection, backend)


def test_bigquery_embedded_credentials_are_refused() -> None:
    with pytest.raises(ConnectorError, match="Application Default Credentials"):
        validate_connection_security(
            "bigquery://project?credentials_base64=very-secret-json",
            "bigquery",
        )


def test_bigquery_embedded_credentials_are_refused_on_loopback() -> None:
    with pytest.raises(ConnectorError, match="Application Default Credentials"):
        validate_connection_security(
            "bigquery://localhost?credentials_info=very-secret-json",
            "bigquery",
        )


def test_bigquery_credential_file_in_uri_is_refused() -> None:
    with pytest.raises(ConnectorError, match="Application Default Credentials"):
        validate_connection_security(
            "bigquery://project?credentials_path=/tmp/service-account.json",
            "bigquery",
        )


def test_bigquery_cannot_request_an_unprovided_client() -> None:
    with pytest.raises(ConnectorError, match="user_supplied_client"):
        validate_connection_security(
            "bigquery://project?user_supplied_client=true",
            "bigquery",
        )


@pytest.mark.parametrize(
    "connection,backend,option",
    [
        (
            "postgresql+psycopg://u:p@localhost/research?"
            "host=outside.example&sslmode=verify-full",
            "postgresql",
            "host",
        ),
        (
            "postgresql+psycopg://u:p@db.example/research?"
            "hostaddr=203.0.113.10&sslmode=verify-full",
            "postgresql",
            "hostaddr",
        ),
        (
            "mysql+pymysql://u:p@db.example/research?"
            "read_default_file=/tmp/my.cnf&ssl_verify_cert=true&"
            "ssl_verify_identity=true",
            "mysql",
            "read_default_file",
        ),
        (
            "redshift+redshift_connector://u:p@db.example/research?"
            "endpoint_url=https://outside.example&sslmode=verify-full",
            "redshift",
            "endpoint_url",
        ),
        (
            "oracle+oracledb://u:p@db.example:1522?service_name=research&"
            "dsn=outside.example/service&protocol=tcps&ssl_server_dn_match=true",
            "oracle",
            "dsn",
        ),
        (
            "snowflake://u:p@account/db?ocsp_fail_open=false&"
            "private_key_file=/tmp/key.p8",
            "snowflake",
            "private_key_file",
        ),
    ],
)
def test_hidden_or_overriding_database_targets_are_refused(
    connection: str,
    backend: str,
    option: str,
) -> None:
    with pytest.raises(ConnectorError, match=option):
        validate_connection_security(connection, backend)


def test_duplicate_security_options_are_refused() -> None:
    with pytest.raises(ConnectorError, match="must not be repeated"):
        validate_connection_security(
            "postgresql+psycopg://u:p@db.example/research?"
            "sslmode=verify-full&sslmode=disable",
            "postgresql",
        )


def test_odbc_security_parser_honours_braces_and_refuses_duplicates() -> None:
    secure = (
        "mssql+pyodbc:///?odbc_connect="
        "DRIVER%3D%7BODBC%20Driver%2018%20for%20SQL%20Server%7D%3B"
        "SERVER%3Dtcp%3Adb.example%2C1433%3B"
        "PWD%3D%7Bpart-one%3Bpart-two%7D%3BEncrypt%3Dmandatory%3B"
        "TrustServerCertificate%3Dno"
    )
    validate_connection_security(secure, "mssql")
    assert connectors._database_policy_endpoint(secure, "mssql") == "https://db.example"

    ambiguous = secure + "%3BEncrypt%3Dno"
    with pytest.raises(ConnectorError, match="repeats.*encrypt"):
        validate_connection_security(ambiguous, "mssql")

    misleading_driver = secure.replace(
        "DRIVER%3D%7BODBC%20Driver%2018%20for%20SQL%20Server%7D",
        "DRIVER%3D%7BOther%20Driver%7D",
    ) + "&driver=ODBC+Driver+18+for+SQL+Server"
    with pytest.raises(ConnectorError, match="ODBC Driver 18"):
        validate_connection_security(misleading_driver, "mssql")


def test_oracle_tns_alias_cannot_claim_verified_direct_transport() -> None:
    with pytest.raises(ConnectorError, match="explicit host, port"):
        validate_connection_security(
            "oracle+oracledb://u:p@wallet_alias?protocol=tcps&"
            "ssl_server_dn_match=true",
            "oracle",
        )


@pytest.mark.parametrize(
    "connection,backend,option",
    [
        (
            "mysql+pymysql://u:p@db.example/research?"
            "ssl_verify_cert=true&ssl_verify_identity=true&"
            "init_command=DELETE%20FROM%20patients",
            "mysql",
            "init_command",
        ),
        (
            "oracle+oracledb://u:p@db.example:1522?service_name=research&"
            "protocol=tcps&ssl_server_dn_match=true&newpassword=changed",
            "oracle",
            "newpassword",
        ),
        (
            "bigquery://project?destination=other.dataset.table",
            "bigquery",
            "destination",
        ),
        (
            "bigquery://project?write_disposition=WRITE_TRUNCATE",
            "bigquery",
            "write_disposition",
        ),
        (
            "bigquery://project?dry_run=true",
            "bigquery",
            "dry_run",
        ),
    ],
)
def test_connection_setup_cannot_bypass_read_only_gate(
    connection: str,
    backend: str,
    option: str,
) -> None:
    with pytest.raises(ConnectorError, match=option):
        validate_connection_security(connection, backend)


def test_ambient_environment_cannot_disable_remote_tls_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIFT_ALLOW_INSECURE_DATABASE", "1")
    with pytest.raises(ConnectorError, match="verify-full"):
        validate_connection_security(
            "postgresql+psycopg://u:p@db.example/research?sslmode=disable",
            "postgresql",
        )


def test_sql_server_requires_odbc_driver_18_even_on_loopback() -> None:
    with pytest.raises(ConnectorError, match="ODBC Driver 18"):
        validate_connection_security(
            "mssql+pyodbc://u:p@localhost/research?Encrypt=yes&TrustServerCertificate=no",
            "mssql",
        )


@pytest.mark.parametrize(
    "connection",
    [
        "databricks://token:secret@host?http_path=/sql/x&_tls_no_verify=true",
        "databricks://token:secret@host?http_path=/sql/x&_enable_ssl=false",
        "databricks://token:secret@host?http_path=/sql/x&_tls_verify_hostname=false",
        "databricks://token:secret@host?http_path=/sql/x&http_scheme=http",
    ],
)
def test_databricks_tls_bypasses_are_refused(connection: str) -> None:
    with pytest.raises(ConnectorError, match="certificate"):
        validate_connection_security(connection, "databricks")


# --------------------------------------------------------------------
# Total input handling
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"connection": None, "sql": "SELECT 1", "dataset_name": "x"},
        {"connection": "", "sql": "SELECT 1", "dataset_name": "x"},
        {"connection": "junk", "sql": "SELECT 1", "dataset_name": "x"},
        {"connection": "sqlite:///nope.db", "sql": None, "dataset_name": "x"},
    ],
)
def test_bad_inputs_raise_connector_error_only(tmp_path: Path, kwargs) -> None:
    """Every failure must arrive as an actionable ConnectorError, not
    a raw driver traceback surfacing in the UI."""
    with pytest.raises(ConnectorError):
        run_extract(tmp_path, **kwargs)  # type: ignore[arg-type]


def test_missing_table_is_a_connector_error(tmp_path: Path) -> None:
    """DuckDB raises its own exception types; they must be wrapped."""
    pq = tmp_path / "d.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(pq)
    with pytest.raises(ConnectorError) as exc:
        run_extract(
            tmp_path,
            connection=str(pq),
            sql="SELECT * FROM no_such_table",
            dataset_name="x",
        )
    assert "read-only statement" in str(exc.value)


def test_missing_sqlite_database_is_not_created_by_read_only_extract(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "must-not-be-created.sqlite"
    with pytest.raises(ConnectorError, match="will not create"):
        run_extract(
            tmp_path,
            connection=f"sqlite:///{missing}",
            sql="SELECT 1",
            dataset_name="x",
        )
    assert not missing.exists()


def test_plain_relative_sqlite_path_is_supported(tmp_path: Path, db: Path) -> None:
    """The path syntax advertised by describe_backend must actually work."""
    result = run_extract(
        tmp_path,
        connection=db.name,
        sql="SELECT * FROM patients",
        dataset_name="plain_path",
        row_limit=10,
    )
    assert result.rows == 10
    assert result.backend == "sqlite"


def test_sqlite_paths_are_rendered_as_portable_file_urls(
    tmp_path: Path, db: Path,
) -> None:
    from sqlalchemy.engine import make_url

    uri = connectors._sqlite_connection_uri(str(db), tmp_path)
    assert make_url(uri).database == db.resolve().as_posix()
    assert "\\" not in uri


def test_relative_sqlite_uri_is_resolved_against_session(
    tmp_path: Path,
    db: Path,
) -> None:
    result = run_extract(
        tmp_path,
        connection=f"sqlite:///{db.name}",
        sql="SELECT * FROM patients",
        dataset_name="relative_uri",
        row_limit=10,
    )
    assert result.rows == 10


def test_odd_row_limits_do_not_crash(tmp_path: Path, db: Path) -> None:
    for limit in (0, -1, 10**9, "x", None):
        result = run_extract(
            tmp_path,
            connection=f"sqlite:///{db}",
            sql="SELECT * FROM patients",
            dataset_name="lim",
            row_limit=limit,
        )  # type: ignore[arg-type]
        assert result.rows > 0


def test_database_operations_support_typed_preflight_cancellation(
    tmp_path: Path,
    db: Path,
) -> None:
    from sift.integration_core import CancellationToken

    token = CancellationToken()
    token.cancel()
    operations = (
        lambda: check_connection(tmp_path, connection=str(db), cancellation=token),
        lambda: inspect_database(tmp_path, connection=str(db), cancellation=token),
        lambda: run_extract(
            tmp_path,
            connection=str(db),
            sql="SELECT * FROM patients",
            dataset_name="cancelled",
            cancellation=token,
        ),
    )
    for operation in operations:
        with pytest.raises(ConnectorError) as raised:
            operation()
        assert raised.value.code == "cancelled"


def test_database_midstream_cancellation_removes_partial_extract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Path,
) -> None:
    from sift.integration_core import CancellationToken

    token = CancellationToken()

    def batches(*_args, **_kwargs):
        yield pd.DataFrame({"x": [1, 2]})
        token.cancel()
        yield pd.DataFrame({"x": [3, 4]})

    monkeypatch.setattr(connectors, "_run_sqlalchemy", batches)
    with pytest.raises(ConnectorError) as raised:
        run_extract(
            tmp_path,
            connection=str(db),
            sql="SELECT * FROM patients",
            dataset_name="cancelled",
            cancellation=token,
        )
    assert raised.value.code == "cancelled"
    assert not (tmp_path / "cancelled.parquet").exists()
    assert not list(tmp_path.glob(".sift-extract-*.parquet"))


@pytest.mark.parametrize(
    ("backend", "statement"),
    [
        ("duckdb", "SELECT * FROM read_parquet('/etc/passwd')"),
        ("duckdb", "SELECT * FROM sqlite_scan('/tmp/other.db', 'secrets')"),
        ("postgresql", "SELECT pg_read_file('/etc/passwd')"),
        ("mysql", "SELECT load_file('/etc/passwd')"),
        ("mariadb", "SELECT master_pos_wait('binlog', 1)"),
        ("mssql", "SELECT * FROM openquery(remote_server, 'select 1')"),
        ("oracle", "SELECT utl_http.request('https://example.com') FROM dual"),
        ("snowflake", "SELECT get_presigned_url('@private', 'x')"),
        ("bigquery", "SELECT * FROM external_query('connection', 'select 1')"),
        ("databricks", "SELECT reflect('java.lang.System', 'getenv')"),
    ],
)
def test_backend_specific_dangerous_functions_are_rejected(
    backend: str,
    statement: str,
) -> None:
    assert normalize_sql(statement, backend=backend) is None


def test_duckdb_file_queries_are_confined_to_registered_source_view() -> None:
    assert normalize_sql(
        "WITH selected AS (SELECT * FROM source) SELECT * FROM selected",
        backend="duckdb-file",
    ) is not None
    assert normalize_sql(
        "SELECT * FROM '/tmp/another.parquet'",
        backend="duckdb-file",
    ) is None
    assert normalize_sql(
        "SELECT * FROM unrelated_table",
        backend="duckdb-file",
    ) is None


def test_safe_preview_never_runs_sqlite_query(tmp_path: Path, db: Path) -> None:
    preview = preview_query(
        tmp_path,
        connection=str(db),
        sql="SELECT * FROM patients",
    )
    assert preview.backend == "sqlite"
    assert preview.executes_query is False
    assert preview.dry_run_supported is False
    assert preview.estimated_bytes is None
    assert len(preview.query_sha256) == 64


def test_bigquery_preview_surfaces_cost_warning_without_sql(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        connectors,
        "_bigquery_dry_run",
        lambda _connection, _sql, *_deadline: (2 * 1024**3, None),
    )
    preview = preview_query(
        tmp_path,
        connection="bigquery://research-project",
        sql="SELECT 1",
    )
    assert preview.dry_run_supported is True
    assert preview.estimated_bytes == 2 * 1024**3
    assert preview.metered_warehouse is True
    assert preview.warnings and "2,147,483,648" in preview.warnings[0]
    assert "SELECT" not in repr(preview)


@pytest.mark.parametrize(
    ("query", "message"),
    [
        ("location=US&location=EU", "location must be specified once"),
        (
            "billing_project_id=one&billing_project_id=two",
            "billing_project_id must be specified once",
        ),
    ],
)
def test_bigquery_dry_run_rejects_duplicate_security_options(
    query: str, message: str,
) -> None:
    with pytest.raises(ConnectorError, match=message):
        connectors._bigquery_dry_run(
            f"bigquery://research-project?{query}", "SELECT 1",
        )


def test_explain_analyze_is_never_accepted_as_preview(
    tmp_path: Path,
    db: Path,
) -> None:
    with pytest.raises(ConnectorError):
        preview_query(
            tmp_path,
            connection=str(db),
            sql="EXPLAIN ANALYZE SELECT * FROM patients",
        )


def test_extraction_reports_value_free_progress(tmp_path: Path, db: Path) -> None:
    events: list[connectors.ExtractionProgress] = []
    result = run_extract(
        tmp_path,
        connection=str(db),
        sql="SELECT * FROM patients",
        dataset_name="progress",
        progress=events.append,
    )
    assert [event.stage for event in events] == [
        "starting", "querying", "materializing", "finalizing", "complete",
    ]
    assert events[-1].rows_materialized == result.rows
    assert all(set(event.__dict__) == {
        "stage", "rows_materialized", "bytes_buffered",
    } for event in events)


def test_materialization_preserves_research_types(tmp_path: Path) -> None:
    frame = pd.DataFrame({
        "decimal": [Decimal("1234567890.123456789")],
        "date": [date(2026, 8, 21)],
        "zoned": [pd.Timestamp("2026-08-21T12:30:00-07:00")],
        "boolean": pd.Series([True], dtype="boolean"),
        "category": pd.Series(["case"], dtype="category"),
        "nested": [{"groups": ["a", "b"], "count": 2}],
        "binary": [b"\x00\xff"],
    })
    result = connectors._materialize_query_batches(
        [frame], tmp_path, "types", 10,
    )
    restored = pd.read_parquet(result.path)
    assert restored.loc[0, "decimal"] == Decimal("1234567890.123456789")
    assert restored.loc[0, "date"] == date(2026, 8, 21)
    assert str(restored.loc[0, "zoned"].tz) != "None"
    assert bool(restored.loc[0, "boolean"]) is True
    assert restored.loc[0, "binary"] == b"\x00\xff"


def test_materialization_closes_every_parquet_part_reader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import pyarrow.parquet as pq

    real_parquet_file = pq.ParquetFile
    opened: list[object] = []

    class TrackingParquetFile:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._inner = real_parquet_file(*args, **kwargs)
            self.closed = False
            opened.append(self)

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

        def close(self) -> None:
            self.closed = True
            self._inner.close()

    monkeypatch.setattr(pq, "ParquetFile", TrackingParquetFile)
    result = connectors._materialize_query_batches(
        [pd.DataFrame({"id": [1, 2]})],
        tmp_path,
        "closed_readers",
        10,
    )
    assert result.path.is_file()
    assert opened and all(reader.closed for reader in opened)  # type: ignore[attr-defined]


def test_materialization_rejects_unsupported_objects_without_values(
    tmp_path: Path,
) -> None:
    class Unsupported:
        def __repr__(self) -> str:
            return "PRIVATE_VALUE"

    with pytest.raises(ConnectorError) as raised:
        connectors._materialize_query_batches(
            [pd.DataFrame({"unsupported": [Unsupported()]})],
            tmp_path,
            "unsupported",
            10,
        )
    assert "represented safely in Parquet" in str(raised.value)
    assert "PRIVATE_VALUE" not in str(raised.value)


def test_expired_authentication_is_typed_and_actionable() -> None:
    error = RuntimeError("OAuth token has expired for alice")
    classified = connectors._driver_connector_error(
        "connection check failed",
        error,
        "postgresql://alice:secret@db/research",
    )
    assert classified.code == "authentication_expired"
    assert classified.retryable is True
    assert "alice" not in str(classified)
    assert "secret" not in str(classified)


def test_sqlite_read_only_uri_is_certified(tmp_path: Path, db: Path) -> None:
    uri = f"sqlite:///file:{db}?mode=ro&uri=true"
    result = run_extract(
        tmp_path,
        connection=uri,
        sql="SELECT COUNT(*) AS n FROM patients",
        dataset_name="readonly",
    )
    assert pd.read_parquet(result.dataset_path).loc[0, "n"] == 1000


def test_malformed_sqlite_fails_without_creating_extract(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.sqlite"
    malformed.write_bytes(b"not a sqlite database")
    with pytest.raises(ConnectorError, match="connection check failed"):
        check_connection(tmp_path, connection=str(malformed))
    assert not list(tmp_path.glob("*.parquet"))


def test_locked_sqlite_fails_cleanly(tmp_path: Path, db: Path) -> None:
    lock = sqlite3.connect(db)
    lock.execute("BEGIN EXCLUSIVE")
    try:
        with pytest.raises(ConnectorError, match="query failed"):
            run_extract(
                tmp_path,
                connection=f"sqlite:///{db}?timeout=0.05",
                sql="SELECT * FROM patients",
                dataset_name="locked",
            )
    finally:
        lock.rollback()
        lock.close()
    assert not (tmp_path / "locked.parquet").exists()


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".json", ".jsonl"])
def test_duckdb_source_view_certifies_text_formats(
    suffix: str,
    tmp_path: Path,
) -> None:
    source = tmp_path / f"source{suffix}"
    frame = pd.DataFrame({"cohort": ["a", "b", "a"], "amount": [1, 2, 3]})
    if suffix == ".csv":
        frame.to_csv(source, index=False)
    elif suffix == ".tsv":
        frame.to_csv(source, sep="\t", index=False)
    else:
        frame.to_json(source, orient="records", lines=suffix == ".jsonl")
    result = run_extract(
        tmp_path,
        connection=source.name,
        sql="SELECT cohort, SUM(amount) AS total FROM source GROUP BY cohort",
        dataset_name=f"text_{suffix[1:]}",
    )
    restored = pd.read_parquet(result.dataset_path).set_index("cohort")
    assert restored.loc["a", "total"] == 4


def test_duckdb_large_operation_stays_out_of_core(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = tmp_path / "large.duckdb"
    connection = duckdb.connect(str(path))
    connection.execute(
        "CREATE TABLE measurements AS SELECT i, i % 100 AS group_id "
        "FROM range(1000000) AS values(i)"
    )
    connection.close()
    result = run_extract(
        tmp_path,
        connection=str(path),
        sql=(
            "SELECT group_id, COUNT(*) AS n, SUM(i) AS total "
            "FROM measurements GROUP BY group_id"
        ),
        dataset_name="large_rollup",
    )
    assert result.rows == 100
    assert int(pd.read_parquet(result.dataset_path)["n"].sum()) == 1_000_000


@pytest.mark.parametrize(
    ("dialect", "expected"),
    [
        ("postgresql", "SET TRANSACTION READ ONLY"),
        ("redshift", "SET TRANSACTION READ ONLY"),
        ("oracle", "SET TRANSACTION READ ONLY"),
        ("mysql", "SET TRANSACTION READ ONLY"),
        ("mariadb", "SET TRANSACTION READ ONLY"),
    ],
)
def test_server_read_only_transactions_are_applied_where_supported(
    dialect: str,
    expected: str,
) -> None:
    class Connection:
        def __init__(self) -> None:
            self.statements: list[str] = []

        def exec_driver_sql(self, statement: str) -> None:
            self.statements.append(statement)

    connection = Connection()
    connectors._configure_connection_read_only(connection, dialect)
    assert connection.statements == [expected]
    assert connectors._read_only_enforcement(dialect) == (
        "database_session_and_query_gate"
    )


@pytest.mark.parametrize(
    "dialect",
    ["mssql", "snowflake", "bigquery", "databricks"],
)
def test_query_gate_only_backends_are_explicit(dialect: str) -> None:
    class Connection:
        def exec_driver_sql(self, _statement: str) -> None:
            raise AssertionError("no portable session read-only control")

    connectors._configure_connection_read_only(Connection(), dialect)
    assert connectors._read_only_enforcement(dialect) == (
        "query_gate_only_use_select_only_principal"
    )


@pytest.mark.parametrize(
    ("dialect", "fragment"),
    [
        ("postgresql", "statement_timeout"),
        ("redshift", "statement_timeout"),
        ("mysql", "MAX_EXECUTION_TIME"),
        ("mariadb", "max_statement_time"),
        ("snowflake", "STATEMENT_TIMEOUT_IN_SECONDS"),
    ],
)
def test_backend_server_query_deadlines(dialect: str, fragment: str) -> None:
    class Raw:
        pass

    class Wrapper:
        driver_connection = Raw()

    class Connection:
        connection = Wrapper()

        def __init__(self) -> None:
            self.statements: list[str] = []

        def exec_driver_sql(self, statement: str) -> None:
            self.statements.append(statement)

    connection = Connection()
    cleanup = connectors._configure_query_timeout(connection, dialect, 7)
    assert fragment in connection.statements[0]
    cleanup()


def test_sqlalchemy_raw_query_preserves_colons_inside_literals() -> None:
    batches = list(connectors._run_sqlalchemy(
        "sqlite:///:memory:",
        "SELECT '{\"nested\":7}' AS json_value, "
        "'2026-08-21T12:34:56Z' AS observed_at",
        "sqlite",
        10,
    ))
    assert len(batches) == 1
    assert batches[0].to_dict(orient="records") == [{
        "json_value": '{"nested":7}',
        "observed_at": "2026-08-21T12:34:56Z",
    }]
