"""Credential-free release check for dynamically loaded integrations.

PyInstaller cannot infer integrations selected from a provider, database,
cloud-source, or data-format registry.  A module being present in the build
environment is therefore weaker evidence than importing it from the final
frozen executable.  This module performs that stronger check without opening
a socket, reading a credential store, or constructing an SDK client.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Any


# SQLAlchemy resolves third-party dialects through entry-point metadata.  The
# import can succeed while the dialect entry point is absent from a frozen
# application, so both boundaries are checked.
_DATABASE_DIALECTS: dict[str, str | None] = {
    "sqlite": "sqlite",
    "duckdb": None,
    "postgresql": "postgresql.psycopg",
    "mysql": "mysql.pymysql",
    "mariadb": "mariadb.pymysql",
    "mssql": "mssql.pyodbc",
    "oracle": "oracle.oracledb",
    "snowflake": "snowflake",
    "bigquery": "bigquery",
    "redshift": "redshift.redshift_connector",
    "databricks": "databricks",
}

_PROVIDER_MODULES: tuple[tuple[str, str], ...] = (
    ("anthropic", "sift.provider.anthropic"),
    ("openai", "sift.provider.openai"),
    ("openai_compatible", "sift.provider.openai_compatible"),
    ("gemini", "sift.provider.gemini"),
    ("azure_openai", "sift.provider.azure_openai"),
    ("vertex_gemini", "sift.provider.vertex_gemini"),
    ("bedrock_anthropic", "sift.provider.bedrock_anthropic"),
    ("vertex_anthropic", "sift.provider.vertex_anthropic"),
)

_PROVIDER_SDKS: tuple[tuple[str, str], ...] = (
    ("anthropic", "claude_agent_sdk"),
    ("openai", "openai"),
    ("gemini", "google.genai"),
    ("azure_identity", "azure.identity"),
    ("aws_bedrock", "boto3"),
    ("vertex_identity", "google.auth"),
    ("vertex_anthropic", "anthropic"),
)

_CORE_DATA_MODULES: tuple[tuple[str, str], ...] = (
    ("csv_tables", "pandas"),
    ("arrow_tables", "pyarrow"),
    ("stata_spss_sas", "pyreadstat"),
    ("r_data", "pyreadr"),
    ("excel_xlsx", "openpyxl"),
    ("excel_xls", "xlrd"),
    ("opendocument", "odf"),
)


def _import_check(
    category: str,
    integration_id: str,
    module: str,
    *,
    required: bool = True,
) -> dict[str, Any]:
    """Import one module and return only content-free diagnostics."""
    try:
        importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - release report must contain failure
        return {
            "category": category,
            "id": integration_id,
            "module": module,
            "required": required,
            "ok": False,
            # Exception messages can contain host paths, endpoints, or SDK
            # configuration.  The type is enough to diagnose bundle omission.
            "detail": type(exc).__name__,
        }
    return {
        "category": category,
        "id": integration_id,
        "module": module,
        "required": required,
        "ok": True,
        "detail": "import passed",
    }


def _package_resource_check(
    integration_id: str,
    module: str,
    relative: str,
) -> dict[str, Any]:
    """Verify a required package-relative runtime data file exists."""
    result = {
        "category": "runtime_data",
        "id": integration_id,
        "module": f"{module}/{relative}",
        "required": True,
        "ok": False,
        "detail": "resource missing",
    }
    try:
        package = importlib.import_module(module)
        frozen_root = getattr(sys, "_MEIPASS", None)
        if frozen_root:
            # Frozen Python modules live in the PYZ archive, so their
            # ``__file__`` value is logical and need not exist as a regular
            # file. Package data is materialized under ``sys._MEIPASS``.
            root = (
                Path(frozen_root) / Path(*module.split("."))
            ).resolve(strict=True)
        else:
            package_file = getattr(package, "__file__", None)
            if not package_file:
                return result
            root = Path(package_file).resolve(strict=True).parent
        resource = (root / relative).resolve(strict=True)
        resource.relative_to(root)
        if resource.is_file():
            result["ok"] = True
            result["detail"] = "runtime data present"
    except Exception as exc:  # noqa: BLE001 - content-free release evidence
        result["detail"] = type(exc).__name__
    return result


def _required_checks_pass(checks: list[dict[str, Any]]) -> bool:
    """True when every required runtime surface passed its own check."""
    return all(
        bool(check["ok"]) or not bool(check["required"])
        for check in checks
    )


def integration_runtime_report() -> dict[str, Any]:
    """Verify every advertised dynamic integration without external access."""
    from sift.format_selection import FORMAT_CAPABILITIES
    from sift.integrations import CLOUD_SOURCE_ADAPTERS, DATABASE_ADAPTERS

    checks: list[dict[str, Any]] = []

    for database_adapter in DATABASE_ADAPTERS:
        imported = _import_check(
            "database_driver", database_adapter.id, database_adapter.driver_module,
        )
        checks.append(imported)
        dialect_name = _DATABASE_DIALECTS[database_adapter.id]
        if dialect_name is not None:
            dialect_check = {
                "category": "database_dialect",
                "id": database_adapter.id,
                "module": dialect_name,
                "required": True,
                "ok": False,
                "detail": "driver import failed",
            }
            if imported["ok"]:
                try:
                    from sqlalchemy.dialects import registry

                    registry.load(dialect_name)
                except Exception as exc:  # noqa: BLE001 - release evidence
                    dialect_check["detail"] = type(exc).__name__
                else:
                    dialect_check["ok"] = True
                    dialect_check["detail"] = "dialect load passed"
            checks.append(dialect_check)

    for cloud_adapter in CLOUD_SOURCE_ADAPTERS:
        checks.append(_import_check(
            "cloud_source", cloud_adapter.id, cloud_adapter.driver_module,
        ))

    for provider_id, module in _PROVIDER_MODULES:
        checks.append(_import_check("provider_implementation", provider_id, module))
    for provider_id, module in _PROVIDER_SDKS:
        checks.append(_import_check("provider_sdk", provider_id, module))
    for format_id, module in _CORE_DATA_MODULES:
        checks.append(_import_check("core_data_format", format_id, module))

    # Check each declared parser dependency under its advertised capability.
    # Duplicate module imports are cheap after the first and keep failures
    # attributable to every affected format in the machine-readable report.
    for capability in FORMAT_CAPABILITIES:
        for module in capability.dependencies:
            # pysam publishes no Windows wheels and the corresponding BCF
            # capability is therefore reported as uninstalled on Windows.
            # Preserve it in this machine-readable inventory, but do not turn
            # an explicitly platform-unavailable optional parser into a false
            # failure for the otherwise complete Windows runtime. If a future
            # pysam release becomes importable there, the check will still
            # record the successful import automatically.
            platform_optional = (
                sys.platform == "win32"
                and capability.id == "bcf"
                and module == "pysam"
            )
            check = _import_check(
                "selected_data_format",
                capability.id,
                module,
                required=not platform_optional,
            )
            if platform_optional and not check["ok"]:
                check["detail"] = "not packaged on win32"
            checks.append(check)

    for integration_id, module, relative in (
        ("geospatial_vector_gdal", "pyogrio", "gdal_data/gdalvrt.xsd"),
        ("geospatial_vector_proj", "pyogrio", "proj_data/proj.db"),
        ("geospatial_raster_gdal", "rasterio", "gdal_data/gdalvrt.xsd"),
        ("geospatial_raster_proj", "rasterio", "proj_data/proj.db"),
    ):
        checks.append(_package_resource_check(integration_id, module, relative))

    return {
        "schema_version": 1,
        "ok": _required_checks_pass(checks),
        "check_count": len(checks),
        "checks": checks,
    }
