"""Security assurance primitives that stay outside the model boundary.

This module collects controls that need to be useful to a researcher or a
security reviewer but must never become model-callable capabilities:

* local pre-provider disclosure review (warnings, never content forwarding),
* explicit retention previews and secure best-effort cleanup,
* password-encrypted, completion-authenticated session bundles,
* Ed25519 signatures over provenance/export directories,
* CycloneDX SBOM generation and local security qualification checks.

The cryptographic formats use ``cryptography`` primitives directly.  There is
no home-grown cipher: scrypt derives a 256-bit key, AES-GCM authenticates each
bounded chunk, and Ed25519 signs a canonical manifest of file hashes.
"""

from __future__ import annotations

import ast
import base64
import email.parser
import email.policy
import hashlib
import io
import json
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import uuid
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]

from sift.subprocess_safety import run_bounded_capture


_ENC_MAGIC_V1 = b"SIFT-SESSION-ENC-v1\n"
_ENC_MAGIC_V2 = b"SIFT-SESSION-ENC-v2\n"
# New bundles always use v2.  Keep the historical name as the current-format
# alias for callers which only need to identify bundles written by this build.
_ENC_MAGIC = _ENC_MAGIC_V2
_ENC_CHUNK_SIZE = 4 * 1024 * 1024
_MAX_ARCHIVE_BYTES = 64 * 1024 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 100_000
_ENC_RECORD_DATA = b"\x01"
_ENC_RECORD_FINAL = b"\x02"
_ENC_FINAL_NONCE_COUNTER = (2**32) - 1
_ENC_FINAL_MAX_BYTES = 4096
_SIGNATURE_NAME = "provenance.signature.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_tool(name: str) -> str | None:
    """Find a scanner on PATH or beside the active Python interpreter."""
    found = shutil.which(name)
    if found:
        return found
    suffix = ".exe" if os.name == "nt" else ""
    # Do not resolve ``sys.executable``: virtual-environment Python is often a
    # symlink to the system interpreter, while its console scripts live beside
    # the symlink in ``.venv/bin`` / ``.venv\\Scripts``.
    sibling = Path(sys.executable).absolute().parent / f"{name}{suffix}"
    return str(sibling) if sibling.is_file() else None


def _private_file(path: Path) -> None:
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _private_directory(path: Path) -> None:
    """Make a directory current-user-only or fail before storing plaintext."""
    target = Path(path)
    is_junction = getattr(target, "is_junction", lambda: False)
    if target.is_symlink() or is_junction():
        raise ValueError("private restore directory cannot be a link or junction")
    if os.name == "nt":
        from sift.windows_private_state import WindowsAclError, secure_private_directory

        try:
            secure_private_directory(target)
        except WindowsAclError as exc:
            raise ValueError(
                "Windows could not apply a current-user-only restore-directory ACL"
            ) from exc
        return
    try:
        target.chmod(0o700)
        metadata = target.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError("restore directory permissions could not be secured") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("private restore path is not a directory")
    get_euid = getattr(os, "geteuid", None)
    if callable(get_euid) and metadata.st_uid != get_euid():
        raise ValueError("private restore directory is not owned by this account")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ValueError("restore directory permits access by another account")


# ---------------------------------------------------------------------------
# Pre-provider disclosure review
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DisclosureFinding:
    category: str
    source: str
    count: int
    severity: str
    guidance: str


_DISCLOSURE_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    (
        "credential",
        "high",
        re.compile(
            r"(?i)(?:sk-[a-z0-9_-]{16,}|AIza[0-9A-Za-z_-]{20,}|"
            r"AKIA[0-9A-Z]{16}|(?:api[_ -]?key|password|secret|token)"
            r"\s*[:=]\s*[^\s,;]{8,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
        ),
    ),
    (
        "email_address",
        "moderate",
        re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w-])"),
    ),
    (
        "us_social_security_number",
        "high",
        re.compile(r"(?<!\d)(?!000|666|9\d\d)\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}(?!\d)"),
    ),
    (
        "phone_number",
        "moderate",
        re.compile(r"(?<!\d)(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}(?!\d)"),
    ),
    (
        "medical_record_context",
        "high",
        re.compile(
            r"(?i)\b(?:medical record|patient id|patient name|diagnosis|"
            r"health insurance|member id|date of birth|dob)\b"
        ),
    ),
)


def review_pre_provider_disclosure(
    text: str,
    *,
    attachment_names: Iterable[str] = (),
    field_names: Iterable[str] = (),
    organization_sensitive_fields: Iterable[str] = (),
    enabled: bool = True,
) -> dict[str, Any]:
    """Return content-free warnings before a remote provider request.

    Findings contain categories and counts only.  The matched text is never
    copied into the result, logs, or model prompt.  This is deliberately a
    warning layer rather than a claim of de-identification: free text and
    context can remain identifying even when no pattern matches.
    """
    if not enabled:
        return {"enabled": False, "warn": False, "findings": []}

    sources: list[tuple[str, str]] = [("message", str(text or ""))]
    sources.extend(("attachment_name", str(v)) for v in attachment_names)
    sources.extend(("field_name", str(v)) for v in field_names)
    findings: list[DisclosureFinding] = []
    for source, value in sources:
        for category, severity, pattern in _DISCLOSURE_PATTERNS:
            count = sum(1 for _ in pattern.finditer(value))
            if count:
                findings.append(
                    DisclosureFinding(
                        category=category,
                        source=source,
                        count=count,
                        severity=severity,
                        guidance=(
                            "Review or remove this content before using a remote "
                            "provider; use a validated local endpoint when disclosure "
                            "is not permitted."
                        ),
                    )
                )

    sensitive = {
        re.sub(r"[^a-z0-9]+", "_", str(v).strip().casefold()).strip("_")
        for v in organization_sensitive_fields
        if str(v).strip()
    }
    for name in field_names:
        normalized = re.sub(
            r"[^a-z0-9]+", "_", str(name).strip().casefold()
        ).strip("_")
        if normalized and normalized in sensitive:
            findings.append(
                DisclosureFinding(
                    category="organization_sensitive_field",
                    source="field_name",
                    count=1,
                    severity="high",
                    guidance="This field is blocked by organization policy.",
                )
            )

    # Aggregate identical categories/sources to keep the UI concise and avoid
    # leaking how the source text was arranged.
    aggregate: dict[tuple[str, str, str, str], int] = {}
    for finding in findings:
        key = (
            finding.category,
            finding.source,
            finding.severity,
            finding.guidance,
        )
        aggregate[key] = aggregate.get(key, 0) + finding.count
    rows = [
        asdict(DisclosureFinding(k[0], k[1], count, k[2], k[3]))
        for k, count in sorted(aggregate.items())
    ]
    return {
        "enabled": True,
        "warn": bool(rows),
        "findings": rows,
        "limitations": (
            "Pattern review is advisory and cannot prove that content is "
            "de-identified or free of confidential information."
        ),
    }


# ---------------------------------------------------------------------------
# Retention and best-effort secure deletion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetentionCandidate:
    path: str
    kind: str
    age_days: int
    bytes: int


def retention_candidates(
    cwd: Path,
    *,
    run_retention_days: int,
    now: datetime | None = None,
) -> list[RetentionCandidate]:
    """Preview expired execution run directories without deleting them."""
    if isinstance(run_retention_days, bool) or run_retention_days < 1:
        raise ValueError("run_retention_days must be a positive integer")
    root = cwd.resolve()
    runs = root / ".sift" / "runs"
    if not runs.is_dir() or runs.is_symlink():
        return []
    current = now or _utc_now()
    cutoff = current - timedelta(days=run_retention_days)
    result: list[RetentionCandidate] = []
    for child in sorted(runs.iterdir(), key=lambda p: p.name):
        if child.is_symlink() or not child.is_dir():
            continue
        try:
            modified = datetime.fromtimestamp(child.stat().st_mtime, timezone.utc)
        except OSError:
            continue
        if modified > cutoff:
            continue
        total = 0
        for item in child.rglob("*"):
            if item.is_file() and not item.is_symlink():
                try:
                    total += item.stat().st_size
                except OSError:
                    pass
        result.append(
            RetentionCandidate(
                # Public records are serialized to JSON and must remain stable
                # when a session moves between Windows, macOS, and Linux.
                path=child.relative_to(root).as_posix(),
                kind="execution_run",
                age_days=max(0, (current - modified).days),
                bytes=total,
            )
        )
    return result


def _overwrite_then_unlink(path: Path) -> None:
    """Overwrite a regular file once, fsync, then unlink.

    This is best effort.  Copy-on-write filesystems, SSD wear levelling, and
    backups can retain old blocks; encryption at rest and storage-level media
    sanitization remain necessary for strong deletion guarantees.
    """
    if path.is_symlink() or not path.is_file():
        path.unlink(missing_ok=True)
        return
    try:
        size = path.stat().st_size
        with path.open("r+b", buffering=0) as handle:
            block = b"\x00" * min(1024 * 1024, max(1, size))
            remaining = size
            while remaining:
                part = block[: min(len(block), remaining)]
                handle.write(part)
                remaining -= len(part)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        path.unlink(missing_ok=True)


def apply_run_retention(
    cwd: Path,
    *,
    run_retention_days: int,
    now: datetime | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Delete exactly the previewed expired run directories after consent."""
    candidates = retention_candidates(
        cwd, run_retention_days=run_retention_days, now=now
    )
    if not confirmed:
        return {
            "deleted": False,
            "requires_confirmation": bool(candidates),
            "candidates": [asdict(row) for row in candidates],
        }
    root = cwd.resolve()
    removed: list[str] = []
    for candidate in candidates:
        target = (root / candidate.path).resolve()
        expected_parent = (root / ".sift" / "runs").resolve()
        if target.parent != expected_parent or target.is_symlink():
            continue
        for item in sorted(target.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if item.is_symlink() or item.is_file():
                _overwrite_then_unlink(item)
            elif item.is_dir():
                item.rmdir()
        target.rmdir()
        removed.append(candidate.path)
    return {
        "deleted": True,
        "requires_confirmation": False,
        "removed": removed,
        "secure_delete_limitations": (
            "Best-effort overwrite cannot guarantee erasure on SSD, copy-on-write, "
            "snapshot, or backup storage."
        ),
    }


# ---------------------------------------------------------------------------
# Password-encrypted session bundles
# ---------------------------------------------------------------------------


def _crypto() -> tuple[Any, Any]:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    except ImportError as exc:  # pragma: no cover - dependency is mandatory
        raise RuntimeError("encrypted sessions require the cryptography package") from exc
    return AESGCM, Scrypt


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    if not isinstance(passphrase, str) or len(passphrase) < 12:
        raise ValueError("passphrase must contain at least 12 characters")
    _, Scrypt = _crypto()
    return Scrypt(salt=salt, length=32, n=2**15, r=8, p=1).derive(
        passphrase.encode("utf-8")
    )


def _iter_session_members(root: Path, output: Path) -> Iterator[Path]:
    """Walk real files/directories without ever materializing an unbounded tree."""
    members_seen = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        entries: list[tuple[Path, bool]] = []
        with os.scandir(directory) as scan:
            for entry in scan:
                path = Path(entry.path)
                if path == output or entry.is_symlink():
                    continue
                is_directory = entry.is_dir(follow_symlinks=False)
                if not is_directory and not entry.is_file(follow_symlinks=False):
                    continue
                members_seen += 1
                if members_seen > _MAX_ARCHIVE_MEMBERS:
                    raise ValueError(
                        "session archive exceeds the member-count safety limit"
                    )
                entries.append((path, is_directory))

        # Sorting is bounded by the global member cap above.  Add all members
        # in a directory deterministically, then visit child directories in
        # deterministic order without recursive Python calls.
        entries.sort(key=lambda row: row[0].name)
        for path, _is_directory in entries:
            yield path
        pending.extend(
            path
            for path, is_directory in reversed(entries)
            if is_directory
        )


def encrypt_session_bundle(
    session_dir: Path,
    output_path: Path,
    *,
    passphrase: str,
) -> dict[str, Any]:
    """Create a portable encrypted session bundle without loading it in RAM."""
    root = session_dir.resolve()
    output = output_path.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("session_dir must be a real directory")
    if output.exists():
        raise FileExistsError(str(output))
    output.parent.mkdir(parents=True, exist_ok=True)

    AESGCM, _ = _crypto()
    salt = os.urandom(16)
    nonce_prefix = os.urandom(8)
    key = _derive_key(passphrase, salt)
    header = {
        "format": "sift-encrypted-session",
        "version": 2,
        "cipher": "AES-256-GCM",
        "kdf": "scrypt-n32768-r8-p1",
        "chunk_size": _ENC_CHUNK_SIZE,
        "salt": base64.b64encode(salt).decode("ascii"),
        "nonce_prefix": base64.b64encode(nonce_prefix).decode("ascii"),
        "created_at": _utc_now().isoformat(),
    }
    header_bytes = _canonical_json(header)
    aes = AESGCM(key)
    member_count = 0
    plaintext_bytes = 0
    with tempfile.NamedTemporaryFile(prefix="sift-session-", suffix=".tar", delete=False) as tmp:
        tar_path = Path(tmp.name)
    try:
        with tarfile.open(tar_path, "w") as archive:
            for path in _iter_session_members(root, output):
                if member_count >= _MAX_ARCHIVE_MEMBERS:
                    raise ValueError(
                        "session archive exceeds the member-count safety limit"
                    )
                archive.add(
                    path,
                    arcname=path.relative_to(root).as_posix(),
                    recursive=False,
                )
                member_count += 1
        plaintext_bytes = tar_path.stat().st_size
        if plaintext_bytes > _MAX_ARCHIVE_BYTES:
            raise ValueError("session archive exceeds the 64 GiB safety limit")
        with output.open("xb") as encrypted, tar_path.open("rb") as source:
            encrypted.write(_ENC_MAGIC)
            encrypted.write(header_bytes + b"\n")
            counter = 0
            plaintext_digest = hashlib.sha256()
            while True:
                chunk = source.read(_ENC_CHUNK_SIZE)
                if not chunk:
                    break
                if counter >= _ENC_FINAL_NONCE_COUNTER:
                    raise ValueError("encrypted session has too many chunks")
                nonce = nonce_prefix + counter.to_bytes(4, "big")
                ciphertext = aes.encrypt(
                    nonce,
                    chunk,
                    header_bytes + _ENC_RECORD_DATA + counter.to_bytes(4, "big"),
                )
                encrypted.write(_ENC_RECORD_DATA)
                encrypted.write(struct.pack(">I", len(ciphertext)))
                encrypted.write(ciphertext)
                plaintext_digest.update(chunk)
                counter += 1

            # Per-chunk AEAD alone does not authenticate *completion*: an
            # attacker can remove one or more complete records without
            # invalidating any record that remains.  A separately authenticated
            # terminal record binds the expected count, total plaintext length,
            # and whole-stream digest.  Its reserved nonce can never collide
            # with a data-record nonce.
            final_plaintext = _canonical_json({
                "chunks": counter,
                "plaintext_bytes": plaintext_bytes,
                "plaintext_sha256": plaintext_digest.hexdigest(),
            })
            final_ciphertext = aes.encrypt(
                nonce_prefix + _ENC_FINAL_NONCE_COUNTER.to_bytes(4, "big"),
                final_plaintext,
                header_bytes + _ENC_RECORD_FINAL,
            )
            encrypted.write(_ENC_RECORD_FINAL)
            encrypted.write(struct.pack(">I", len(final_ciphertext)))
            encrypted.write(final_ciphertext)
            encrypted.flush()
            os.fsync(encrypted.fileno())
        _private_file(output)
    except Exception:
        output.unlink(missing_ok=True)
        raise
    finally:
        _overwrite_then_unlink(tar_path)
    return {
        "path": str(output),
        "members": member_count,
        "plaintext_bytes": plaintext_bytes,
        "encrypted_bytes": output.stat().st_size,
        "cipher": header["cipher"],
        "format_version": header["version"],
        "authenticated_completion": True,
    }


def _safe_tar_target(root: Path, member_name: str) -> Path:
    pure = Path(member_name)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError("encrypted session contains an unsafe path")
    target = (root / pure).resolve()
    if target == root or root not in target.parents:
        raise ValueError("encrypted session path escapes destination")
    return target


def _read_encrypted_header(encrypted: Any) -> tuple[int, bytes, dict[str, Any]]:
    magic = encrypted.read(len(_ENC_MAGIC_V2))
    if magic == _ENC_MAGIC_V2:
        version = 2
    elif magic == _ENC_MAGIC_V1:
        version = 1
    else:
        raise ValueError("not a Sift encrypted session")
    header_line = encrypted.readline(16 * 1024)
    if not header_line.endswith(b"\n"):
        raise ValueError("invalid encrypted-session header")
    header_bytes = header_line[:-1]
    try:
        header = json.loads(header_bytes)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid encrypted-session header") from exc
    if (
        not isinstance(header, dict)
        or header.get("format") != "sift-encrypted-session"
        or header.get("version") != version
        or header.get("cipher") != "AES-256-GCM"
        or header.get("kdf") != "scrypt-n32768-r8-p1"
        or header.get("chunk_size") != _ENC_CHUNK_SIZE
    ):
        raise ValueError("unsupported encrypted-session format")
    return version, header_bytes, header


def _decode_encrypted_parameters(
    header: dict[str, Any], *, passphrase: str,
) -> tuple[Any, bytes, bytes]:
    AESGCM, _ = _crypto()
    try:
        salt = base64.b64decode(header["salt"], validate=True)
        nonce_prefix = base64.b64decode(header["nonce_prefix"], validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid encrypted-session parameters") from exc
    if len(salt) != 16 or len(nonce_prefix) != 8:
        raise ValueError("invalid encrypted-session parameters")
    return AESGCM(_derive_key(passphrase, salt)), nonce_prefix, salt


def _decrypt_v2_stream(
    encrypted: Any,
    decrypted: Any,
    *,
    aes: Any,
    nonce_prefix: bytes,
    header_bytes: bytes,
) -> tuple[int, int]:
    """Decrypt a v2 stream and require its authenticated terminal record."""
    counter = 0
    total = 0
    plaintext_digest = hashlib.sha256()
    while True:
        record_type = encrypted.read(1)
        if not record_type:
            raise ValueError("encrypted session is missing its authenticated final record")
        length_bytes = encrypted.read(4)
        if len(length_bytes) != 4:
            raise ValueError("truncated encrypted-session record header")
        length = struct.unpack(">I", length_bytes)[0]

        if record_type == _ENC_RECORD_DATA:
            if counter >= _ENC_FINAL_NONCE_COUNTER:
                raise ValueError("encrypted session has too many chunks")
            if length < 16 or length > _ENC_CHUNK_SIZE + 16:
                raise ValueError("invalid encrypted-session chunk length")
            ciphertext = encrypted.read(length)
            if len(ciphertext) != length:
                raise ValueError("truncated encrypted-session chunk")
            plaintext = aes.decrypt(
                nonce_prefix + counter.to_bytes(4, "big"),
                ciphertext,
                header_bytes + _ENC_RECORD_DATA + counter.to_bytes(4, "big"),
            )
            total += len(plaintext)
            if total > _MAX_ARCHIVE_BYTES:
                raise ValueError("decrypted session exceeds the safety limit")
            plaintext_digest.update(plaintext)
            decrypted.write(plaintext)
            counter += 1
            continue

        if record_type != _ENC_RECORD_FINAL:
            raise ValueError("encrypted session contains an unknown record type")
        if length < 16 or length > _ENC_FINAL_MAX_BYTES:
            raise ValueError("invalid encrypted-session final-record length")
        ciphertext = encrypted.read(length)
        if len(ciphertext) != length:
            raise ValueError("truncated encrypted-session final record")
        final_plaintext = aes.decrypt(
            nonce_prefix + _ENC_FINAL_NONCE_COUNTER.to_bytes(4, "big"),
            ciphertext,
            header_bytes + _ENC_RECORD_FINAL,
        )
        try:
            final = json.loads(final_plaintext)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid encrypted-session final record") from exc
        expected = {
            "chunks": counter,
            "plaintext_bytes": total,
            "plaintext_sha256": plaintext_digest.hexdigest(),
        }
        if final != expected:
            raise ValueError("encrypted-session completion metadata did not match")
        if encrypted.read(1):
            raise ValueError("encrypted session contains trailing data")
        return counter, total


def _decrypt_v1_stream(
    encrypted: Any,
    decrypted: Any,
    *,
    aes: Any,
    nonce_prefix: bytes,
    header_bytes: bytes,
) -> tuple[int, int]:
    """Decode legacy per-chunk-authenticated v1 framing.

    V1 has no authenticated completion marker.  Callers must explicitly opt in
    to this compatibility path, and its result must never be described as a
    fully authenticated bundle.
    """
    counter = 0
    total = 0
    while True:
        length_bytes = encrypted.read(4)
        if not length_bytes:
            break
        if len(length_bytes) != 4:
            raise ValueError("truncated encrypted-session chunk header")
        length = struct.unpack(">I", length_bytes)[0]
        if length < 16 or length > _ENC_CHUNK_SIZE + 16:
            raise ValueError("invalid encrypted-session chunk length")
        ciphertext = encrypted.read(length)
        if len(ciphertext) != length:
            raise ValueError("truncated encrypted-session chunk")
        if counter >= 2**32:
            raise ValueError("encrypted session has too many chunks")
        plaintext = aes.decrypt(
            nonce_prefix + counter.to_bytes(4, "big"),
            ciphertext,
            header_bytes + counter.to_bytes(4, "big"),
        )
        total += len(plaintext)
        if total > _MAX_ARCHIVE_BYTES:
            raise ValueError("decrypted session exceeds the safety limit")
        decrypted.write(plaintext)
        counter += 1
    return counter, total


def _extract_session_tar(tar_path: Path, destination: Path) -> int:
    """Extract a verified tar in streaming mode with count/size bounds."""
    extracted = 0
    members_seen = 0
    declared = 0
    with tarfile.open(tar_path, "r|") as archive:
        for member in archive:
            members_seen += 1
            if members_seen > _MAX_ARCHIVE_MEMBERS:
                raise ValueError("session archive exceeds the member-count safety limit")
            target = _safe_tar_target(destination, member.name)
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError("links and device files are not allowed")
            if member.isdir():
                target.mkdir(mode=0o700, parents=True, exist_ok=True)
                current = destination
                for part in target.relative_to(destination).parts:
                    current = current / part
                    _private_directory(current)
                continue
            if not member.isfile():
                raise ValueError("unsupported archive member type")
            if member.size < 0:
                raise ValueError("session archive member has an invalid size")
            declared += member.size
            if declared > _MAX_ARCHIVE_BYTES:
                raise ValueError("session members exceed the safety limit")
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            current = destination
            for part in target.parent.relative_to(destination).parts:
                current = current / part
                _private_directory(current)
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError("archive member could not be read")
            with target.open("xb") as output:
                shutil.copyfileobj(handle, output, length=1024 * 1024)
            _private_file(target)
            extracted += 1
    return extracted


def restore_encrypted_session(
    encrypted_path: Path,
    destination: Path,
    *,
    passphrase: str,
    allow_legacy_v1: bool = False,
) -> dict[str, Any]:
    """Authenticate and restore an encrypted bundle into an empty directory.

    V2 authenticates both every chunk and the end of the byte stream.  Legacy
    v1 bundles lack an authenticated completion marker and are rejected unless
    ``allow_legacy_v1=True`` is supplied explicitly; a legacy restore reports
    ``authenticated=False`` even when all present chunks validate.
    """
    source = encrypted_path.resolve()
    requested_destination = Path(destination).expanduser()
    is_junction = getattr(requested_destination, "is_junction", lambda: False)
    if requested_destination.is_symlink() or is_junction():
        raise ValueError("destination cannot be a link or junction")
    dest = requested_destination.resolve()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError("destination must be absent or empty")
    if dest.exists() and not dest.is_dir():
        raise NotADirectoryError(str(dest))
    destination_existed = dest.exists()
    destination_identity = dest.stat() if destination_existed else None
    if destination_existed:
        _private_directory(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".sift-restore-", dir=dest.parent))
    _private_directory(staging)
    with tempfile.NamedTemporaryFile(
        prefix="sift-restore-", suffix=".tar", delete=False
    ) as tmp:
        tar_path = Path(tmp.name)
    extracted = 0
    version = 0
    try:
        with source.open("rb") as encrypted, tar_path.open("wb") as decrypted:
            version, header_bytes, header = _read_encrypted_header(encrypted)
            if version == 1 and not allow_legacy_v1:
                raise ValueError(
                    "legacy encrypted-session v1 has no authenticated completion "
                    "marker; pass allow_legacy_v1=True only to perform an "
                    "explicit unauthenticated-completion compatibility restore"
                )
            aes, nonce_prefix, _salt = _decode_encrypted_parameters(
                header, passphrase=passphrase,
            )
            if version == 2:
                _decrypt_v2_stream(
                    encrypted,
                    decrypted,
                    aes=aes,
                    nonce_prefix=nonce_prefix,
                    header_bytes=header_bytes,
                )
            else:
                _decrypt_v1_stream(
                    encrypted,
                    decrypted,
                    aes=aes,
                    nonce_prefix=nonce_prefix,
                    header_bytes=header_bytes,
                )
            decrypted.flush()
            os.fsync(decrypted.fileno())
        extracted = _extract_session_tar(tar_path, staging)
        if destination_existed:
            # This is a security invariant, not a developer assertion:
            # optimized Python removes ``assert`` statements entirely.
            if destination_identity is None:
                raise RuntimeError("destination identity was not captured")
            current_identity = dest.stat()
            if (
                current_identity.st_dev != destination_identity.st_dev
                or current_identity.st_ino != destination_identity.st_ino
                or any(dest.iterdir())
            ):
                raise FileExistsError("destination changed while the session restored")
            try:
                os.replace(staging, dest)
            except OSError:
                # Windows cannot always replace an existing directory even
                # when it is empty. Preserve the researcher's original empty
                # destination if the fallback move itself fails.
                dest.rmdir()
                try:
                    os.replace(staging, dest)
                except Exception:
                    dest.mkdir(mode=0o700)
                    _private_directory(dest)
                    raise
        elif dest.exists():
            raise FileExistsError("destination appeared while the session restored")
        else:
            os.replace(staging, dest)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        _overwrite_then_unlink(tar_path)
    authenticated = version == 2
    return {
        "path": str(dest),
        "files": extracted,
        "authenticated": authenticated,
        "format_version": version,
        "authentication_status": (
            "authenticated_complete"
            if authenticated
            else "legacy_chunks_authenticated_completion_unverified"
        ),
    }


# ---------------------------------------------------------------------------
# Signed provenance exports
# ---------------------------------------------------------------------------


def _signing_crypto() -> tuple[Any, Any, Any]:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("signed provenance requires cryptography") from exc
    return serialization, Ed25519PrivateKey, Ed25519PublicKey


def generate_ed25519_private_key() -> str:
    serialization, Ed25519PrivateKey, _ = _signing_crypto()
    key = Ed25519PrivateKey.generate().private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return base64.b64encode(key).decode("ascii")


def _load_or_create_signing_key(key_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", key_id):
        raise ValueError("invalid signing key id")
    try:
        import keyring
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OS credential storage is unavailable") from exc
    service = "Sift provenance signing"
    value = keyring.get_password(service, key_id)
    if value:
        return value
    value = generate_ed25519_private_key()
    keyring.set_password(service, key_id, value)
    return value


def _bundle_manifest(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda p: p.as_posix()):
        if path.name == _SIGNATURE_NAME or path.is_symlink() or not path.is_file():
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return {"version": 1, "hash_algorithm": "SHA-256", "files": files}


def sign_provenance_export(
    bundle_dir: Path,
    *,
    key_id: str = "default",
    private_key_b64: str | None = None,
) -> dict[str, Any]:
    """Sign the complete regular-file manifest of an export directory."""
    root = bundle_dir.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("bundle_dir must be a real directory")
    serialization, Ed25519PrivateKey, _ = _signing_crypto()
    encoded = private_key_b64 or _load_or_create_signing_key(key_id)
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) != 32:
        raise ValueError("invalid Ed25519 private key")
    private_key = Ed25519PrivateKey.from_private_bytes(raw)
    manifest = _bundle_manifest(root)
    payload = _canonical_json(manifest)
    signature = private_key.sign(payload)
    public = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    envelope = {
        "format": "sift-provenance-signature",
        "version": 1,
        "algorithm": "Ed25519",
        "key_id": key_id,
        "signed_at": _utc_now().isoformat(),
        "public_key": base64.b64encode(public).decode("ascii"),
        "signature": base64.b64encode(signature).decode("ascii"),
        "manifest": manifest,
    }
    destination = root / _SIGNATURE_NAME
    from sift.reliability import atomic_write_json
    atomic_write_json(destination, envelope, use_lock=False)
    return {"path": str(destination), "files": len(manifest["files"]), "key_id": key_id}


def verify_provenance_signature(bundle_dir: Path) -> dict[str, Any]:
    root = bundle_dir.resolve()
    signature_path = root / _SIGNATURE_NAME
    try:
        envelope = json.loads(signature_path.read_text(encoding="utf-8"))
        if (
            envelope.get("format") != "sift-provenance-signature"
            or envelope.get("version") != 1
            or envelope.get("algorithm") != "Ed25519"
        ):
            raise ValueError("unsupported signature envelope")
        expected = envelope["manifest"]
        actual = _bundle_manifest(root)
        if actual != expected:
            return {"valid": False, "reason": "file manifest does not match"}
        _, _, Ed25519PublicKey = _signing_crypto()
        public = base64.b64decode(envelope["public_key"], validate=True)
        signature = base64.b64decode(envelope["signature"], validate=True)
        Ed25519PublicKey.from_public_bytes(public).verify(
            signature, _canonical_json(expected)
        )
        return {
            "valid": True,
            "files": len(expected.get("files", [])),
            "key_id": envelope.get("key_id"),
        }
    except Exception as exc:  # noqa: BLE001 - verifier is a safe boundary
        return {"valid": False, "reason": str(exc)}


# ---------------------------------------------------------------------------
# SBOM and local security qualification
# ---------------------------------------------------------------------------

_SECURITY_SOURCE_ROOTS = ("src", "scripts", "packaging", "siftbench", "docs")
_SECURITY_SOURCE_FILES = (
    ".gitignore", "pyproject.toml", "CHANGELOG.md", "LICENSE", "README.md",
    "SECURITY.md",
)
# Qualification outputs are derived evidence, not release inputs.  In
# particular, the live database report is deliberately refreshed *after* a
# wheel is built so that it can bind itself to that wheel.  Treating the
# refreshed report as source would make the artifact stale the moment the
# qualifier recorded that binding, creating an impossible build/report cycle.
_SECURITY_DERIVED_EVIDENCE_FILES = frozenset({
    "docs/live_database_certification.json",
})
_MAX_SECURITY_SURFACE_FILE_BYTES = 16 * 1024 * 1024
_MAX_SECURITY_SURFACE_TOTAL_BYTES = 512 * 1024 * 1024
_MAX_SECURITY_SURFACE_MEMBERS = 10_000


def _content_manifest_sha256(entries: Iterable[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    count = 0
    for relative, content in sorted(entries):
        count += 1
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest() if count else ""


def _current_security_source_entries(root: Path) -> list[tuple[str, bytes]] | None:
    entries: list[tuple[str, bytes]] = []
    for root_name in _SECURITY_SOURCE_ROOTS:
        source_root = root / root_name
        if not source_root.exists():
            continue
        if source_root.is_symlink() or not source_root.is_dir():
            return None
        for path in sorted(source_root.rglob("*")):
            if path.is_dir():
                continue
            relative = path.relative_to(root)
            # Desktop builds materialize a complete third-party interpreter
            # under packaging/vendor/python. Hatch excludes that tree from
            # release archives, so it must not participate in the current-
            # source side of the artifact binding either.
            if _is_generated_security_scan_path(path, root):
                continue
            if relative.as_posix() in _SECURITY_DERIVED_EVIDENCE_FILES:
                continue
            if (
                "__pycache__" in relative.parts
                or path.suffix.casefold() in {".pyc", ".pyo"}
                or path.name == ".DS_Store"
            ):
                continue
            if path.is_symlink() or not path.is_file():
                return None
            entries.append((relative.as_posix(), path.read_bytes()))
    for name in _SECURITY_SOURCE_FILES:
        path = root / name
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file():
            return None
        entries.append((name, path.read_bytes()))
    return entries or None


def _sdist_security_source_entries(
    sdist: Path, *, include_all_files: bool = False,
) -> list[tuple[str, bytes]] | None:
    entries: list[tuple[str, bytes]] = []
    try:
        with tarfile.open(sdist, "r:gz") as archive:
            root_name: str | None = None
            member_count = 0
            seen_files: set[str] = set()
            total = 0
            for member in archive:
                member_count += 1
                if member_count > _MAX_SECURITY_SURFACE_MEMBERS:
                    return None
                archive_path = PurePosixPath(member.name)
                if (
                    archive_path.is_absolute()
                    or len(archive_path.parts) < 2
                    or any(part in {"", ".", ".."} for part in archive_path.parts)
                ):
                    return None
                if root_name is None:
                    root_name = archive_path.parts[0]
                elif archive_path.parts[0] != root_name:
                    return None
                relative = PurePosixPath(*archive_path.parts[1:]).as_posix()
                if member.isdir():
                    continue
                if not member.isfile() or relative in seen_files:
                    return None
                seen_files.add(relative)
                path = PurePosixPath(relative)
                if relative in _SECURITY_DERIVED_EVIDENCE_FILES:
                    continue
                included = include_all_files or (
                    path.parts[0] in _SECURITY_SOURCE_ROOTS
                    or relative in _SECURITY_SOURCE_FILES
                )
                if not included:
                    continue
                if member.size > _MAX_SECURITY_SURFACE_FILE_BYTES:
                    return None
                total += member.size
                if total > _MAX_SECURITY_SURFACE_TOTAL_BYTES:
                    return None
                extracted = archive.extractfile(member)
                if extracted is None:
                    return None
                content = extracted.read()
                if len(content) != member.size:
                    return None
                entries.append((relative, content))
            if member_count == 0 or root_name is None:
                return None
    except (OSError, tarfile.TarError, ValueError):
        return None
    return entries or None


def _source_package_sha256(root: Path) -> str | None:
    source = root / "src" / "sift"
    if source.is_symlink() or not source.is_dir():
        return None
    entries: list[tuple[str, bytes]] = []
    for path in sorted(source.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(source)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink() or not path.is_file():
            return None
        entries.append((relative.as_posix(), path.read_bytes()))
    return _content_manifest_sha256(entries) or None


def _wheel_package_sha256(wheel: Path) -> str | None:
    try:
        with zipfile.ZipFile(wheel) as archive:
            infos = archive.infolist()
            if len(infos) > _MAX_SECURITY_SURFACE_MEMBERS:
                return None
            package_infos: list[tuple[zipfile.ZipInfo, str]] = []
            seen: set[str] = set()
            total = 0
            for info in infos:
                path = PurePosixPath(info.filename)
                if (
                    path.is_absolute()
                    or any(part in {"", ".", ".."} for part in path.parts)
                    or info.filename in seen
                ):
                    return None
                seen.add(info.filename)
                if info.is_dir():
                    continue
                # A wheel is an installable archive, not a symlink container.
                if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                    return None
                if (
                    info.file_size > _MAX_SECURITY_SURFACE_FILE_BYTES
                    or info.file_size < 0
                ):
                    return None
                total += info.file_size
                if total > _MAX_SECURITY_SURFACE_TOTAL_BYTES:
                    return None
                if len(path.parts) >= 2 and path.parts[0] == "sift":
                    package_infos.append((info, PurePosixPath(*path.parts[1:]).as_posix()))
            if not package_infos:
                return None
            entries = []
            for info, relative in package_infos:
                content = archive.read(info)
                if len(content) != info.file_size:
                    return None
                entries.append((relative, content))
    except (OSError, ValueError, zipfile.BadZipFile):
        return None
    return _content_manifest_sha256(entries) or None


def _sdist_package_sha256(sdist: Path) -> str | None:
    entries = _sdist_security_source_entries(sdist)
    if entries is None:
        return None
    package_entries = []
    prefix = "src/sift/"
    for relative, content in entries:
        if relative.startswith(prefix):
            package_entries.append((relative[len(prefix):], content))
    return _content_manifest_sha256(package_entries) or None


def _release_metadata(path: Path, *, kind: str) -> dict[str, Any] | None:
    raw: bytes | None = None
    try:
        if kind == "wheel":
            with zipfile.ZipFile(path) as archive:
                infos = archive.infolist()
                if len(infos) > _MAX_SECURITY_SURFACE_MEMBERS:
                    return None
                seen: set[str] = set()
                metadata: list[zipfile.ZipInfo] = []
                total = 0
                for info in infos:
                    archive_path = PurePosixPath(info.filename)
                    if (
                        archive_path.is_absolute()
                        or any(part in {"", ".", ".."} for part in archive_path.parts)
                        or info.filename in seen
                    ):
                        return None
                    seen.add(info.filename)
                    if info.is_dir():
                        continue
                    if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                        return None
                    if info.file_size < 0 or info.file_size > _MAX_SECURITY_SURFACE_FILE_BYTES:
                        return None
                    total += info.file_size
                    if total > _MAX_SECURITY_SURFACE_TOTAL_BYTES:
                        return None
                    if info.filename.endswith(".dist-info/METADATA"):
                        metadata.append(info)
                if len(metadata) != 1:
                    return None
                raw = archive.read(metadata[0])
                if len(raw) != metadata[0].file_size:
                    return None
        else:
            with tarfile.open(path, "r:gz") as archive:
                root_name: str | None = None
                member_count = 0
                for member in archive:
                    member_count += 1
                    if member_count > _MAX_SECURITY_SURFACE_MEMBERS:
                        return None
                    archive_path = PurePosixPath(member.name)
                    if (
                        archive_path.is_absolute()
                        or len(archive_path.parts) < 2
                        or any(part in {"", ".", ".."} for part in archive_path.parts)
                    ):
                        return None
                    if root_name is None:
                        root_name = archive_path.parts[0]
                    elif archive_path.parts[0] != root_name:
                        return None
                    relative = PurePosixPath(*archive_path.parts[1:]).as_posix()
                    if member.isdir():
                        continue
                    if not member.isfile():
                        return None
                    if relative != "PKG-INFO":
                        continue
                    if raw is not None or member.size > _MAX_SECURITY_SURFACE_FILE_BYTES:
                        return None
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        return None
                    raw = extracted.read()
                    if len(raw) != member.size:
                        return None
                if raw is None:
                    return None
        message = email.parser.BytesParser(policy=email.policy.default).parsebytes(raw)
        requirements = sorted(str(value) for value in message.get_all("Requires-Dist", []))
        return {
            "name": str(message.get("Name", "")),
            "version": str(message.get("Version", "")),
            "requires_dist": requirements,
            "metadata_sha256": hashlib.sha256(raw).hexdigest(),
        }
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile):
        return None


def security_release_binding(project_root: Path) -> dict[str, Any]:
    """Bind qualification to current source, lock, wheel, and source archive."""
    root = Path(project_root).resolve()
    lock = root / "uv.lock"
    pyproject = root / "pyproject.toml"
    dist = root / "dist"
    wheels = sorted(dist.glob("sift-*.whl")) if dist.is_dir() else []
    sdists = sorted(dist.glob("sift-*.tar.gz")) if dist.is_dir() else []
    source_entries = _current_security_source_entries(root)
    safe = bool(
        source_entries
        and lock.is_file() and not lock.is_symlink()
        and pyproject.is_file() and not pyproject.is_symlink()
        and len(wheels) == 1 and wheels[0].is_file() and not wheels[0].is_symlink()
        and len(sdists) == 1 and sdists[0].is_file() and not sdists[0].is_symlink()
    )
    if not safe:
        return {"status": "invalid", "reason": "exact source, lock, wheel, and sdist are required"}
    wheel, sdist = wheels[0], sdists[0]
    source_package = _source_package_sha256(root)
    wheel_package = _wheel_package_sha256(wheel)
    sdist_package = _sdist_package_sha256(sdist)
    wheel_metadata = _release_metadata(wheel, kind="wheel")
    sdist_metadata = _release_metadata(sdist, kind="sdist")
    sdist_entries = _sdist_security_source_entries(sdist)
    source_tree = _content_manifest_sha256(source_entries or [])
    sdist_tree = _content_manifest_sha256(sdist_entries or [])
    package_matches = bool(
        source_package and source_package == wheel_package == sdist_package
    )
    tree_matches = bool(source_tree and source_tree == sdist_tree)
    locked_versions: dict[str, set[Any]] = {}
    declared_names: set[str] = set()
    parsed_requirements: list[Any] = []
    requirements_valid = False
    try:
        # ``Requires-Dist`` is PEP 508, so a name-only regular expression is
        # not enough: malformed requirements and constraints which no locked
        # version satisfies must both make the release binding stale.
        try:
            from packaging.requirements import Requirement
            from packaging.version import Version
        except ImportError:
            # Source checkouts contain Sift's top-level ``packaging/`` build
            # directory, which can shadow the third-party distribution. Pip's
            # vendored copy implements the same PEP 440/508 contracts and is
            # available in qualification environments even in that case.
            from pip._vendor.packaging.requirements import Requirement  # type: ignore[no-redef,assignment]
            from pip._vendor.packaging.version import Version  # type: ignore[no-redef,assignment]

        with pyproject.open("rb") as handle:
            project_version = str(tomllib.load(handle)["project"]["version"])
        with lock.open("rb") as handle:
            lock_packages = tomllib.load(handle).get("package", [])
        for package in lock_packages:
            if not isinstance(package, dict):
                continue
            source = package.get("source")
            if not isinstance(source, dict) or "registry" not in source:
                continue
            raw_name, raw_version = package.get("name"), package.get("version")
            if (
                not isinstance(raw_name, str)
                or not re.fullmatch(r"[A-Za-z0-9_.-]+", raw_name)
                or not isinstance(raw_version, str)
                or not raw_version
            ):
                raise ValueError("locked dependency identity is invalid")
            name = re.sub(r"[-_.]+", "-", raw_name).casefold()
            version = Version(raw_version)
            locked_versions.setdefault(name, set()).add(version)

        for raw in (wheel_metadata or {}).get("requires_dist", []):
            requirement = Requirement(raw)
            # A direct URL cannot be proven to equal a registry lock pin by
            # package name/version, even if its display name happens to match.
            if requirement.url is not None:
                raise ValueError("direct-reference dependency is not lock-verifiable")
            parsed_requirements.append(requirement)
            declared_names.add(
                re.sub(r"[-_.]+", "-", requirement.name).casefold()
            )
        requirements_valid = True
    except (ImportError, OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError):
        project_version = ""
        locked_versions = {}
        declared_names = set()
    metadata_match = bool(
        wheel_metadata and sdist_metadata
        and wheel_metadata["name"].casefold() == sdist_metadata["name"].casefold() == "sift"
        and wheel_metadata["version"] == sdist_metadata["version"] == project_version
        and wheel_metadata["requires_dist"] == sdist_metadata["requires_dist"]
    )
    dependencies_covered = bool(
        metadata_match
        and requirements_valid
        and all(
            any(version in requirement.specifier for version in locked_versions.get(
                re.sub(r"[-_.]+", "-", requirement.name).casefold(), set()
            ))
            for requirement in parsed_requirements
        )
    )
    body: dict[str, Any] = {
        "status": (
            "ready" if package_matches and tree_matches and dependencies_covered else "stale"
        ),
        "source_tree_sha256": source_tree or None,
        "source_tree_files": len(source_entries or []),
        "sdist_source_tree_sha256": sdist_tree or None,
        "sdist_source_tree_files": len(sdist_entries or []),
        "source_package_sha256": source_package,
        "wheel_package_sha256": wheel_package,
        "sdist_package_sha256": sdist_package,
        "release_packages_match_source": package_matches,
        "sdist_matches_source_tree": tree_matches,
        "artifact_metadata_match": metadata_match,
        "declared_requirements_valid": requirements_valid,
        "declared_dependencies_covered_by_lock": dependencies_covered,
        "declared_dependency_count": len(declared_names),
        "declared_dependencies_sha256": hashlib.sha256(
            _canonical_json((wheel_metadata or {}).get("requires_dist", [])),
        ).hexdigest(),
        "wheel_metadata_sha256": (wheel_metadata or {}).get("metadata_sha256"),
        "sdist_metadata_sha256": (sdist_metadata or {}).get("metadata_sha256"),
        "pyproject_sha256": _sha256_file(pyproject),
        "uv_lock_sha256": _sha256_file(lock),
        "wheel": {
            "path": wheel.relative_to(root).as_posix(),
            "sha256": _sha256_file(wheel), "size_bytes": wheel.stat().st_size,
        },
        "sdist": {
            "path": sdist.relative_to(root).as_posix(),
            "sha256": _sha256_file(sdist), "size_bytes": sdist.stat().st_size,
        },
    }
    body["binding_sha256"] = hashlib.sha256(_canonical_json(body)).hexdigest()
    return body


def generate_cyclonedx_sbom(project_root: Path, output_path: Path) -> dict[str, Any]:
    """Generate a CycloneDX 1.6 JSON SBOM from the locked environment."""
    root = project_root.resolve()
    lock_path = root / "uv.lock"
    if not lock_path.is_file():
        raise FileNotFoundError("uv.lock is required for an exact SBOM")
    with lock_path.open("rb") as handle:
        lock = tomllib.load(handle)
    packages = [p for p in lock.get("package", []) if isinstance(p, dict)]
    by_name = {str(p.get("name", "")).casefold(): p for p in packages}
    sift_package = by_name.get("sift", {})
    direct = {
        str(dep.get("name", "")).casefold()
        for dep in sift_package.get("dependencies", [])
        if isinstance(dep, dict)
    }
    components: list[dict[str, Any]] = []
    for package in sorted(packages, key=lambda p: str(p.get("name", ""))):
        name = str(package.get("name", ""))
        version = str(package.get("version", ""))
        if not name or not version or name.casefold() == "sift":
            continue
        normalized = name.replace("_", "-").casefold()
        component = {
            "type": "library",
            "bom-ref": f"pkg:pypi/{normalized}@{version}",
            "name": name,
            "version": version,
            "purl": f"pkg:pypi/{normalized}@{version}",
            "properties": [
                {
                    "name": "sift:dependency-scope",
                    "value": "direct" if name.casefold() in direct else "transitive",
                }
            ],
        }
        components.append(component)
    root_ref = f"pkg:pypi/sift@{sift_package.get('version', '0.1.0')}"
    document = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": _utc_now().isoformat(),
            "component": {
                "type": "application",
                "bom-ref": root_ref,
                "name": "sift",
                "version": str(sift_package.get("version", "0.1.0")),
                "purl": root_ref,
            },
            "properties": [
                {"name": "sift:source-lock", "value": "uv.lock"},
                {"name": "sift:source-lock-sha256", "value": _sha256_file(lock_path)},
            ],
        },
        "components": components,
        "dependencies": [
            {
                "ref": root_ref,
                "dependsOn": [
                    row["bom-ref"]
                    for row in components
                    if row["properties"][0]["value"] == "direct"
                ],
            }
        ],
    }
    from sift.reliability import atomic_write_json
    atomic_write_json(output_path, document)
    return {"path": str(output_path), "components": len(components), "spec": "1.6"}


_SECRET_SCAN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("openai_style_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (
        "assigned_secret",
        re.compile(
            r"(?i)\b(?:api[_-]?key|password|secret|token)\s*=\s*"
            r"['\"][A-Za-z0-9_./+=-]{16,}['\"]"
        ),
    ),
)

# Maintainer-local generated runtimes are not source surfaces.  In
# particular, ``packaging/vendor/python`` can contain more than a gigabyte of
# third-party wheels after a desktop build; scanning it both reports false
# positives from dependency fixtures/assets and turns a sub-second project
# scan into a multi-minute crawl.  Published artifacts are scanned through
# ``sdist_path`` above, so excluding this generated tree cannot hide shipped
# Sift source.
_DEVELOPER_SECURITY_SCAN_EXCLUDED_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("packaging", "vendor", "python"),
)


def _is_generated_security_scan_path(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return True
    return any(
        parts[: len(prefix)] == prefix
        for prefix in _DEVELOPER_SECURITY_SCAN_EXCLUDED_PREFIXES
    ) or "__pycache__" in parts


def _security_text_surfaces(
    project_root: Path, *, sdist_path: Path | None = None,
) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    root = project_root.resolve()
    surfaces: list[tuple[str, str]] = []
    errors: list[dict[str, Any]] = []
    if sdist_path is not None:
        entries = _sdist_security_source_entries(
            Path(sdist_path), include_all_files=True,
        )
        if entries is None:
            return [], [{
                "severity": "high", "category": "unsafe_or_unreadable_sdist",
                "path": Path(sdist_path).name, "line": 0,
            }]
        for relative, content in entries:
            try:
                surfaces.append((relative, content.decode("utf-8")))
            except UnicodeDecodeError:
                continue
        return surfaces, errors

    # Developer fallback for targeted checks before release artifacts exist.
    targets = [root / "src", root / "scripts", root / "packaging", root / "siftbench"]
    targets.extend(path for path in (root / "pyproject.toml",) if path.exists())
    paths = []
    for target in targets:
        if target.is_file() and not target.is_symlink():
            paths.append(target)
        elif target.is_dir() and not target.is_symlink():
            paths.extend(
                path for path in target.rglob("*")
                if (
                    path.is_file()
                    and not path.is_symlink()
                    and not _is_generated_security_scan_path(path, root)
                )
            )
    for path in sorted(set(paths)):
        try:
            surfaces.append((path.relative_to(root).as_posix(), path.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError):
            continue
    return surfaces, errors


def scan_source_secrets(
    project_root: Path, *, sdist_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Conservative secret scan over every decodable shipped sdist surface."""
    findings: list[dict[str, Any]] = []
    surfaces, errors = _security_text_surfaces(project_root, sdist_path=sdist_path)
    findings.extend(errors)
    for relative, text in surfaces:
        for number, line in enumerate(text.splitlines(), 1):
            if "sift-secret-scan: allow" in line:
                continue
            for category, pattern in _SECRET_SCAN_PATTERNS:
                if pattern.search(line):
                    findings.append(
                        {
                            "severity": "high",
                            "category": category,
                            "path": relative,
                            "line": number,
                        }
                    )
    return findings


def scan_python_static_security(
    project_root: Path, *, sdist_path: Path | None = None,
) -> list[dict[str, Any]]:
    """AST-based checks over every shipped Python surface; never executes it."""
    findings: list[dict[str, Any]] = []
    surfaces, errors = _security_text_surfaces(project_root, sdist_path=sdist_path)
    findings.extend(errors)
    for relative, text in surfaces:
        if Path(relative).suffix.casefold() != ".py":
            continue
        try:
            tree = ast.parse(text, filename=relative)
        except SyntaxError as exc:
            findings.append(
                {
                    "severity": "high",
                    "category": "unparseable_python",
                    "path": relative,
                    "line": getattr(exc, "lineno", 0) or 0,
                }
            )
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                value = node.value
                for target in targets:
                    if not isinstance(target, ast.Attribute):
                        continue
                    if (
                        target.attr == "check_hostname"
                        and isinstance(value, ast.Constant)
                        and value.value is False
                    ):
                        findings.append({
                            "severity": "high",
                            "category": "tls_hostname_verification_disabled",
                            "path": relative,
                            "line": getattr(node, "lineno", 0),
                        })
                    if (
                        target.attr == "verify_mode"
                        and isinstance(value, ast.Attribute)
                        and value.attr == "CERT_NONE"
                    ):
                        findings.append({
                            "severity": "high",
                            "category": "tls_certificate_verification_disabled",
                            "path": relative,
                            "line": getattr(node, "lineno", 0),
                        })
            if not isinstance(node, ast.Call):
                continue
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                base = node.func.value.id if isinstance(node.func.value, ast.Name) else ""
                name = f"{base}.{node.func.attr}" if base else node.func.attr
            category: str | None = None
            severity = "high"
            if name in {"eval", "exec"}:
                category = "dynamic_code_execution"
            elif name in {"pickle.load", "pickle.loads"}:
                category = "unsafe_deserialization"
            elif name in {
                "ssl._create_unverified_context", "_create_unverified_context",
            }:
                category = "tls_verification_disabled"
            elif name == "tempfile.mktemp":
                category = "insecure_temporary_file"
            elif any(
                kw.arg == "verify"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is False
                for kw in node.keywords
            ):
                category = "tls_verification_disabled"
            elif name in {"hashlib.md5", "hashlib.sha1"} and not any(
                kw.arg == "usedforsecurity"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is False
                for kw in node.keywords
            ):
                category = "weak_hash_without_integrity_marker"
            elif name in {
                "subprocess.run", "subprocess.call", "subprocess.Popen",
                "subprocess.check_call", "subprocess.check_output", "Popen",
            }:
                for kw in node.keywords:
                    if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        category = "subprocess_shell_true"
            elif name in {"yaml.load", "yaml.unsafe_load", "load"}:
                # Only flag an explicit yaml.load without a safe loader. Bare
                # ``load`` is ignored because it is too ambiguous.
                if name == "yaml.unsafe_load" or (
                    name == "yaml.load" and not any(
                        kw.arg == "Loader" for kw in node.keywords
                    )
                ):
                    category = "unsafe_yaml_load"
            if category:
                findings.append(
                    {
                        "severity": severity,
                        "category": category,
                        "path": relative,
                        "line": getattr(node, "lineno", 0),
                    }
                )
    return findings


def run_dependency_vulnerability_scan(
    project_root: Path, *, timeout: int = 300
) -> dict[str, Any]:
    """Audit every registry package/version pinned by ``uv.lock``.

    Exact generated manifests are used with dependency resolution disabled;
    the launcher environment is never an audit input. Disjoint platform
    versions of one distribution are split across manifests so none is lost.
    """
    root = Path(project_root).resolve()
    binding = security_release_binding(root)
    if binding.get("status") != "ready":
        return {
            "status": "scanner_error", "tool": "pip-audit",
            "reason": "current source, uv.lock, wheel, and sdist are not exactly bound",
            "release_binding_sha256": binding.get("binding_sha256"),
        }
    executable = _find_tool("pip-audit")
    if not executable:
        return {
            "status": "unavailable", "tool": "pip-audit",
            "reason": "pip-audit is not installed",
            "release_binding_sha256": binding.get("binding_sha256"),
        }
    try:
        version_result = run_bounded_capture(
            [executable, "--version"], timeout=min(timeout, 30), check=False,
            cwd=str(root), stdout_limit=64 * 1024, stderr_limit=64 * 1024,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "scanner_error", "tool": "pip-audit", "reason": str(exc)}
    scanner_version = (version_result.stdout or version_result.stderr or "").strip()[:500]
    if version_result.returncode != 0 or not scanner_version:
        return {
            "status": "scanner_error", "tool": "pip-audit",
            "reason": "scanner version could not be established",
            "scanner_version": scanner_version or None,
        }

    try:
        with (root / "uv.lock").open("rb") as handle:
            lock = tomllib.load(handle)
        packages: list[tuple[str, str]] = []
        for package in lock.get("package", []):
            if not isinstance(package, dict):
                continue
            source = package.get("source")
            if not isinstance(source, dict) or "registry" not in source:
                continue
            name, version = package.get("name"), package.get("version")
            if (
                not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name)
                or not isinstance(version, str) or not version
                or any(character in version for character in "\r\n; ")
            ):
                raise ValueError("invalid registry package pin")
            packages.append((name, version))
        packages = sorted(set(packages), key=lambda row: (row[0].casefold(), row[1]))
        if not packages:
            raise ValueError("no auditable registry packages")
    except (OSError, ValueError, TypeError, tomllib.TOMLDecodeError) as exc:
        return {
            "status": "scanner_error", "tool": "pip-audit",
            "reason": f"locked dependency inventory is invalid ({type(exc).__name__})",
            "scanner_version": scanner_version,
        }

    by_name: dict[str, list[tuple[str, str]]] = {}
    for package in packages:
        normalized = re.sub(r"[-_.]+", "-", package[0]).casefold()
        by_name.setdefault(normalized, []).append(package)
    manifests: list[list[tuple[str, str]]] = [
        [] for _ in range(max(len(rows) for rows in by_name.values()))
    ]
    for rows in by_name.values():
        for index, package in enumerate(rows):
            manifests[index].append(package)

    audited_packages: set[tuple[str, str]] = set()
    findings: list[dict[str, Any]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="sift-dependency-audit-") as temporary:
            temporary_root = Path(temporary)
            for index, manifest in enumerate(manifests):
                requirements = temporary_root / f"locked-{index}.txt"
                requirements.write_text(
                    "".join(f"{name}=={version}\n" for name, version in sorted(manifest)),
                    encoding="utf-8",
                )
                completed = run_bounded_capture(
                    [
                        executable, "--requirement", str(requirements), "--no-deps",
                        "--disable-pip", "--strict", "--format=json",
                        "--progress-spinner=off",
                    ],
                    timeout=timeout, check=False, cwd=str(root),
                    stdout_limit=32 * 1024 * 1024, stderr_limit=2 * 1024 * 1024,
                )
                try:
                    payload = json.loads(completed.stdout or "")
                except json.JSONDecodeError:
                    return {
                        "status": "scanner_error", "tool": "pip-audit",
                        "reason": "scanner did not return valid JSON",
                        "scanner_version": scanner_version,
                        "returncode": completed.returncode,
                        "stderr": (completed.stderr or "")[-2000:],
                    }
                dependencies = payload.get("dependencies") if isinstance(payload, dict) else None
                if not isinstance(dependencies, list):
                    return {
                        "status": "scanner_error", "tool": "pip-audit",
                        "reason": "scanner JSON omitted the dependency inventory",
                        "scanner_version": scanner_version,
                    }
                expected = {
                    (re.sub(r"[-_.]+", "-", name).casefold(), version)
                    for name, version in manifest
                }
                reported: set[tuple[str, str]] = set()
                manifest_findings = 0
                skipped = False
                for dependency in dependencies:
                    if not isinstance(dependency, dict):
                        return {
                            "status": "scanner_error", "tool": "pip-audit",
                            "reason": "scanner returned a malformed dependency record",
                            "scanner_version": scanner_version,
                        }
                    name, version = dependency.get("name"), dependency.get("version")
                    if not isinstance(name, str) or not isinstance(version, str):
                        return {
                            "status": "scanner_error", "tool": "pip-audit",
                            "reason": "scanner returned a dependency without an identity",
                            "scanner_version": scanner_version,
                        }
                    identity = (re.sub(r"[-_.]+", "-", name).casefold(), version)
                    if identity in reported:
                        return {
                            "status": "scanner_error", "tool": "pip-audit",
                            "reason": "scanner returned a duplicate dependency record",
                            "scanner_version": scanner_version,
                        }
                    reported.add(identity)
                    skipped = skipped or bool(dependency.get("skip_reason"))
                    vulnerabilities = dependency.get("vulns")
                    if not isinstance(vulnerabilities, list):
                        return {
                            "status": "scanner_error", "tool": "pip-audit",
                            "reason": "scanner returned a malformed vulnerability inventory",
                            "scanner_version": scanner_version,
                        }
                    for vulnerability in vulnerabilities:
                        if (
                            not isinstance(vulnerability, dict)
                            or not isinstance(vulnerability.get("id"), str)
                            or not vulnerability["id"]
                            or not isinstance(vulnerability.get("aliases", []), list)
                        ):
                            return {
                                "status": "scanner_error", "tool": "pip-audit",
                                "reason": "scanner returned a malformed vulnerability record",
                                "scanner_version": scanner_version,
                            }
                        manifest_findings += 1
                        findings.append({
                            "package": name, "version": version,
                            "id": vulnerability.get("id"),
                            "aliases": vulnerability.get("aliases", []),
                        })
                if reported != expected or skipped:
                    return {
                        "status": "scanner_error", "tool": "pip-audit",
                        "reason": "scanner did not audit every exact locked package/version",
                        "scanner_version": scanner_version,
                        "expected_packages": len(expected),
                        "reported_packages": len(reported),
                    }
                allowed_returncodes = {0, 1} if manifest_findings else {0}
                if completed.returncode not in allowed_returncodes:
                    return {
                        "status": "scanner_error", "tool": "pip-audit",
                        "reason": "scanner exited unsuccessfully without vulnerability findings",
                        "scanner_version": scanner_version,
                        "returncode": completed.returncode,
                        "stderr": (completed.stderr or "")[-2000:],
                    }
                audited_packages.update(reported)
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "status": "scanner_error", "tool": "pip-audit",
            "reason": f"scanner execution failed ({type(exc).__name__})",
            "scanner_version": scanner_version,
        }

    return {
        "status": "findings" if findings else "pass",
        "tool": "pip-audit", "scanner_version": scanner_version,
        "input": "uv.lock exact registry pins",
        "uv_lock_sha256": binding["uv_lock_sha256"],
        "release_binding_sha256": binding["binding_sha256"],
        "wheel_sha256": binding["wheel"]["sha256"],
        "sdist_sha256": binding["sdist"]["sha256"],
        "manifests_audited": len(manifests),
        "packages_expected": len(packages),
        "packages_audited": len(audited_packages),
        "vulnerabilities": len(findings),
        "highest_severity": "none" if not findings else "unknown",
        "findings": findings,
    }


def run_bandit_static_scan(project_root: Path, *, timeout: int = 300) -> dict[str, Any]:
    executable = _find_tool("bandit")
    if not executable:
        return {"status": "unavailable", "tool": "bandit", "reason": "bandit is not installed"}
    try:
        version_result = run_bounded_capture(
            [executable, "--version"], timeout=min(timeout, 30), check=False,
            stdout_limit=64 * 1024, stderr_limit=64 * 1024,
        )
        scanner_version = (version_result.stdout or version_result.stderr or "").strip()[:500]
        if version_result.returncode != 0 or not scanner_version:
            return {
                "status": "scanner_error", "tool": "bandit",
                "reason": "scanner version could not be established",
            }
        root = project_root.resolve()
        scan_targets = [
            str(root / name) for name in ("src", "scripts", "packaging", "siftbench")
            if (root / name).exists()
        ]
        excluded_targets = [
            str(root.joinpath(*prefix))
            for prefix in _DEVELOPER_SECURITY_SCAN_EXCLUDED_PREFIXES
            if root.joinpath(*prefix).exists()
        ]
        command = [executable, "-r", *scan_targets, "-f", "json", "-q"]
        if excluded_targets:
            # Bandit accepts one comma-delimited exclusion argument.  The
            # excluded runtime contains third-party dependencies, not Sift
            # source; those exact packages are covered by uv.lock, the SBOM,
            # and the separately authorized vulnerability scan.
            command.extend(("-x", ",".join(excluded_targets)))
        completed = run_bounded_capture(
            command,
            timeout=timeout,
            check=False,
            stdout_limit=32 * 1024 * 1024,
        )
        payload = json.loads(completed.stdout or "{}")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return {"status": "scanner_error", "tool": "bandit", "reason": str(exc)}
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list) or completed.returncode not in {0, 1}:
        return {
            "status": "scanner_error", "tool": "bandit",
            "scanner_version": scanner_version,
            "reason": "scanner failed or omitted its result inventory",
            "returncode": completed.returncode,
            "stderr": (completed.stderr or "")[-2000:],
        }
    counts = {"low": 0, "medium": 0, "high": 0}
    reviewed: list[dict[str, Any]] = []
    for row in results:
        if not isinstance(row, dict):
            return {
                "status": "scanner_error", "tool": "bandit",
                "scanner_version": scanner_version,
                "reason": "scanner returned a malformed finding record",
            }
        severity_value = row.get("issue_severity")
        if not isinstance(severity_value, str) or severity_value.casefold() not in counts:
            return {
                "status": "scanner_error", "tool": "bandit",
                "scanner_version": scanner_version,
                "reason": "scanner returned an invalid finding severity",
            }
        severity = severity_value.casefold()
        counts[severity] += 1
        if severity in {"medium", "high"}:
            filename = row.get("filename")
            if not isinstance(filename, str) or not filename:
                return {
                    "status": "scanner_error", "tool": "bandit",
                    "scanner_version": scanner_version,
                    "reason": "scanner finding omitted its source path",
                }
            reported_path = Path(filename)
            resolved_path = (
                reported_path.resolve()
                if reported_path.is_absolute()
                else (root / reported_path).resolve()
            )
            try:
                relative_path = resolved_path.relative_to(root)
            except ValueError:
                return {
                    "status": "scanner_error", "tool": "bandit",
                    "scanner_version": scanner_version,
                    "reason": "scanner finding referenced a path outside the project",
                }
            reviewed.append(
                {
                    "severity": severity,
                    "test_id": row.get("test_id"),
                    "path": relative_path.as_posix(),
                    "line": row.get("line_number"),
                    "description": row.get("issue_text"),
                }
            )
    return {
        "status": "pass" if counts["medium"] == counts["high"] == 0 else "findings",
        "tool": "bandit", "scanner_version": scanner_version,
        "returncode": completed.returncode,
        "counts": counts,
        "medium_high_findings": reviewed,
        "gate": "high severity findings must be zero; medium findings require explicit review",
    }


def security_qualification_report(
    project_root: Path, *, run_external: bool = False,
    run_dependency_scan: bool | None = None,
    run_static_scan: bool | None = None,
    pentest_attestation: Path | None = None,
    pentest_trust_store: Path | None = None,
    pentest_approved_key_id: str | None = None,
) -> dict[str, Any]:
    """Run local, non-destructive qualification checks and state blockers."""
    root = Path(project_root).resolve()
    dependency_enabled = (
        run_external if run_dependency_scan is None else run_dependency_scan
    )
    static_enabled = run_external if run_static_scan is None else run_static_scan
    release_binding = security_release_binding(root)
    sdist_path = None
    # Never let a stale archive determine the current tree's scan result. A
    # release-ready binding proves the sdist is the current source; otherwise
    # scan the live, bounded production surfaces and report the binding blocker.
    if (
        release_binding.get("status") == "ready"
        and isinstance(release_binding.get("sdist"), dict)
    ):
        relative_sdist = release_binding["sdist"].get("path")
        if isinstance(relative_sdist, str):
            sdist_path = root / relative_sdist
    secrets = scan_source_secrets(root, sdist_path=sdist_path)
    static = scan_python_static_security(root, sdist_path=sdist_path)
    dependency = (
        run_dependency_vulnerability_scan(root)
        if dependency_enabled
        else {
            "status": "not_run",
            "reason": "public advisory lookup was not explicitly authorized",
            "input": "uv.lock exact registry pins",
            "release_binding_sha256": release_binding.get("binding_sha256"),
        }
    )
    bandit = (
        run_bandit_static_scan(root)
        if static_enabled
        else {"status": "not_run", "reason": "external scanner execution was not requested"}
    )
    high = [row for row in (*secrets, *static) if row.get("severity") in {"high", "critical"}]
    blockers: list[str] = []
    if release_binding.get("status") != "ready":
        blockers.append("release_artifact_binding_not_ready")
    if high:
        blockers.append("local_high_or_critical_findings")
    if dependency.get("status") != "pass":
        blockers.append("dependency_scan_not_clear")
    if bandit.get("status") != "pass":
        blockers.append("independent_static_scan_not_clear")
    from sift.pentest_assurance import verify_pentest_attestation

    pentest = verify_pentest_attestation(
        project_root,
        attestation_path=pentest_attestation,
        trust_store_path=pentest_trust_store,
        approved_key_id=pentest_approved_key_id,
    )
    if pentest.get("status") == "external_required":
        blockers.append("independent_third_party_penetration_test_not_supplied")
    elif pentest.get("status") == "findings_open":
        blockers.append("penetration_test_high_or_critical_findings_unresolved")
    elif pentest.get("status") != "pass":
        blockers.append("independent_third_party_penetration_test_invalid")
    qualification_binding: dict[str, Any] = {
        "release": release_binding,
        "scanners": {
            "dependency": {
                "tool": dependency.get("tool"),
                "version": dependency.get("scanner_version"),
                "status": dependency.get("status"),
            },
            "static": {
                "tool": bandit.get("tool"),
                "version": bandit.get("scanner_version"),
                "status": bandit.get("status"),
            },
        },
    }
    qualification_binding["binding_sha256"] = hashlib.sha256(
        _canonical_json(qualification_binding),
    ).hexdigest()
    blocking_secrets = [
        row for row in secrets if row.get("severity") in {"high", "critical"}
    ]
    blocking_static = [
        row for row in static if row.get("severity") in {"high", "critical"}
    ]
    return {
        "generated_at": _utc_now().isoformat(),
        "qualification_binding": qualification_binding,
        "secret_scan": {
            "status": "pass" if not blocking_secrets else "findings",
            "surface": "exact_sdist" if sdist_path is not None else "source_fallback",
            "findings": secrets,
        },
        "static_analysis": {
            "status": "pass" if not blocking_static else "findings",
            "surface": "exact_sdist" if sdist_path is not None else "source_fallback",
            "findings": static,
        },
        "bandit_static_analysis": bandit,
        "dependency_scan": dependency,
        "independent_penetration_test": pentest,
        "confidential_production_ready": not blockers,
        "blockers": blockers,
    }


def write_security_qualification_report(
    project_root: Path, output_path: Path, *, run_external: bool = False,
    run_dependency_scan: bool | None = None,
    run_static_scan: bool | None = None,
    pentest_attestation: Path | None = None,
    pentest_trust_store: Path | None = None,
    pentest_approved_key_id: str | None = None,
) -> dict[str, Any]:
    report = security_qualification_report(
        project_root, run_external=run_external,
        run_dependency_scan=run_dependency_scan,
        run_static_scan=run_static_scan,
        pentest_attestation=pentest_attestation,
        pentest_trust_store=pentest_trust_store,
        pentest_approved_key_id=pentest_approved_key_id,
    )
    from sift.reliability import atomic_write_json
    atomic_write_json(output_path, report)
    return report
