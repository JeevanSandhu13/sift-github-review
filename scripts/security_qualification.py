"""Generate release security artifacts without modifying product data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sift.security_assurance import (
    generate_cyclonedx_sbom,
    write_security_qualification_report,
)
from sift.pentest_assurance import pentest_preflight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("dist/security"))
    parser.add_argument(
        "--run-external", action="store_true",
        help="run installed third-party scanners, including the public dependency advisory lookup",
    )
    parser.add_argument(
        "--run-static", action="store_true",
        help="run the installed local Bandit scanner without any network disclosure",
    )
    parser.add_argument(
        "--authorize-public-dependency-disclosure", action="store_true",
        help=(
            "explicitly acknowledge that exact dependency names and versions "
            "from uv.lock will be sent to the configured public advisory service"
        ),
    )
    parser.add_argument(
        "--pentest-preflight",
        action="store_true",
        help="print an artifact-bound external assessment request without writing qualification artifacts",
    )
    parser.add_argument(
        "--pentest-artifact",
        type=Path,
        help=(
            "project-relative canonical all-platform release manifest under "
            "dist, required with --pentest-preflight"
        ),
    )
    parser.add_argument("--pentest-attestation", type=Path)
    parser.add_argument(
        "--pentest-trust-store", type=Path,
        help=(
            "Absolute path to an administrator-controlled assessor trust store; "
            "defaults to SIFT_PENTEST_TRUST_STORE or the OS system location."
        ),
    )
    parser.add_argument(
        "--pentest-key-id",
        help="Approved trust-store key ID expected in the signed attestation.",
    )
    args = parser.parse_args()
    if args.run_external and not args.authorize_public_dependency_disclosure:
        parser.error(
            "--run-external requires --authorize-public-dependency-disclosure",
        )
    root = args.root.resolve()
    if args.pentest_preflight:
        if args.pentest_artifact is None:
            parser.error("--pentest-artifact is required with --pentest-preflight")
        try:
            request = pentest_preflight(root, artifact_path=args.pentest_artifact)
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps(request, indent=2))
        return 0
    output = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    sbom = generate_cyclonedx_sbom(root, output / "sift.cdx.json")
    report = write_security_qualification_report(
        root,
        output / "qualification.json",
        run_external=args.run_external,
        run_static_scan=args.run_static or args.run_external,
        pentest_attestation=args.pentest_attestation,
        pentest_trust_store=args.pentest_trust_store,
        pentest_approved_key_id=args.pentest_key_id,
    )
    print(
        json.dumps(
            {
                "sbom": sbom,
                "confidential_production_ready": report["confidential_production_ready"],
                "blockers": report["blockers"],
            },
            indent=2,
        )
    )
    # This is a release qualification command: any unresolved local or
    # external gate must produce a failing process status. The JSON report
    # distinguishes missing external evidence from a discovered vulnerability.
    return 0 if report["confidential_production_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
