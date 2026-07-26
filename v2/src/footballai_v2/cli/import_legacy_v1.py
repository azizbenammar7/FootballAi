"""Safely import preserved V1 artifacts into an isolated V2 analysis run."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from footballai_v2.contracts.v1 import CodeReference, ContractValidationError
from footballai_v2.importers import LegacyImportError, LegacyV1Importer
from footballai_v2.storage import LocalAnalysisRunStore, RunAlreadyExistsError


def _local_code_reference(repository_root: Path) -> CodeReference:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LegacyImportError("Git provenance could not be determined") from exc
    return CodeReference(
        repository="https://github.com/azizbenammar7/FootballAi",
        revision=revision,
        dirty=dirty,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="read-only V1 artifact directory")
    parser.add_argument("--output-root", type=Path, required=True, help="caller-configured V2 run root")
    parser.add_argument("--run-id", help="optional UUID-v4 run ID")
    parser.add_argument("--verbose", action="store_true", help="show chained diagnostic details")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[4]
    try:
        store = LocalAnalysisRunStore(args.output_root)
        importer = LegacyV1Importer(store, _local_code_reference(repository_root))
        run = importer.import_directory(args.source, run_id=args.run_id)
    except (LegacyImportError, RunAlreadyExistsError, ContractValidationError, OSError) as exc:
        print(f"Legacy import failed: {exc}", file=sys.stderr)
        if args.verbose and exc.__cause__ is not None:
            print(f"Diagnostic: {type(exc.__cause__).__name__}: {exc.__cause__}", file=sys.stderr)
        return 2
    print(f"run_id={run.run_id}")
    print(f"status={run.status.value}")
    print(f"manifest={store.manifest_path(run.run_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
