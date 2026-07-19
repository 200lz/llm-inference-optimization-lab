#!/usr/bin/env python3
"""Generate deterministic, versioned serving workload JSONL plus a manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.serving_common import ValidationError, generate_records, load_json, sha256_file, write_workload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    try:
        if manifest_path.exists() and not args.force:
            raise FileExistsError(f"refusing to overwrite {manifest_path}; use --force")
        records, manifest = generate_records(load_json(args.config))
        write_workload(records, args.output, force=args.force)
        manifest["workload_sha256"] = sha256_file(args.output)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (ValidationError, FileExistsError, OSError) as error:
        parser.error(str(error))
    print(f"generated {len(records)} requests: {args.output}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
