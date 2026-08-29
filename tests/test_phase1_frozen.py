"""Phase 1 freeze guard.

Re-hashes every file under the frozen Phase 1 paths against the manifest
at ``phase1_frozen_hashes.json``.  Fails on:
1. Content modification (hash mismatch)
2. File removal (manifest entry missing from disk)
3. File addition (disk file not in manifest — caught via directory listing)
"""

from __future__ import annotations

import hashlib
import json
import os
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST_PATH = Path(__file__).resolve().parent / "phase1_frozen_hashes.json"

# Frozen directories (Phase 1 paths that must never be modified)
_FROZEN_DIRS: list[str] = [
    "src/evtyre/schema",
    "src/evtyre/config",
    "src/evtyre/ingest",
]

# Frozen individual files
_FROZEN_FILES: list[str] = [
    "legacy/ev_tyre_fusion.py",
]


def _load_manifest() -> dict:
    with open(_MANIFEST_PATH) as f:
        return json.load(f)


def _discover_python_files(directory: str) -> list[str]:
    """Walk a directory and return sorted .py file paths (relative to repo root)."""
    abs_dir = _REPO_ROOT / directory
    result: list[str] = []
    for root, dirs, files in os.walk(abs_dir):
        # Skip __pycache__
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fname in sorted(files):
            if fname.endswith(".py"):
                rel = os.path.relpath(os.path.join(root, fname), _REPO_ROOT)
                # Normalize to forward slashes for cross-platform manifest keys
                rel = rel.replace(os.sep, "/")
                result.append(rel)
    return sorted(result)


def _sha256_of_file(filepath: str) -> str:
    with open(_REPO_ROOT / filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class Phase1FrozenHashTests(unittest.TestCase):
    """Re-hash every frozen file and compare against the manifest."""

    def setUp(self) -> None:
        manifest = _load_manifest()
        self.hashes: dict[str, str] = manifest["hashes"]
        self.directories: dict[str, list[str]] = manifest["directories"]

    def test_all_manifest_files_exist_and_hash_matches(self) -> None:
        """Every file in the manifest must exist on disk with the same hash."""
        for filepath, expected_hash in self.hashes.items():
            with self.subTest(filepath=filepath):
                full = _REPO_ROOT / filepath
                self.assertTrue(
                    full.exists(),
                    f"Frozen file {filepath} exists in manifest but not on disk",
                )
                actual_hash = _sha256_of_file(filepath)
                self.assertEqual(
                    actual_hash,
                    expected_hash,
                    f"Frozen file {filepath} has been modified! "
                    f"Expected {expected_hash}, got {actual_hash}",
                )

    def test_no_frozen_files_added(self) -> None:
        """A new .py file added under a frozen directory must be caught.

        This is the directory-listing assertion: a file that was never in the
        manifest would pass a per-file hash check (it wouldn't be checked
        at all).  This test walks each frozen directory and asserts the file
        list matches the manifest.
        """
        for dir_path, expected_files in self.directories.items():
            with self.subTest(directory=dir_path):
                actual_files = _discover_python_files(dir_path)
                self.assertEqual(
                    actual_files,
                    expected_files,
                    f"File list under {dir_path} differs from manifest. "
                    f"Added or removed files detected:\n"
                    f"  expected: {expected_files}\n"
                    f"  actual:   {actual_files}",
                )

    def test_no_frozen_files_removed(self) -> None:
        """Every file in the manifest must still exist on disk."""
        for filepath in self.hashes:
            with self.subTest(filepath=filepath):
                full = _REPO_ROOT / filepath
                self.assertTrue(
                    full.exists(),
                    f"Frozen file {filepath} has been removed from disk",
                )


if __name__ == "__main__":
    unittest.main()
