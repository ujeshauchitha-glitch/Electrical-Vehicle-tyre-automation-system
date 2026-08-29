"""Puts src/ on sys.path so tests can `import evtyre` without an install step.

Kept deliberately dependency-light: no packaging/build config, no pytest -
tests run with the standard library's unittest (see the "run the tests"
section of CLAUDE.md's development notes / the implementation report).
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
