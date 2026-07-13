"""Shared pytest fixtures for the characterization suite.

Hard rule: this file (and everything under tests/) must NEVER import
core.services.firebase, directly or transitively — that module initializes
firebase_admin at import time. The suite must run with no Firebase
credentials and no network access.

Goldens live in tests/goldens/ — generated once from current main, reviewed,
and committed. A refactor phase may never regenerate a golden.
"""

from pathlib import Path

import pytest

GOLDENS_DIR = Path(__file__).parent / "goldens"


@pytest.fixture
def goldens_dir() -> Path:
    return GOLDENS_DIR
