"""Importing the domain engine must never touch firebase_admin.

Pins the Phase 0 lazy-import fix and the Phase 5 dissolution: core/domain/ is
pure math (no I/O imports at module scope), so `import core.domain` — whose
façade pulls in scenario, lifetime, and everything beneath them — is
credential-free.

Run in a subprocess so the check is immune to import order elsewhere in the
test session.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

CHECK = (
    "import sys; "
    "import core.domain; "
    "assert 'firebase_admin' not in sys.modules, 'firebase_admin was imported'; "
    "assert 'core.services.firebase' not in sys.modules, "
    "'core.services.firebase was imported'"
)


def test_importing_domain_does_not_touch_firebase():
    result = subprocess.run(
        [sys.executable, "-c", CHECK],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
