"""Importing the scenario engine must never touch firebase_admin.

Pins the Phase 0 lazy-import fix (retirement.py / forecast.py): the math
modules only import core.services.investments inside their edge functions,
so `import core.services.scenario` is credential-free. From Phase 2 on this
test extends to `import core.domain`.

Run in a subprocess so the check is immune to import order elsewhere in the
test session.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

CHECK = (
    "import sys; "
    "import core.services.scenario; "
    "assert 'firebase_admin' not in sys.modules, 'firebase_admin was imported'; "
    "assert 'core.services.firebase' not in sys.modules, "
    "'core.services.firebase was imported'"
)


def test_importing_scenario_does_not_touch_firebase():
    result = subprocess.run(
        [sys.executable, "-c", CHECK],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
