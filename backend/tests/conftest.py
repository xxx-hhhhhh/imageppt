from __future__ import annotations

import sys
import os
from pathlib import Path

os.environ["LOCAL_INPAINT_ENABLED"] = "false"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

