"""Makes the helper modules of the site tests importable by name.

The suite runs in importlib mode, which adds nothing to `sys.path`, so the
shared helpers `_results` (this directory) and `_color` (the parent) are put
there explicitly.
"""

from pathlib import Path
import sys

for directory in (Path(__file__).parent, Path(__file__).parent.parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
