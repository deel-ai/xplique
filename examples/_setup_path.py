"""
Setup helper for examples to import xplique from source.

This module adds the parent directory to sys.path, allowing examples
to import xplique directly from the source code without requiring
installation.

Usage:
    Simply import this module at the top of any example script:
    
    import _setup_path  # noqa: F401
    import xplique
"""

import sys
from pathlib import Path

# Add parent directory (project root) to Python path
_parent_dir = Path(__file__).resolve().parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))
