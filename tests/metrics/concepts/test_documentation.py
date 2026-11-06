"""Executable concept-metric documentation examples."""

import re
from pathlib import Path


def test_documentation_examples():
    """Keep the published small NumPy examples executable together."""
    page = Path(__file__).resolve().parents[3] / "docs/api/concepts/metrics/api_metrics.md"
    namespace = {}
    for example in re.findall(r"```python\n(.*?)```", page.read_text(encoding="utf8"), re.S):
        exec(compile(example, str(page), "exec"), namespace)
