"""Keep concept-metric documentation examples independently executable."""

import re
from pathlib import Path

import pytest

DOCS_DIR = Path(__file__).resolve().parents[3] / "docs" / "api" / "concepts" / "metrics"
DOC_PAGES = sorted(DOCS_DIR.glob("*.md"))
EXAMPLES = [
    (page, idx, code)
    for page in DOC_PAGES
    for idx, code in enumerate(
        re.findall(r"^```python\n(.*?)^```", page.read_text(encoding="utf-8"), re.M | re.S)
    )
]


@pytest.mark.parametrize(
    ("page", "idx", "example"),
    EXAMPLES,
    ids=[f"{page.name}[{idx}]" for page, idx, _ in EXAMPLES],
)
def test_api_example_in_fresh_namespace(page, idx, example):
    assert "import numpy as np" in example
    assert "from xplique.metrics.concepts import" in example
    exec(compile(example, f"{page}#{idx}", "exec"), {})


def test_api_examples_present():
    assert {page.name for page in DOC_PAGES} == {
        "api_metrics.md",
        "reconstruction.md",
        "usage.md",
        "alignment.md",
        "geometry_coherence.md",
    }
    assert len(EXAMPLES) == 6
