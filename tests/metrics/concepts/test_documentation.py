"""Keep concept-metric documentation examples independently executable."""

import re
from pathlib import Path

import pytest

API_PAGE = (
    Path(__file__).resolve().parents[3] / "docs" / "api" / "concepts" / "metrics" / "api_metrics.md"
)
EXAMPLES = re.findall(r"^```python\n(.*?)^```", API_PAGE.read_text(encoding="utf-8"), re.M | re.S)


@pytest.mark.parametrize("example", EXAMPLES, ids=range(len(EXAMPLES)))
def test_api_example_in_fresh_namespace(example):
    assert "import numpy as np" in example
    assert "from xplique.metrics.concepts import" in example
    exec(compile(example, str(API_PAGE), "exec"), {})


def test_api_examples_present():
    assert len(EXAMPLES) == 4
