"""Check the built site's math markup and renderer wiring, not just build success.

Requires the docs dependency group. Browser typesetting is checked separately;
these tests do not fetch JavaScript or fonts from the network.
"""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("mkdocs", reason="requires the docs dependency group")

ROOT = Path(__file__).resolve().parents[1]


class PageMath(HTMLParser):
    def __init__(self, html: str) -> None:
        super().__init__()
        self.scripts: list[str] = []
        self.display = 0
        self.inline = 0
        self.feed(html)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "script" and attributes.get("src"):
            self.scripts.append(attributes["src"])
        if "arithmatex" in (attributes.get("class") or "").split():
            self.display += tag == "div"
            self.inline += tag == "span"


@pytest.fixture(scope="module")
def built_docs(tmp_path_factory) -> Path:
    destination = tmp_path_factory.mktemp("math-docs")
    result = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "--site-dir", str(destination)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return destination


def test_every_page_loads_mathjax_after_its_configuration(built_docs: Path) -> None:
    pages = list(built_docs.rglob("*.html"))
    assert pages
    assert (built_docs / "javascripts/mathjax.js").is_file()
    for path in pages:
        scripts = PageMath(path.read_text()).scripts
        configuration = [i for i, src in enumerate(scripts) if src.endswith("/mathjax.js")]
        renderer = [i for i, src in enumerate(scripts) if src.endswith("/tex-mml-chtml.js")]
        assert len(configuration) == len(renderer) == 1, path
        assert configuration[0] < renderer[0], path


@pytest.mark.parametrize("page", ["fixed_income", "differentiable_iv"])
def test_equations_survive_markdown_as_math_markup(built_docs: Path, page: str) -> None:
    html = (built_docs / page / "index.html").read_text()
    math = PageMath(html)
    assert math.display >= 2
    assert math.inline >= 1
    assert "$$" not in html
