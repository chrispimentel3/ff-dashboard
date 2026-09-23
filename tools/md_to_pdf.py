"""Render a Markdown file to PDF, styled like the dashboard.

    PYTHONPATH=. .venv/bin/python tools/md_to_pdf.py docs/expected-points.md [out.pdf]

Uses the Chromium that Playwright already installs for the Yahoo scraper, so there is no
new dependency — no pandoc, no LaTeX, no wkhtmltopdf. Markdown becomes HTML, Chromium
prints it, and the page furniture (margins, running footer, page numbers) comes from CSS
`@page` rather than from a PDF library.

Type matches mega/ui.py: Archivo for headings, Archivo Narrow for body.
"""
from __future__ import annotations

import sys
from pathlib import Path

import markdown

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=Archivo+Narrow:wght@400;600&family=JetBrains+Mono:wght@400&display=swap');

@page {
  size: A4;
  margin: 18mm 16mm 20mm 16mm;
  @bottom-center { content: counter(page); }
}

:root {
  --ink:      #12161c;
  --muted:    #5b6472;
  --rule:     #dfe3e8;
  --accent:   #0b3d91;
  --accent-2: #c8102e;
  --code-bg:  #f4f6f8;
}

* { box-sizing: border-box; }

body {
  font-family: 'Archivo Narrow', 'Helvetica Neue', Arial, sans-serif;
  font-size: 10.5pt;
  line-height: 1.52;
  color: var(--ink);
  margin: 0;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}

h1, h2, h3, h4 {
  font-family: 'Archivo', 'Helvetica Neue', Arial, sans-serif;
  font-weight: 700;
  line-height: 1.22;
  margin: 1.5em 0 0.5em;
  page-break-after: avoid;
}
h1 {
  font-size: 23pt; margin-top: 0; color: var(--accent);
  border-bottom: 2.5px solid var(--accent); padding-bottom: 0.32em;
  letter-spacing: -0.01em;
}
h2 {
  font-size: 14.5pt; margin-top: 1.7em;
  border-bottom: 1px solid var(--rule); padding-bottom: 0.22em;
}
h3 { font-size: 11.5pt; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }

p { margin: 0.55em 0; }
strong { font-weight: 600; }

a { color: var(--accent); text-decoration: none; border-bottom: 0.5px solid #b9c6dd; }

/* The lede paragraph under the title */
h1 + p { font-size: 11.5pt; color: var(--muted); }

blockquote {
  margin: 1em 0; padding: 0.7em 1em;
  background: #f7f9fc; border-left: 3px solid var(--accent);
  page-break-inside: avoid;
}
blockquote p { margin: 0.2em 0; }

table {
  border-collapse: collapse; width: 100%;
  margin: 0.9em 0; font-size: 9.3pt;
  page-break-inside: avoid;
}
th {
  font-family: 'Archivo', sans-serif; font-weight: 600; font-size: 8.4pt;
  text-transform: uppercase; letter-spacing: 0.045em;
  text-align: left; color: var(--muted);
  border-bottom: 1.5px solid var(--ink); padding: 0.45em 0.6em;
}
td { padding: 0.42em 0.6em; border-bottom: 0.5px solid var(--rule); vertical-align: top; }
tbody tr:nth-child(even) { background: #fafbfc; }

code {
  font-family: 'JetBrains Mono', Menlo, Consolas, monospace;
  font-size: 8.8pt; background: var(--code-bg);
  padding: 0.1em 0.34em; border-radius: 3px;
}
pre {
  background: var(--code-bg); border: 0.5px solid var(--rule); border-left: 3px solid var(--accent-2);
  padding: 0.75em 0.9em; border-radius: 3px; overflow-x: auto;
  page-break-inside: avoid; margin: 0.9em 0;
}
pre code { background: none; padding: 0; font-size: 8.5pt; line-height: 1.45; }

hr { border: none; border-top: 1px solid var(--rule); margin: 2em 0; }

ul, ol { margin: 0.55em 0; padding-left: 1.35em; }
li { margin: 0.22em 0; }
"""

HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>{css}</style></head><body>{body}</body></html>"""


def render(md_path: Path, pdf_path: Path) -> Path:
    text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists", "attr_list"]
    )
    title = next((l.lstrip("# ").strip() for l in text.splitlines() if l.startswith("# ")),
                 md_path.stem)
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        # set_content, not goto(file://...). A file:// origin cannot fetch cross-origin, so
        # the Google Fonts @import is silently dropped and the PDF quietly embeds Helvetica
        # instead — it renders, it just isn't the typeface you asked for.
        page.set_content(HTML.format(title=title, css=CSS, body=body), wait_until="networkidle")
        page.evaluate("document.fonts.ready")
        page.wait_for_timeout(400)
        page.emulate_media(media="print")
        page.pdf(
            path=str(pdf_path), format="A4", print_background=True,
            margin={"top": "18mm", "bottom": "20mm", "left": "16mm", "right": "16mm"},
            display_header_footer=True,
            header_template="<div></div>",
            footer_template=(
                '<div style="width:100%;font-family:Arial,sans-serif;font-size:7.5pt;'
                'color:#8a93a0;padding:0 16mm;display:flex;justify-content:space-between;">'
                f'<span>{title}</span>'
                '<span class="pageNumber"></span></div>'
            ),
        )
        browser.close()
    return pdf_path


def main(argv: list[str]) -> None:
    if not argv:
        sys.exit(__doc__)
    src = Path(argv[0])
    if not src.is_file():
        sys.exit(f"not found: {src}")
    out = Path(argv[1]) if len(argv) > 1 else src.with_suffix(".pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    render(src, out)
    print(f"{src}  ->  {out}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main(sys.argv[1:])
