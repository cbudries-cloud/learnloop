"""Turn a file path or URL into (title, plain text)."""
import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst", ".org"}
BLOCK_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "blockquote", "pre", "dd", "dt", "figcaption"}
SKIP_TAGS = {"script", "style", "nav", "header", "footer", "aside", "noscript", "sup", "table", "form"}


class _Extractor(HTMLParser):
    """Keep readable prose blocks; drop chrome, scripts and tables."""

    def __init__(self):
        super().__init__()
        self.blocks, self.buf, self.skip, self.depth, self.title, self._in_title = [], [], 0, 0, "", False

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self.skip += 1
        elif tag in BLOCK_TAGS:
            self.depth += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS:
            self.skip = max(0, self.skip - 1)
        elif tag in BLOCK_TAGS:
            self.depth = max(0, self.depth - 1)
            if self.depth == 0:
                text = re.sub(r"\s+", " ", "".join(self.buf)).strip()
                if len(text) > 2:
                    self.blocks.append(text)
                self.buf = []
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif self.depth and not self.skip:
            self.buf.append(data)


def html_to_text(html: str) -> tuple[str, str]:
    ex = _Extractor()
    ex.feed(html)
    return ex.title.strip(), "\n\n".join(ex.blocks)


def load(source: str) -> tuple[str, str]:
    if re.match(r"https?://", source):
        req = urllib.request.Request(source, headers={"user-agent": "learnloop/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read().decode(r.headers.get_content_charset() or "utf-8", "replace")
        if "<html" in body[:2000].lower():
            title, text = html_to_text(body)
            return (title or source), text
        return source, body
    path = Path(source).expanduser()
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise SystemExit("PDF support needs pypdf: pip install pypdf") from None
        return path.stem, "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if path.suffix.lower() in {".html", ".htm"}:
        title, text = html_to_text(path.read_text(errors="replace"))
        return title or path.stem, text
    return path.stem, path.read_text(errors="replace")


def expand(sources: list[str]) -> list[str]:
    """Directories expand to the text/PDF/HTML files inside them."""
    out = []
    for s in sources:
        p = Path(s).expanduser()
        if p.is_dir():
            out += sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in TEXT_SUFFIXES | {".pdf", ".html", ".htm"})
        else:
            out.append(s)
    return out


def chunks(text: str, size: int = 14000) -> list[str]:
    """Split on paragraph boundaries into model-sized pieces."""
    paras, out, cur = text.split("\n\n"), [], ""
    for para in paras:
        if cur and len(cur) + len(para) > size:
            out.append(cur)
            cur = ""
        cur += para + "\n\n"
    if cur.strip():
        out.append(cur)
    return out
