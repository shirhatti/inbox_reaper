"""Text helpers for turning raw email bodies into compact plain text."""

import html
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlparse

# Data URLs for inline images: data:image/[type];base64,[base64data]
_BASE64_IMAGE_RE = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+")
_TAG_RE = re.compile(r"<[^>]+>")
_HSPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n\s*\n\s*\n+")
_URL_RE = re.compile(r"https?://[^\s<>\"'\])]+")
# Zero-width and spacer characters marketing emails use as preheader padding
_INVISIBLE_RE = re.compile(
    "[\u00ad\u034f\u061c\u115f\u1160\u17b4\u17b5\u180e\u200b-\u200f\u2060-\u2064\ufeff]"
)
# CSS that leaked into text: plain parts generated from HTML keep the <style>
# contents. Only blocks of "property: value;" count, so code in text survives.
_CSS_AT_RULE_RE = re.compile(
    r"@(media|font-face|supports|page|keyframes|-[a-z-]+)[^{};]*"
    r"\{(?:[^{}]|\{[^{}]*\})*\}|@(import|charset)[^;{}]*;",
    re.IGNORECASE,
)
# A selector made of CSS syntax only (lowercase tags, .class, #id, [attr],
# :pseudo), so prose before a rule ("Parking Receipt body{...}") is kept
_CSS_COMPOUND = (
    r"(?=[a-z*.#\[:])(?:[a-z][a-z0-9-]*|\*)?"
    r"(?:[.#][\w-]+|\[[^\]\n]*\]|::?[\w-]+(?:\([^)\n]*\))?)*"
)
_CSS_RULE_RE = re.compile(
    rf"(?<![\w.#\[:-]){_CSS_COMPOUND}"
    rf"(?:\s*[>+~,]\s*{_CSS_COMPOUND}|[ \t]+{_CSS_COMPOUND})*"
    r"\s*\{\s*(?:[-a-zA-Z]+\s*:[^;{}]*;\s*)*[-a-zA-Z]+\s*:[^;{}]*;?\s*\}"
)
# A MIME header block at the start of a part's text (some senders nest one)
_MIME_PREAMBLE_RE = re.compile(
    r"\A\s*(?:(?:content-[\w-]+|mime-version):[^\n]*\n(?:[ \t][^\n]*\n)*)+\r?\n",
    re.IGNORECASE,
)
_SAFELINKS_HOST = "safelinks.protection.outlook.com"
# Upper bound on links kept per email (feature extraction only needs a sample)
MAX_LINKS = 100


def real_url(url: str) -> str:
    """Undo Outlook Safe Links wrapping to recover the destination URL."""
    parsed = urlparse(url)
    if parsed.netloc.endswith(_SAFELINKS_HOST):
        inner = parse_qs(parsed.query).get("url")
        if inner and inner[0]:
            return inner[0]
    return url


def url_domain(url: str) -> str:
    """Host of a URL, lowercased and without a leading "www."."""
    host = urlparse(real_url(url)).netloc.lower().rsplit("@", 1)[-1].split(":")[0]
    return host.removeprefix("www.")


def compact_urls(text: str) -> str:
    """Replace each URL with ``[link: domain]``.

    Tracking and Safe Links URLs are hundreds of characters of noise to a
    model; the destination domain carries the signal.
    """
    return _URL_RE.sub(lambda m: f"[link: {url_domain(m.group(0)) or 'url'}]", text)


def find_urls(text: str) -> list[str]:
    """Destination URLs mentioned in plain text (Safe Links unwrapped)."""
    return [real_url(u) for u in _URL_RE.findall(text)][:MAX_LINKS]


@dataclass
class HTMLStructure:
    """Structural facts about an HTML body, for feature extraction."""

    links: list[str] = field(default_factory=list)
    images: int = 0
    tracking_pixels: int = 0


class _StructureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.result = HTMLStructure()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "a":
            href = a.get("href", "")
            if (
                href.startswith(("http://", "https://"))
                and len(self.result.links) < MAX_LINKS
            ):
                self.result.links.append(real_url(href))
        elif tag == "img":
            self.result.images += 1
            style = a.get("style", "").replace(" ", "").lower()
            tiny = all(
                a.get(dim, "").strip().rstrip("px") in {"0", "1", "2"}
                for dim in ("width", "height")
            )
            if tiny or "display:none" in style:
                self.result.tracking_pixels += 1


def html_structure(markup: str) -> HTMLStructure:
    """Links, image count and tracking pixels in an HTML document."""
    parser = _StructureParser()
    try:
        parser.feed(markup)
        parser.close()
    except Exception:  # malformed HTML: keep whatever was collected
        pass
    return parser.result


class HTMLTextExtractor(HTMLParser):
    """Extract plain text from HTML, ignoring tags and scripts."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_content = False

    def handle_starttag(self, tag, attrs):
        # Skip content in script and style tags
        if tag.lower() in ("script", "style"):
            self.skip_content = True

    def handle_endtag(self, tag):
        if tag.lower() in ("script", "style"):
            self.skip_content = False

    def handle_data(self, data):
        if not self.skip_content:
            self.text_parts.append(data)

    def get_text(self) -> str:
        return "".join(self.text_parts)


def remove_base64_images(text: str) -> str:
    """Remove base64 encoded images from text.

    Removes data URLs containing base64 encoded images which can appear in
    HTML emails as inline images. These bloat the text and provide no value
    for classification.

    Args:
        text: The text that may contain base64 encoded images

    Returns:
        Text with base64 image data removed
    """
    # Match data URLs for images: data:image/[type];base64,[base64data]
    # The base64 data consists of A-Za-z0-9+/= characters
    # We match until we hit a quote, whitespace, or < (end of attribute value)
    return _BASE64_IMAGE_RE.sub("", text)


def strip_css(text: str) -> str:
    """Remove CSS rules and @-blocks (e.g. ``@media``) that leaked into text."""
    text = _CSS_AT_RULE_RE.sub("", text)
    while (stripped := _CSS_RULE_RE.sub("", text)) != text:
        text = stripped
    return text


def strip_mime_preamble(text: str) -> str:
    """Drop a MIME header block that a sender left at the start of a part."""
    return _MIME_PREAMBLE_RE.sub("", text.replace("\r\n", "\n"), count=1)


def strip_html(text: str) -> str:
    """Strip HTML tags and extract plain text content.

    Removes HTML tags, scripts, styles, base64 images, and converts HTML
    entities to text. Cleans up excessive whitespace while preserving
    paragraph breaks.

    Args:
        text: The HTML or plain text to process

    Returns:
        Plain text with HTML removed and whitespace normalized
    """
    # Remove base64 encoded images first (they bloat the text)
    text = _INVISIBLE_RE.sub("", remove_base64_images(text))

    # If text doesn't contain HTML tags, only drop leaked CSS
    if "<" not in text:
        return strip_css(text)

    # Parse HTML and extract text
    extractor = HTMLTextExtractor()
    try:
        extractor.feed(text)
        plain_text = extractor.get_text()
    except Exception:
        # If HTML parsing fails, fall back to regex stripping
        plain_text = _TAG_RE.sub("", text)

    # Decode HTML entities (e.g., &nbsp; -> space, &lt; -> <)
    plain_text = html.unescape(plain_text)

    # Normalize whitespace: replace multiple spaces/tabs with single space
    plain_text = _HSPACE_RE.sub(" ", strip_css(plain_text))

    # Preserve paragraph breaks but remove excessive newlines (more than 2)
    plain_text = _BLANK_LINES_RE.sub("\n\n", plain_text)

    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in plain_text.split("\n")]
    plain_text = "\n".join(lines)

    return plain_text.strip()


def truncate_symmetric(text: str, max_length: int = 2000) -> str:
    """Truncate text symmetrically, keeping start and end.

    Strips HTML markup first to extract only meaningful text content,
    then preserves the beginning and end of the text (which often contains
    important footer information like unsubscribe links), removing the
    middle portion if the text exceeds max_length.

    Args:
        text: The text to truncate (HTML or plain text)
        max_length: Maximum length of the result (default: 2000)

    Returns:
        Truncated plain text with start and end preserved
    """
    # Strip HTML and shorten URLs to their domain before measuring length
    clean_text = compact_urls(strip_html(text))

    if len(clean_text) <= max_length:
        return clean_text

    # Reserve space for the truncation marker
    marker = "\n\n[... content truncated ...]\n\n"
    available = max_length - len(marker)

    # Split available space evenly between start and end
    start_length = available // 2
    end_length = available - start_length

    start = clean_text[:start_length]
    end = clean_text[-end_length:]

    return start + marker + end
