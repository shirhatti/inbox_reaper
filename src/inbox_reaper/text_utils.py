"""Text helpers for turning raw email bodies into compact plain text."""

import html
import re
from html.parser import HTMLParser

# Data URLs for inline images: data:image/[type];base64,[base64data]
_BASE64_IMAGE_RE = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+")
_TAG_RE = re.compile(r"<[^>]+>")
_HSPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n\s*\n\s*\n+")


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
    text = remove_base64_images(text)

    # If text doesn't contain HTML tags, return as-is
    if "<" not in text:
        return text

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
    plain_text = _HSPACE_RE.sub(" ", plain_text)

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
    # Strip HTML first to get clean text
    clean_text = strip_html(text)

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
