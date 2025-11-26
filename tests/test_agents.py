"""Tests for agent functions, specifically text processing utilities."""

import pytest

from inbox_reaper.agents import remove_base64_images, strip_html


class TestRemoveBase64Images:
    """Tests for remove_base64_images function."""

    def test_removes_png_base64_image(self):
        """Test removal of PNG base64 encoded image."""
        # Small 1x1 transparent PNG in base64
        base64_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        text = f'<img src="data:image/png;base64,{base64_png}">'
        result = remove_base64_images(text)
        assert base64_png not in result
        assert 'data:image' not in result
        assert '<img src="">' == result

    def test_removes_jpeg_base64_image(self):
        """Test removal of JPEG base64 encoded image."""
        # Sample JPEG base64 data (shortened for test)
        base64_jpeg = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAA"
        text = f'<img src="data:image/jpeg;base64,{base64_jpeg}">'
        result = remove_base64_images(text)
        assert base64_jpeg not in result
        assert 'data:image' not in result

    def test_removes_gif_base64_image(self):
        """Test removal of GIF base64 encoded image."""
        # Small GIF base64 data
        base64_gif = "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
        text = f'<img src="data:image/gif;base64,{base64_gif}">'
        result = remove_base64_images(text)
        assert base64_gif not in result

    def test_removes_multiple_base64_images(self):
        """Test removal of multiple base64 images in same text."""
        base64_1 = "iVBORw0KGgoAAAANSUhEUg=="
        base64_2 = "R0lGODlhAQABAIAAA=="
        text = f'''
        <div>
            <img src="data:image/png;base64,{base64_1}">
            <p>Some text</p>
            <img src="data:image/gif;base64,{base64_2}">
        </div>
        '''
        result = remove_base64_images(text)
        assert base64_1 not in result
        assert base64_2 not in result
        assert "Some text" in result

    def test_preserves_text_without_base64(self):
        """Test that text without base64 images is preserved."""
        text = "Hello world! This is a normal email."
        result = remove_base64_images(text)
        assert result == text

    def test_preserves_regular_img_tags(self):
        """Test that regular img tags with URLs are preserved."""
        text = '<img src="https://example.com/image.png">'
        result = remove_base64_images(text)
        assert result == text

    def test_preserves_non_image_data_urls(self):
        """Test that non-image data URLs are preserved."""
        text = '<a href="data:text/plain;base64,SGVsbG8gV29ybGQ=">Download</a>'
        result = remove_base64_images(text)
        assert result == text

    def test_handles_webp_base64_image(self):
        """Test removal of WebP base64 encoded image."""
        base64_webp = "UklGRh4AAABXRUJQVlA4TBEAAAAvAAAAAAfQ//"
        text = f'<img src="data:image/webp;base64,{base64_webp}">'
        result = remove_base64_images(text)
        assert base64_webp not in result

    def test_handles_svg_xml_base64_image(self):
        """Test removal of SVG base64 encoded image."""
        base64_svg = "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg=="
        text = f'<img src="data:image/svg+xml;base64,{base64_svg}">'
        result = remove_base64_images(text)
        assert base64_svg not in result

    def test_handles_empty_string(self):
        """Test that empty string returns empty string."""
        result = remove_base64_images("")
        assert result == ""

    def test_handles_large_base64_image(self):
        """Test removal of large base64 encoded image."""
        # Simulate a large base64 string (10KB)
        large_base64 = "A" * 10000 + "=="
        text = f'<img src="data:image/png;base64,{large_base64}">'
        result = remove_base64_images(text)
        assert large_base64 not in result
        assert len(result) < len(text)

    def test_preserves_surrounding_content(self):
        """Test that content surrounding base64 images is preserved."""
        base64_img = "iVBORw0KGgoAAAANSUhEUg=="
        text = f'''
        <html>
        <body>
        <h1>Welcome!</h1>
        <img src="data:image/png;base64,{base64_img}">
        <p>Click <a href="https://example.com">here</a> to unsubscribe.</p>
        </body>
        </html>
        '''
        result = remove_base64_images(text)
        assert "Welcome!" in result
        assert "Click" in result
        assert "unsubscribe" in result
        assert "https://example.com" in result
        assert base64_img not in result


class TestStripHtmlWithBase64:
    """Tests for strip_html function with base64 image handling."""

    def test_strip_html_removes_base64_images(self):
        """Test that strip_html removes base64 images before processing."""
        base64_img = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        html = f'''
        <html>
        <body>
        <p>Hello World</p>
        <img src="data:image/png;base64,{base64_img}">
        <p>Goodbye</p>
        </body>
        </html>
        '''
        result = strip_html(html)
        assert base64_img not in result
        assert "Hello World" in result
        assert "Goodbye" in result

    def test_strip_html_cleans_text_with_multiple_base64_images(self):
        """Test strip_html with multiple base64 images."""
        html = '''
        <div>
            <p>Product 1</p>
            <img src="data:image/png;base64,ABC123==">
            <p>Product 2</p>
            <img src="data:image/jpeg;base64,XYZ789==">
            <p>Product 3</p>
        </div>
        '''
        result = strip_html(html)
        assert "data:image" not in result
        assert "base64" not in result
        assert "Product 1" in result
        assert "Product 2" in result
        assert "Product 3" in result

    def test_strip_html_handles_inline_base64_in_style(self):
        """Test strip_html with base64 in CSS background-image."""
        html = '''
        <div style="background-image: url(data:image/png;base64,ABC123==)">
            <p>Content here</p>
        </div>
        '''
        result = strip_html(html)
        assert "base64" not in result
        assert "Content here" in result

    def test_strip_html_preserves_text_only_email(self):
        """Test that text-only emails are handled correctly."""
        text = "This is a plain text email without any HTML or images."
        result = strip_html(text)
        assert result == text

    def test_strip_html_realistic_marketing_email(self):
        """Test with realistic marketing email containing base64 images."""
        html = '''
        <html>
        <head><title>Special Offer!</title></head>
        <body>
        <table>
            <tr>
                <td>
                    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==" alt="Logo">
                </td>
            </tr>
            <tr>
                <td>
                    <h1>50% OFF Everything!</h1>
                    <p>Don't miss our biggest sale of the year!</p>
                    <a href="https://example.com/sale">Shop Now</a>
                </td>
            </tr>
            <tr>
                <td>
                    <p>To unsubscribe, click <a href="https://example.com/unsubscribe">here</a></p>
                </td>
            </tr>
        </table>
        </body>
        </html>
        '''
        result = strip_html(html)
        # Base64 should be removed
        assert "iVBORw0KGgo" not in result
        # Important classification text should remain
        assert "50% OFF" in result
        assert "sale" in result.lower()
        assert "unsubscribe" in result.lower()
