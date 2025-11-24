"""PII sanitization utilities for test data generation.

Provides tools to sanitize personally identifiable information (PII) from emails
while preserving the structure and characteristics needed for classification testing.
"""

import hashlib
import re
from typing import Any

from .state import Email


class PiiSanitizer:
    """Sanitize PII from emails while preserving structure."""

    def __init__(self):
        """Initialize sanitizer with mapping caches."""
        self.email_map: dict[str, str] = {}
        self.domain_map: dict[str, str] = {}
        self.phone_map: dict[str, str] = {}
        self.name_counter = 0
        self.person_map: dict[str, str] = {}

    def _deterministic_hash(self, text: str, prefix: str = "") -> str:
        """Create deterministic hash for consistent anonymization.

        Args:
            text: Text to hash
            prefix: Optional prefix for the hash

        Returns:
            Deterministic hash string
        """
        hash_obj = hashlib.sha256(text.encode())
        return prefix + hash_obj.hexdigest()[:8]

    def sanitize_email_address(self, email_addr: str) -> str:
        """Sanitize email address while preserving domain patterns.

        Args:
            email_addr: Email address to sanitize

        Returns:
            Sanitized email address
        """
        # Return cached if already seen
        if email_addr in self.email_map:
            return self.email_map[email_addr]

        # Extract local and domain parts
        if "@" not in email_addr:
            sanitized = f"user_{self._deterministic_hash(email_addr, 'u')}"
            self.email_map[email_addr] = sanitized
            return sanitized

        local, domain = email_addr.rsplit("@", 1)

        # Sanitize local part
        local_hash = self._deterministic_hash(local, "u")

        # Preserve common domains, sanitize others
        common_domains = {
            "gmail.com",
            "outlook.com",
            "hotmail.com",
            "yahoo.com",
            "icloud.com",
        }

        if domain.lower() in common_domains:
            sanitized_domain = domain.lower()
        elif domain in self.domain_map:
            sanitized_domain = self.domain_map[domain]
        else:
            domain_hash = self._deterministic_hash(domain, "d")
            sanitized_domain = f"domain_{domain_hash}.example"
            self.domain_map[domain] = sanitized_domain

        sanitized = f"{local_hash}@{sanitized_domain}"
        self.email_map[email_addr] = sanitized
        return sanitized

    def sanitize_phone_number(self, phone: str) -> str:
        """Sanitize phone numbers.

        Args:
            phone: Phone number to sanitize

        Returns:
            Sanitized phone number
        """
        if phone in self.phone_map:
            return self.phone_map[phone]

        sanitized = "[PHONE_REDACTED]"
        self.phone_map[phone] = sanitized
        return sanitized

    def sanitize_credit_card(self, card: str) -> str:
        """Sanitize credit card numbers.

        Args:
            card: Credit card number to sanitize

        Returns:
            Redacted string
        """
        return "[CC_REDACTED]"

    def sanitize_ssn(self, ssn: str) -> str:
        """Sanitize Social Security Numbers.

        Args:
            ssn: SSN to sanitize

        Returns:
            Redacted string
        """
        return "[SSN_REDACTED]"

    def sanitize_person_name(self, name: str) -> str:
        """Sanitize person names.

        Args:
            name: Person name to sanitize

        Returns:
            Generic person identifier
        """
        if name in self.person_map:
            return self.person_map[name]

        self.name_counter += 1
        sanitized = f"PERSON_{self.name_counter}"
        self.person_map[name] = sanitized
        return sanitized

    def sanitize_text(self, text: str) -> str:
        """Sanitize text content using pattern matching.

        Args:
            text: Text to sanitize

        Returns:
            Sanitized text
        """
        if not text:
            return text

        # Email addresses (more specific pattern)
        text = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            lambda m: self.sanitize_email_address(m.group(0)),
            text,
        )

        # Credit card numbers (13-19 digits, possibly with spaces/dashes)
        text = re.sub(
            r"\b(?:\d[ -]*?){13,19}\b",
            lambda m: self.sanitize_credit_card(m.group(0))
            if self._looks_like_cc(m.group(0))
            else m.group(0),
            text,
        )

        # SSN (XXX-XX-XXXX)
        text = re.sub(
            r"\b\d{3}-\d{2}-\d{4}\b",
            lambda m: self.sanitize_ssn(m.group(0)),
            text,
        )

        # Phone numbers (various formats)
        text = re.sub(
            r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            lambda m: self.sanitize_phone_number(m.group(0)),
            text,
        )

        # IP addresses (keep structure but sanitize)
        text = re.sub(
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "[IP_REDACTED]",
            text,
        )

        # URLs with paths that might contain IDs (preserve domain patterns)
        text = re.sub(
            r"(https?://[^/\s]+)/[^\s]*",
            r"\1/[PATH_REDACTED]",
            text,
        )

        return text

    def _looks_like_cc(self, number_str: str) -> bool:
        """Check if a number string looks like a credit card using Luhn algorithm.

        Args:
            number_str: Potential credit card number

        Returns:
            True if passes Luhn check
        """
        # Remove spaces and dashes
        digits = re.sub(r"[- ]", "", number_str)

        # Must be all digits and correct length
        if not digits.isdigit() or len(digits) < 13 or len(digits) > 19:
            return False

        # Luhn algorithm
        total = 0
        reverse_digits = digits[::-1]

        for i, digit in enumerate(reverse_digits):
            n = int(digit)
            if i % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            total += n

        return total % 10 == 0

    def sanitize_email(self, email: Email) -> Email:
        """Sanitize an entire email object.

        Args:
            email: Email to sanitize

        Returns:
            New Email object with sanitized content
        """
        return Email(
            uid=email.uid,
            subject=self.sanitize_text(email.subject),
            sender=self.sanitize_email_address(email.sender),
            body=self.sanitize_text(email.body),
            date=email.date,
            attachments=[self.sanitize_text(att) for att in email.attachments],
        )


def email_to_dict(email: Email) -> dict[str, Any]:
    """Convert Email object to dictionary for JSON serialization.

    Args:
        email: Email object

    Returns:
        Dictionary representation
    """
    return {
        "uid": email.uid,
        "subject": email.subject,
        "sender": email.sender,
        "body": email.body,
        "date": email.date.isoformat(),
        "attachments": email.attachments,
    }


def dict_to_email(data: dict[str, Any]) -> Email:
    """Convert dictionary to Email object.

    Args:
        data: Dictionary with email data

    Returns:
        Email object
    """
    from datetime import datetime

    return Email(
        uid=data["uid"],
        subject=data["subject"],
        sender=data["sender"],
        body=data["body"],
        date=datetime.fromisoformat(data["date"]),
        attachments=data["attachments"],
    )
