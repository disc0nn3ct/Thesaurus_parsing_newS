"""Helpers for Telegram messages, without importing or starting a client."""


def split_message(text: str, limit: int = 3900) -> list[str]:
    """Split even a single long paragraph, counting astral symbols as two units.

    UTF-16 counting is conservative for Telegram's message length limit. Keep
    every character and prefer paragraph, line, or word boundaries when possible.
    """
    if limit < 2:
        raise ValueError("limit must be at least 2")
    chunks = []
    start = 0
    while start < len(text):
        end, units = start, 0
        while end < len(text):
            width = 2 if ord(text[end]) > 0xFFFF else 1
            if units + width > limit:
                break
            units += width
            end += 1
        if end < len(text):
            for separator in ("\n\n", "\n", " "):
                boundary = text.rfind(separator, start, end)
                if boundary > start:
                    end = boundary + len(separator)
                    break
        chunks.append(text[start:end])
        start = end
    return chunks
