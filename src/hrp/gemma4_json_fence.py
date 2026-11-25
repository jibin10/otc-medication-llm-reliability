"""Controlled Markdown fence normalization shared by Gemma 4 evaluations."""

import re

POLICY_VERSION = "gemma4_json_fence_v1"
_OPEN = re.compile(r" {0,3}(`{3,}|~{3,})[ \t]*json[ \t]*")
_CLOSE = re.compile(r" {0,3}(`{3,}|~{3,})[ \t]*")
_LINE_BREAKS = re.compile(r"(?<=\n)|(?<=\r)(?!\n)")
_JSON_WHITESPACE = " \t\r\n"


def normalize_json_fence(text: str) -> tuple[str, bool]:
    """Remove the two fence lines when they enclose the whole response."""
    lines = _LINE_BREAKS.split(text)
    start = next((i for i, line in enumerate(lines) if line.strip(_JSON_WHITESPACE)), None)
    if start is None:
        return text, False
    opener = _OPEN.fullmatch(lines[start].rstrip("\r\n"))
    if opener is None:
        return text, False

    marker = opener[1]
    for end in range(start + 1, len(lines)):
        closer = _CLOSE.fullmatch(lines[end].rstrip("\r\n"))
        if closer is None or closer[1][0] != marker[0] or len(closer[1]) < len(marker):
            continue
        if any(line.strip(_JSON_WHITESPACE) for line in lines[end + 1:]):
            return text, False
        return "".join(lines[:start] + lines[start + 1:end] + lines[end + 1:]), True
    return text, False
