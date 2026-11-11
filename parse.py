import re

UNSURE = [
    "difficult to determine",
    "hard to determine",
    "not possible to determine",
    "cannot determine",
    "can't determine",
    "in the classical sense",
    "could be interpreted",
    "may or may not",
]


NO = [
    "does not contain",
    "doesn't contain",
    "no political propaganda",
    "no overt",
    "not political propaganda",
    "not appear to contain",
    "no evidence of",
]


YES = [
    "contains political propaganda",
    "contain political propaganda",
    "is political propaganda",
    "appears to contain",
    "considered propaganda",
    "considered political propaganda",
    "form of propaganda",
]


def parse_safety_response(text):
    text = str(text).lower().strip()
    match = re.match(r"\W*(yes|no)\b", text)

    if match:
        return match.group(1)
    else:
        if any(p in text for p in UNSURE):
            return "unclear"
        elif any(p in text for p in YES):
            return "yes"
        elif any(p in text for p in NO):
            return "no"
        else:
            return "unclear"
