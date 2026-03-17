"""
Symbol Descriptions - Human-readable descriptions of symbolic trace representations

Purpose: Convert symbolic action representations (e.g. CLICK_BID_SUCCESS__R_EXPLORE)
into plain-English descriptions for interpretability output and failure explanations.

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

SYMBOL_COMPONENT_MAP: dict[str, str] = {
    "CLICK": "clicked",
    "TYPE": "typed text",
    "SCROLL": "scrolled",
    "NAVIGATE": "navigated",
    "STOP": "sent stop signal",
    "UNKNOWN": "produced unparseable action",
    "BID": "element by BID",
    "NONE": "no selector",
    "SUCCESS": "action succeeded",
    "FAIL": "action failed",
    "R_EXPLORE": "exploring",
    "R_RETRY": "retrying",
    "R_STUCK": "appears stuck",
    "R_CONFUSED": "confused",
    "R_BACK": "backtracking",
}

REASONING_SEPARATOR = "__"


def describe_symbol(symbol: str) -> str:
    """Convert a symbolic representation to a human-readable description.

    Splits the symbol on the reasoning separator ``__``, maps each token
    through SYMBOL_COMPONENT_MAP, and composes a sentence.

    Args:
        symbol: A symbol string like ``CLICK_BID_SUCCESS__R_EXPLORE``.

    Returns:
        Human-readable description like
        ``Clicked element by BID (action succeeded, exploring)``.
    """
    parts = symbol.split(REASONING_SEPARATOR, maxsplit=1)
    action_part = parts[0]
    reasoning_part = parts[1] if len(parts) > 1 else ""

    action_tokens = action_part.split("_")
    descs = [SYMBOL_COMPONENT_MAP.get(t, t.lower()) for t in action_tokens]

    if len(descs) >= 3:
        verb = descs[0].capitalize()
        middle = " ".join(descs[1:-1])
        parens_items: list[str] = [descs[-1]]
    elif len(descs) == 2:
        verb = descs[0].capitalize()
        middle = ""
        parens_items = [descs[1]]
    else:
        verb = descs[0].capitalize() if descs else symbol
        middle = ""
        parens_items = []

    if reasoning_part:
        r_tokens = reasoning_part.split("_")
        i = 0
        while i < len(r_tokens):
            if r_tokens[i] == "R" and i + 1 < len(r_tokens):
                key = f"R_{r_tokens[i + 1]}"
                parens_items.append(SYMBOL_COMPONENT_MAP.get(key, key.lower()))
                i += 2
            else:
                parens_items.append(r_tokens[i].lower())
                i += 1

    phrase = verb
    if middle:
        phrase += f" {middle}"
    if parens_items:
        phrase += f" ({', '.join(parens_items)})"
    return phrase


def describe_pattern(pattern: list[str]) -> str:
    """Describe a full pattern sequence in plain English.

    Args:
        pattern: List of symbol strings forming a sequential pattern.

    Returns:
        Arrow-joined description of each symbol.
    """
    return " -> ".join(describe_symbol(s) for s in pattern)
