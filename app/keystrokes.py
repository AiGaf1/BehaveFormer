"""Tkinter keysym → JS keyCode mapping, matching the Aalto training data.

Aalto stores raw W3C `KeyboardEvent.keyCode` values: letters use the uppercase
VK table (A=65..Z=90), digits 48..57, special keys follow the same table.
Mapping live tkinter events to the *same* codes ensures the encoder sees the
embedding rows it was trained on — `hash()` or `ord()` would collide with
unrelated keys (e.g. `ord('.')=46` is the Delete key in JS).
"""

from __future__ import annotations

import tkinter as tk

# Named multi-char keysyms → JS keyCode.
_SPECIAL_KEY_CODES: dict[str, int] = {
    "BackSpace":   8,   "Tab":         9,
    "Return":      13,  "KP_Enter":    13,
    "Shift_L":     16,  "Shift_R":     16,
    "Control_L":   17,  "Control_R":   17,
    "Alt_L":       18,  "Alt_R":       18,
    "Pause":       19,  "Caps_Lock":   20,
    "Escape":      27,  "Space":       32,
    "Prior":       33,  "Next":        34,  # PageUp / PageDown
    "End":         35,  "Home":        36,
    "Left":        37,  "Up":          38,
    "Right":       39,  "Down":        40,
    "Print":       44,  "Insert":      45,  "Delete":      46,
    "Super_L":     91,  "Super_R":     92,  # Windows / Meta keys
    "F1":  112, "F2":  113, "F3":  114, "F4":  115,
    "F5":  116, "F6":  117, "F7":  118, "F8":  119,
    "F9":  120, "F10": 121, "F11": 122, "F12": 123,
    "Num_Lock":    144, "Scroll_Lock": 145,
    "semicolon":   186, "plus":        187, "comma":       188,
    "minus":       189, "period":      190, "slash":       191,
    "asciitilde":  192, "grave":       192,
    "bracketleft": 219, "backslash":   220, "bracketright": 221,
    "quoteright":  222, "apostrophe":  222,
}
# vk229 (IME process key) — the most common Aalto code (~48% of OOV).
_SPECIAL_KEY_OOV = 229

# ASCII punctuation → JS keyCode (physical key position, not the character).
# Typing '.' or '>' both come from the same physical key → JS keyCode 190.
_PUNCT_KEY_CODES: dict[str, int] = {
    ";": 186, ":": 186,
    "=": 187, "+": 187,
    ",": 188, "<": 188,
    "-": 189, "_": 189,
    ".": 190, ">": 190,
    "/": 191, "?": 191,
    "`": 192, "~": 192,
    "[": 219, "{": 219,
    "\\": 220, "|": 220,
    "]": 221, "}": 221,
    "'": 222, '"': 222,
    # Shifted digits → same VK as the digit
    "!": 49, "@": 50, "#": 51, "$": 52, "%": 53,
    "^": 54, "&": 55, "*": 56, "(": 57, ")": 48,
}

def event_key_name(event: tk.Event) -> str:
    if event.keysym == "space":
        return "Space"
    if len(event.char) == 1 and event.char.isprintable() and event.char != "\x00":
        return event.char
    return event.keysym


def key_code(key: str) -> int:
    """Map a tkinter keysym (or single character) to a JS keyCode.

    Letters: uppercased so 'a' and 'A' both → 65. Punctuation and named
    keysyms are looked up explicitly; unknown named keys → vk229 (IME).
    """
    if len(key) == 1:
        upper = key.upper()
        if upper in _PUNCT_KEY_CODES:
            return _PUNCT_KEY_CODES[upper]
        return ord(upper)
    return _SPECIAL_KEY_CODES.get(key, _SPECIAL_KEY_OOV)
