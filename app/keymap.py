"""Live tkinter input → canonical key id.

A thin adapter over `data.keymap` (the single source of truth for the key
vocabulary). `event_key_name` turns a tkinter event into a character or a key
name; `key_code` maps that to the same dense id the encoder was trained on, so
live keystrokes land on the right key-embedding rows.
"""

from __future__ import annotations

import tkinter as tk

from data.keymap import char_to_id, name_to_id

# tkinter keysyms for non-character keys → the canonical name `data.keymap` knows.
# (Punctuation arrives as its character via event.char, so it isn't listed here.)
_KEYSYM_ALIASES: dict[str, str] = {
    "Space": "space", "BackSpace": "backspace", "Return": "enter", "KP_Enter": "enter",
    "Tab": "tab", "Shift_L": "shift", "Shift_R": "shift",
    "Control_L": "ctrl", "Control_R": "ctrl", "Alt_L": "alt", "Alt_R": "alt",
    "Caps_Lock": "caps", "Escape": "escape", "Prior": "pageup", "Next": "pagedown",
    "End": "end", "Home": "home", "Left": "left", "Up": "up", "Right": "right",
    "Down": "down", "Insert": "insert", "Delete": "delete",
    "Super_L": "super", "Super_R": "super",
}


def event_key_name(event: tk.Event) -> str:
    if event.keysym == "space":
        return "Space"
    if len(event.char) == 1 and event.char.isprintable() and event.char != "\x00":
        return event.char
    return event.keysym


def key_code(key: str) -> int:
    """Map a tkinter keysym (or single character) to its canonical dense key id."""
    if len(key) == 1:
        return char_to_id(key)
    return name_to_id(_KEYSYM_ALIASES.get(key, key))
