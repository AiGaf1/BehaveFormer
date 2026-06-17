"""Keystroke-timing capture — turns raw press/release events into the parallel
arrays the model and bank I/O consume.

UI-agnostic on purpose: the view feeds it primitive `(keycode, key_name, time)`
values (extracted from tkinter events), so the timing logic can be unit-tested
without a display. Keys are paired by HARDWARE keycode (identical for the down
and up of one physical key) — pairing by character breaks under Shift, dead keys
and IME, leaving stale entries that later produce absurd multi-second holds.
"""

from __future__ import annotations

import numpy as np

from app.keymap import key_code


class KeystrokeRecorder:
    """Accumulates a typing session as parallel press/release/key lists (ms)."""

    def __init__(self) -> None:
        self._pressed_at: dict[int, float] = {}
        self._last_release_at: float | None = None
        # Release-ordered logs (ms / codes / names), one entry per keystroke.
        self.press_ms:   list[float] = []
        self.release_ms: list[float] = []
        self.key_ids:    list[int]   = []
        self.key_names:  list[str]   = []

    def __len__(self) -> int:
        return len(self.key_ids)

    def on_press(self, keycode: int, t: float) -> None:
        """Record a key-down at time `t` (seconds). Ignores auto-repeat presses."""
        if keycode not in self._pressed_at:
            self._pressed_at[keycode] = t

    def on_release(self, keycode: int, key_name: str, t: float) -> tuple[float, float] | None:
        """Record a key-up at time `t` (seconds); log the keystroke.

        Returns (hold_ms, flight_ms) for display, or None if the matching press
        was never seen. Here `flight_ms` is the gap BEFORE this key (since the
        previous release) — the live-feedback convention, not the model's.
        """
        start = self._pressed_at.pop(keycode, None)
        if start is None:
            return None

        hold_ms = (t - start) * 1000.0
        flight_ms = (0.0 if self._last_release_at is None
                     else max(0.0, (start - self._last_release_at) * 1000.0))
        self._last_release_at = t

        self.press_ms.append(start * 1000.0)
        self.release_ms.append(t * 1000.0)
        self.key_ids.append(key_code(key_name))
        self.key_names.append(key_name)
        return hold_ms, flight_ms

    def session_hold_flight_ms(self) -> tuple[list[float], list[float]]:
        """Per-keystroke (hold_ms, flight_ms) in the MODEL convention for the CSV:
        hold = release-press, flight = next_press - this_release, last flight = 0.
        Reloading the CSV via `app.engine.holdflight_to_features` reproduces the
        exact features built from this session."""
        press = np.asarray(self.press_ms, dtype=np.float64)
        release = np.asarray(self.release_ms, dtype=np.float64)
        hold = release - press
        flight = np.zeros(len(press))
        flight[:-1] = press[1:] - release[:-1]
        return hold.tolist(), flight.tolist()

    def clear(self) -> None:
        """Reset to an empty session."""
        self._pressed_at.clear()
        self._last_release_at = None
        self.press_ms.clear()
        self.release_ms.clear()
        self.key_ids.clear()
        self.key_names.clear()
