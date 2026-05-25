"""Simple keystroke authentication interface.

The window is split into two main areas:
  - a top probability panel for the current authentication score
  - a bottom table for keystrokes with hold and flight times

This file is intentionally UI-only for now so the detector can be wired in later.
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Iterable

import numpy as np

# Allow running as a script (`python app/keyboard_auth.py`) as well as a module
# (`python -m app.keyboard_auth`) by ensuring the project root is importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from app.inference import (
    BANK_DIR,
    BANK_PATH,
    AuthEngine,
    build_bank_windows,
    calibrate_threshold,
    keystrokes_to_features,
    save_bank,
)

# Keystrokes needed to fill one detector window (must match the model's seq_len).
SEQ_LEN          = 25
# Bank size: how many enrollment windows to keep. Trained with K=5; bumping to 10
# at inference uses the same attention path but gives a more stable target representation.
BANK_K           = 10
# Multi-session enrollment: K windows are distributed across this many independent
# sessions (mirrors training's build_raw_banks(multi_session=True)).
ENROLL_SESSIONS  = 5
# Each session needs ≥ 2·seq_len keys so `_nonoverlap_starts` can pick two non-overlapping
# windows when BANK_K // ENROLL_SESSIONS = 2.
MIN_PER_SESSION  = 2 * SEQ_LEN


@dataclass(frozen=True)
class KeystrokeRow:
    key: str
    hold_time: float
    flight_time: float


class KeyboardAuthApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Keyboard Authentication")
        self.root.geometry("980x680")
        self.root.minsize(820, 560)
        self._pressed_at: dict[str, float] = {}
        self._last_release_at: float | None = None
        self._capture_started = False
        self._probability_history: deque[float] = deque(maxlen=120)

        # Raw keystroke log for the live stream: parallel lists of press/release
        # timestamps (ms) and key codes, in release order (matches feature build).
        self._press_ms:   list[float] = []
        self._release_ms: list[float] = []
        self._key_ids:    list[int]   = []
        # Capture mode: "idle" (typing does nothing), "auth" (live scoring vs the
        # loaded bank), or "enroll" (collecting keystrokes for a new bank).
        self._mode = "idle"
        # Impostor decision threshold (calibrated per-bank on Start auth; 0.5 fallback).
        self._threshold: float = 0.5
        # Multi-session enrollment: feature arrays for each completed session.
        self._enrollment_sessions: list[np.ndarray] = []

        self._configure_style()
        self._build_layout()
        self.set_probability(0.0)
        self.set_keystrokes([])
        self._update_counter()

        # Load the trained model and the saved user bank (if any). Done after the
        # UI exists so we can report status into the panel.
        self.engine: AuthEngine | None = None
        self._init_engine()

    def _configure_style(self) -> None:
        self.root.configure(bg="#0f172a")
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background="#0f172a")
        style.configure("Panel.TFrame", background="#111827", relief="flat")
        style.configure("Header.TLabel", background="#111827", foreground="#e5e7eb", font=("TkDefaultFont", 20, "bold"))
        style.configure("Subtle.TLabel", background="#111827", foreground="#9ca3af", font=("TkDefaultFont", 10))
        style.configure("Value.TLabel", background="#111827", foreground="#f9fafb", font=("TkDefaultFont", 28, "bold"))
        style.configure("Status.TLabel", background="#111827", foreground="#34d399", font=("TkDefaultFont", 13, "bold"))
        style.configure("Input.TEntry", fieldbackground="#0b1220", foreground="#f9fafb", insertcolor="#f9fafb")
        style.configure("TableTitle.TLabel", background="#0f172a", foreground="#e5e7eb", font=("TkDefaultFont", 14, "bold"))
        style.configure("App.Treeview", font=("TkDefaultFont", 11), rowheight=28)
        style.configure("App.Treeview.Heading", font=("TkDefaultFont", 11, "bold"))
        style.map("App.Treeview", background=[("selected", "#2563eb")])

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, style="App.TFrame", padding=18)
        container.pack(fill="both", expand=True)

        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)
        container.columnconfigure(0, weight=1)

        self.probability_panel = ttk.Frame(container, style="Panel.TFrame", padding=22)
        self.probability_panel.grid(row=0, column=0, sticky="nsew")
        self.probability_panel.columnconfigure(0, weight=1)

        title = ttk.Label(self.probability_panel, text="Authentication Probability", style="Header.TLabel")
        title.grid(row=0, column=0, sticky="w")

        subtitle = ttk.Label(
            self.probability_panel,
            text="Current impostor probability computed from the live keystroke stream.",
            style="Subtle.TLabel",
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(4, 18))

        self.probability_status = ttk.Label(self.probability_panel, text="Awaiting input", style="Status.TLabel")
        self.probability_status.grid(row=2, column=0, sticky="w", pady=(6, 0))

        self.probability_scores = ttk.Label(self.probability_panel, text="—", style="Subtle.TLabel")
        self.probability_scores.grid(row=3, column=0, sticky="w", pady=(4, 0))

        plot_frame = ttk.Frame(self.probability_panel, style="Panel.TFrame")
        plot_frame.grid(row=4, column=0, sticky="nsew", pady=(16, 0))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self.probability_figure = Figure(figsize=(7.2, 2.8), dpi=100, facecolor="#111827")
        self.probability_ax = self.probability_figure.add_subplot(111)
        self.probability_ax.set_facecolor("#0b1220")
        self.probability_ax.set_ylim(0.0, 1.0)
        self.probability_ax.set_xlim(0, 119)
        self.probability_ax.set_ylabel("Probability", color="#e5e7eb")
        self.probability_ax.set_xlabel("Latest samples", color="#e5e7eb")
        self.probability_ax.tick_params(colors="#cbd5e1")
        for spine in self.probability_ax.spines.values():
            spine.set_color("#475569")
        self.probability_ax.grid(True, color="#334155", alpha=0.35, linewidth=0.8)
        self.probability_line, = self.probability_ax.plot([], [], color="#38bdf8", linewidth=2.4)
        self.probability_fill = None

        self.probability_canvas = FigureCanvasTkAgg(self.probability_figure, master=plot_frame)
        canvas_widget = self.probability_canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, sticky="nsew")

        capture_row = ttk.Frame(self.probability_panel, style="Panel.TFrame")
        capture_row.grid(row=5, column=0, sticky="ew", pady=(18, 0))
        capture_row.columnconfigure(1, weight=1)

        capture_label = ttk.Label(capture_row, text="Type here:", style="Subtle.TLabel")
        capture_label.grid(row=0, column=0, sticky="w", padx=(0, 10))

        self.capture_var = tk.StringVar()
        self.capture_entry = ttk.Entry(capture_row, textvariable=self.capture_var, style="Input.TEntry")
        self.capture_entry.grid(row=0, column=1, sticky="ew")
        self.capture_entry.focus_set()
        self.capture_entry.bind("<KeyPress>", self._on_key_press)
        self.capture_entry.bind("<KeyRelease>", self._on_key_release)
        self.capture_entry.bind("<FocusIn>", self._start_capture)

        capture_hint = ttk.Label(
            self.probability_panel,
            text="Keystrokes are captured from this input field and added to the table below.",
            style="Subtle.TLabel",
        )
        capture_hint.grid(row=6, column=0, sticky="w", pady=(8, 0))

        controls = ttk.Frame(self.probability_panel, style="Panel.TFrame")
        controls.grid(row=7, column=0, sticky="ew", pady=(14, 0))

        self.auth_button = ttk.Button(controls, text="Start auth",
                                      command=self._start_auth)
        self.auth_button.grid(row=0, column=0, sticky="w")

        self.enroll_button = ttk.Button(controls, text="Save enrollment",
                                        command=self._save_enrollment)
        self.enroll_button.grid(row=0, column=2, sticky="w", padx=(10, 0))

        self.clear_button = ttk.Button(controls, text="Clear everything",
                                       command=self._reset_all)
        self.clear_button.grid(row=0, column=3, sticky="w", padx=(10, 0))

        self.engine_status = ttk.Label(self.probability_panel, text="", style="Subtle.TLabel")
        self.engine_status.grid(row=8, column=0, sticky="w", pady=(8, 0))

        style = ttk.Style(self.root)
        style.configure("Counter.TLabel", background="#111827", foreground="#fbbf24",
                         font=("TkDefaultFont", 13, "bold"))
        self.counter_label = ttk.Label(self.probability_panel, text="", style="Counter.TLabel")
        self.counter_label.grid(row=9, column=0, sticky="w", pady=(4, 0))

        style.configure("Threshold.TLabel", background="#111827", foreground="#818cf8",
                         font=("TkDefaultFont", 12, "bold"))
        self.threshold_label = ttk.Label(self.probability_panel, text="", style="Threshold.TLabel")
        self.threshold_label.grid(row=10, column=0, sticky="w", pady=(2, 0))

        table_section = ttk.Frame(container, style="App.TFrame")
        table_section.grid(row=1, column=0, sticky="nsew", pady=(18, 0))
        table_section.rowconfigure(1, weight=1)
        table_section.columnconfigure(0, weight=1)

        table_title = ttk.Label(table_section, text="Keystrokes", style="TableTitle.TLabel")
        table_title.grid(row=0, column=0, sticky="w", pady=(0, 10))

        table_panel = ttk.Frame(table_section, style="Panel.TFrame", padding=14)
        table_panel.grid(row=1, column=0, sticky="nsew")
        table_panel.rowconfigure(0, weight=1)
        table_panel.columnconfigure(0, weight=1)

        columns = ("key", "hold_time", "flight_time")
        self.tree = ttk.Treeview(table_panel, columns=columns, show="headings", style="App.Treeview", selectmode="browse")
        self.tree.heading("key", text="Key")
        self.tree.heading("hold_time", text="Hold Time (ms)")
        self.tree.heading("flight_time", text="Flight Time (ms)")
        self.tree.column("key", anchor="w", width=200, stretch=True)
        self.tree.column("hold_time", anchor="center", width=180, stretch=False)
        self.tree.column("flight_time", anchor="center", width=180, stretch=False)

        y_scroll = ttk.Scrollbar(table_panel, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=y_scroll.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")

        self.empty_label = ttk.Label(
            table_panel,
            text="No keystrokes loaded yet.",
            style="Subtle.TLabel",
        )
        self.empty_label.grid(row=0, column=0)
        self.empty_label.lift(self.tree)

    def set_probability(self, probability: float, threshold: float | None = None,
                        last_window: float | None = None) -> None:
        threshold = self._threshold if threshold is None else threshold
        probability = max(0.0, min(1.0, float(probability)))
        self._probability_history.append(probability)

        y_values = list(self._probability_history)
        x_values = list(range(len(y_values)))
        self.probability_line.set_data(x_values, y_values)
        self.probability_ax.set_xlim(0, max(1, len(y_values) - 1))
        self.probability_ax.figure.canvas.draw_idle()

        if probability >= threshold:
            status_text = "High impostor risk"
            status_color = "#f87171"
        else:
            status_text = "Authentication looks stable"
            status_color = "#34d399"

        self.probability_status.configure(text=status_text)
        ttk.Style(self.root).configure("Status.TLabel", foreground=status_color)

        # Show momentary (last window) alongside cumulative — last window is the raw
        # detector output for the most recent seq_len keys (no GRU history), while
        # cumulative is the GRU-integrated score over the whole stream.
        last_str = "—" if last_window is None else f"{last_window:.3f}"
        self.probability_scores.configure(
            text=f"Last window: {last_str}  ·  Cumulative: {probability:.3f}"
        )
    def set_keystrokes(self, rows: Iterable[KeystrokeRow | tuple[str, float, float]]) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

        count = 0
        for row in rows:
            if isinstance(row, KeystrokeRow):
                key, hold_time, flight_time = row.key, row.hold_time, row.flight_time
            else:
                key, hold_time, flight_time = row
            self.tree.insert("", "end", values=(key, f"{hold_time:.1f}", f"{flight_time:.1f}"))
            count += 1

        if count:
            self.empty_label.grid_remove()
        else:
            self.empty_label.grid()

    def _start_capture(self, _event: tk.Event) -> str | None:
        self._capture_started = True
        return None

    def _event_key_name(self, event: tk.Event) -> str:
        if event.keysym == "space":
            return "Space"
        if len(event.char) == 1 and event.char.isprintable() and event.char != "\x00":
            return event.char
        return event.keysym

    def _on_key_press(self, event: tk.Event) -> str | None:
        key = self._event_key_name(event)
        now = time.perf_counter()
        if key not in self._pressed_at:
            self._pressed_at[key] = now
        return None

    def _on_key_release(self, event: tk.Event) -> str | None:
        key = self._event_key_name(event)
        now = time.perf_counter()
        start = self._pressed_at.pop(key, None)
        if start is None:
            return None

        hold_time = (now - start) * 1000.0
        flight_time = 0.0 if self._last_release_at is None else max(0.0, (start - self._last_release_at) * 1000.0)
        self._last_release_at = now

        # Log raw keystroke for the model: press/release timestamps in ms + key code.
        # The key code is the character ordinal (or keysym hash for non-printables);
        # consistency between enroll and score is what matters, not the dataset's codes.
        self._press_ms.append(start * 1000.0)
        self._release_ms.append(now * 1000.0)
        self._key_ids.append(self._key_code(key))

        self.tree.insert("", "end", values=(key, f"{hold_time:.1f}", f"{flight_time:.1f}"))
        self.empty_label.grid_remove()

        self._update_live_score()
        self._update_counter()
        return None

    @staticmethod
    def _key_code(key: str) -> int:
        """Map a key name to an integer code matching the Aalto training data.

        Aalto was collected on Windows with VK codes: letters are uppercase
        VK_A=65..VK_Z=90, digits VK_0=48..VK_9=57, layout-independent.
        We mirror that by uppercasing single chars so 'a' and 'A' both → 65.
        Non-printable keys (shift, ctrl, …) get a stable hash in 128–230.
        """
        if len(key) == 1:
            return ord(key.upper())
        return hash(key) % 103 + 128  # non-printable: stable, outside letter range

    # ── model integration ────────────────────────────────────────────────────

    def _init_engine(self) -> None:
        """Load the trained model and any saved bank; report into the status label."""
        try:
            self.engine = AuthEngine(seq_len=SEQ_LEN)
        except Exception as exc:  # noqa: BLE001 — surface load failures in the UI
            self.engine = None
            self.engine_status.configure(text=f"Model failed to load: {exc}")
            return

        if self.engine.load_bank():
            self.engine_status.configure(
                text=f"Model ready · user loaded from {BANK_PATH.name} · click 'Start auth' to score")
        else:
            self.engine_status.configure(
                text="Model ready · no user loaded — 'Load user' or 'Save enrollment' to enrol one")

    def _current_features(self):
        """Build the model feature stream from the logged keystrokes, or None."""
        if not self._key_ids:
            return None
        return keystrokes_to_features(self._press_ms, self._release_ms, self._key_ids)

    def _update_counter(self) -> None:
        n = len(self._key_ids)
        if self._mode == "enroll":
            done = len(self._enrollment_sessions)
            if done >= ENROLL_SESSIONS:
                self.counter_label.configure(
                    text=f"All {ENROLL_SESSIONS} sessions captured — ready to finish"
                )
            else:
                self.counter_label.configure(
                    text=f"Session {done + 1} / {ENROLL_SESSIONS}  ·  "
                         f"Keys: {n} / {MIN_PER_SESSION}"
                )
        else:
            self.counter_label.configure(text=f"Keys typed: {n}")

    def _update_live_score(self) -> None:
        """Score the full stream and update the probability panel (auth mode only).

        Cumulative (plot, alarm): engine.score() runs the detector over all windows
        so the GRU integrates history → "is this session impostor".
        Last window (momentary): engine.score_latest() runs the detector on just the
        last seq_len keys with fresh GRU state → "is the latest burst suspicious".
        """
        if self.engine is None or self._mode != "auth" or not self.engine.has_bank:
            return
        feats = self._current_features()
        if feats is None:
            return
        scores = self.engine.score(feats)
        if len(scores) > 0:
            last = self.engine.score_latest(feats)
            self.set_probability(float(scores[-1]), last_window=last)

    def _start_auth(self) -> None:
        """Pick a bank file, load it, calibrate τ vs Aalto impostors, enter auth mode."""
        if self.engine is None:
            return
        path = filedialog.askopenfilename(
            title="Select enrollment bank",
            initialdir=str(BANK_PATH.parent),
            filetypes=[("Bank files", "*.npz"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.engine.load_bank(Path(path))
        except Exception as exc:  # noqa: BLE001
            self.engine_status.configure(text=f"Failed to load bank: {exc}")
            return

        # Calibrate the decision threshold against this specific bank.
        # ~15s on GPU; show status and force a redraw so the user isn't staring at a frozen UI.
        self.engine_status.configure(text=f"Calibrating threshold for {Path(path).name} …")
        self.root.update_idletasks()
        try:
            bank_raw = np.load(path)["windows"]
            cal = calibrate_threshold(self.engine, bank_raw, n_users=20)
            self._threshold = cal["tau"]
            self.threshold_label.configure(
                text=f"τ = {cal['tau']:.3f}  |  "
                     f"AUSC {cal['ausc']:.3f}  "
                     f"PTCR {cal['ptcr']*100:.1f}%  "
                     f"Usability {cal['usability']*100:.1f}%  "
                     f"EER {cal['eer']*100:.1f}%")
            status = f"Authenticating · bank: {Path(path).name}"
        except Exception as exc:  # noqa: BLE001
            self._threshold = 0.5
            self.threshold_label.configure(text="τ = 0.5  (calibration failed — fallback)")
            status = (f"Authenticating · bank: {Path(path).name} · "
                      f"calibration failed ({exc})")

        self._mode = "auth"
        self._clear_stream()
        self.engine_status.configure(text=status)
        self._update_counter()

    def _save_enrollment(self) -> None:
        """Multi-session enrollment state machine.

        Press 1 (idle → enroll): begin capturing session 1. Button = "Add session".
        Press 2..ENROLL_SESSIONS (enroll, after typing ≥ MIN_PER_SESSION keys): append
            current stream as a session, clear the buffer, prompt for the next session.
            Once ENROLL_SESSIONS collected, button = "Finish enrollment".
        Final press: build_bank_windows distributes BANK_K windows across the collected
            sessions (mirrors training's multi_session=True), save, return to idle.
        """
        if self.engine is None:
            return

        if self._mode != "enroll":
            self._mode = "enroll"
            self._enrollment_sessions = []
            self._clear_stream()
            self.enroll_button.configure(text="Add session")
            self.engine_status.configure(
                text=f"Session 1 / {ENROLL_SESSIONS}: type at least {MIN_PER_SESSION} keys, "
                     f"then click 'Add session'"
            )
            self._update_counter()
            return

        # Already collected all sessions → this press finalises.
        if len(self._enrollment_sessions) >= ENROLL_SESSIONS:
            all_windows = build_bank_windows(
                self._enrollment_sessions, seq_len=SEQ_LEN, k=BANK_K
            )
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            BANK_DIR.mkdir(parents=True, exist_ok=True)
            save_path = BANK_DIR / f"bank_{ts}.npz"
            save_bank(all_windows, path=save_path)
            self.engine.set_bank_from_windows(all_windows)
            self._enrollment_sessions = []
            self._mode = "idle"
            self.enroll_button.configure(text="Save enrollment")
            self.engine_status.configure(
                text=f"Enrolled · {all_windows.shape[0]} windows · saved to {save_path.name} "
                     f"· 'Start auth' to score"
            )
            self._clear_stream()
            return

        # Mid-enrollment: capture the current stream as one session.
        feats = self._current_features()
        n = 0 if feats is None else len(feats)
        if feats is None or n < MIN_PER_SESSION:
            self.engine_status.configure(
                text=f"Need at least {MIN_PER_SESSION} keys (got {n}) — keep typing"
            )
            return

        self._enrollment_sessions.append(feats)
        self._clear_stream()
        done = len(self._enrollment_sessions)
        if done >= ENROLL_SESSIONS:
            self.enroll_button.configure(text="Finish enrollment")
            self.engine_status.configure(
                text=f"All {ENROLL_SESSIONS} sessions captured · click 'Finish enrollment' to save"
            )
        else:
            self.engine_status.configure(
                text=f"Session {done + 1} / {ENROLL_SESSIONS}: type at least "
                     f"{MIN_PER_SESSION} keys, then click 'Add session'"
            )
        self._update_counter()

    def _reset_all(self) -> None:
        """Hard reset: live buffer + enrollment state + return to idle mode."""
        self._enrollment_sessions = []
        if self._mode == "enroll":
            self.enroll_button.configure(text="Save enrollment")
        self._mode = "idle"
        self._clear_stream()

    def _clear_stream(self) -> None:
        """Reset the live keystroke log, table, and probability history (keeps the bank)."""
        self._press_ms.clear()
        self._release_ms.clear()
        self._key_ids.clear()
        self._pressed_at.clear()
        self._last_release_at = None
        self._probability_history.clear()
        self.capture_var.set("")
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.empty_label.grid()
        self.set_probability(0.0)
        self._update_counter()


def _demo_rows() -> list[KeystrokeRow]:
    return [
        KeystrokeRow("A", 120.0, 80.0),
        KeystrokeRow("L", 105.5, 72.2),
        KeystrokeRow("T", 98.4, 65.1),
        KeystrokeRow("O", 110.0, 78.7),
        KeystrokeRow("Enter", 88.9, 50.0),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyboard authentication interface")
    parser.add_argument("--demo", action="store_true", help="launch with sample probability and keystrokes")
    args = parser.parse_args()

    root = tk.Tk()
    app = KeyboardAuthApp(root)

    if args.demo:
        app.set_probability(0.23)
        app.set_keystrokes(_demo_rows())

    root.mainloop()


if __name__ == "__main__":
    main()
