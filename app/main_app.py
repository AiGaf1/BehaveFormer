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
from app.keystrokes import event_key_name, key_code

# Keystrokes needed to fill one detector window (must match the model's seq_len).
SEQ_LEN          = 25
# Bank size: matches training (K=5) so sim_std / lme feature distributions match
# what the detector's linear score layer saw during training.
BANK_K           = 5
# Multi-session enrollment: K windows are distributed across this many independent
# sessions (mirrors training's build_raw_banks(multi_session=True)).
ENROLL_SESSIONS  = 5
# With BANK_K // ENROLL_SESSIONS = 1, one window per session is enough.
MIN_PER_SESSION  = SEQ_LEN
# Self-calibration: keystrokes of genuine typing scored against the user's own bank
# right after enrollment, used as the *real* genuine distribution for threshold τ.
VERIFY_KEYS      = 200


@dataclass(frozen=True)
class KeystrokeRow:
    key: str
    hold_time: float
    flight_time: float


class KeyboardAuthApp:
    def __init__(self, root: tk.Tk, debug_log: bool = False):
        self.root = root
        self._debug_log = debug_log
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
        # Verify-mode state: the just-saved bank waiting for self-calibration.
        self._verify_bank_path: Path | None = None
        self._verify_bank_windows: np.ndarray | None = None

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

        # Diagnostic line: windows scored + recent p_t distribution. Helps tell
        # whether a wrong authentication outcome is from drift, bank quality, or
        # threshold calibration. See docs note in _update_live_score.
        self.probability_debug = ttk.Label(self.probability_panel, text="", style="Subtle.TLabel")
        self.probability_debug.grid(row=3, column=0, sticky="e", pady=(4, 0))

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
        self.tau_line = self.probability_ax.axhline(
            self._threshold, color="#f87171", linewidth=1.4, linestyle="--", alpha=0.85,
        )
        self.tau_line.set_visible(False)

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

    def set_probability(self, probability: float, threshold: float | None = None) -> None:
        threshold = self._threshold if threshold is None else threshold
        probability = max(0.0, min(1.0, float(probability)))
        self._probability_history.append(probability)

        y_values = list(self._probability_history)
        x_values = list(range(len(y_values)))
        self.probability_line.set_data(x_values, y_values)
        self.probability_ax.set_xlim(0, max(1, len(y_values) - 1))
        engine = getattr(self, "engine", None)
        if engine is not None and engine.has_bank:
            self.tau_line.set_ydata([threshold, threshold])
            self.tau_line.set_visible(True)
        else:
            self.tau_line.set_visible(False)
        self.probability_ax.figure.canvas.draw_idle()

        if probability >= threshold:
            status_text = "High impostor risk"
            status_color = "#f87171"
        else:
            status_text = "Authentication looks stable"
            status_color = "#34d399"

        self.probability_status.configure(text=status_text)
        ttk.Style(self.root).configure("Status.TLabel", foreground=status_color)

        self.probability_scores.configure(text=f"Cumulative score: {probability:.3f}")
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

    def _on_key_press(self, event: tk.Event) -> str | None:
        key = event_key_name(event)
        now = time.perf_counter()
        if key not in self._pressed_at:
            self._pressed_at[key] = now
        return None

    def _on_key_release(self, event: tk.Event) -> str | None:
        key = event_key_name(event)
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
        self._key_ids.append(key_code(key))

        self.tree.insert("", "end", values=(key, f"{hold_time:.1f}", f"{flight_time:.1f}"))
        self.empty_label.grid_remove()

        self._update_live_score()
        self._update_counter()
        return None

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
            npz = np.load(BANK_PATH)
            if "cal_tau" in npz.files:
                self.engine_status.configure(
                    text=f"Model ready · user loaded from {BANK_PATH.name} · click 'Start auth' to score")
            else:
                self.engine_status.configure(
                    text=f"Model ready · {BANK_PATH.name} loaded (no calibration — 'Start auth' will calibrate automatically)")
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
        elif self._mode == "verify":
            self.counter_label.configure(
                text=f"Verification (genuine typing)  ·  Keys: {n} / {VERIFY_KEYS}"
            )
        else:
            self.counter_label.configure(text=f"Keys typed: {n}")

    @staticmethod
    def _threshold_from_cal(cal: dict) -> float:
        """Pick the auth threshold from a calibration dict, in raw-p_t space.

        Prefers z-norm (μ + z·σ) when available — transfers across users.
        Falls back to the legacy `tau` (per-bank EER/τ_op).
        """
        sigma = cal.get("znorm_sigma")
        mu = cal.get("znorm_mu")
        if sigma is not None and mu is not None and float(sigma) > 0:
            z = float(cal.get("z_threshold", 3.0))
            return float(mu) + z * float(sigma)
        return float(cal["tau"])

    @classmethod
    def _format_cal_label(cls, cal: dict, mode: str) -> str:
        """Render the calibration line. Shows z-norm fields when present."""
        thr = cls._threshold_from_cal(cal)
        head = f"τ_eff = {thr:.3f}"
        sigma = cal.get("znorm_sigma")
        mu = cal.get("znorm_mu")
        if sigma is not None and mu is not None and float(sigma) > 0:
            z = float(cal.get("z_threshold", 3.0))
            head += f"  (z={z:.1f}  μ={float(mu):.3f}  σ={float(sigma):.3f})"
        else:
            head += f"  (τ_op={float(cal['tau']):.3f})"
        return (f"{head}  |  AUSC {cal['ausc']:.3f}  PTCR {cal['ptcr']*100:.1f}%  "
                f"Usability {cal['usability']*100:.1f}%  EER {cal['eer']*100:.1f}%  ({mode})")

    def _update_live_score(self) -> None:
        """Score the full stream and update the probability panel (auth mode only).

        engine.score() runs the full detector (encoder + CUSUM accumulator) over all
        windows so history is integrated. scores[-1] is the latest window's p_t.

        The diagnostic line reports (n_windows, min/mean/max of last 20 p_t). If the
        curve climbs steadily during genuine typing, the accumulator is drifting; if
        it stays high but flat, the bank is unrepresentative; if it looks fine but
        auth still fails, the threshold τ is wrong.
        """
        if (self.engine is None or self._mode not in ("auth", "verify")
                or not self.engine.has_bank):
            return
        feats = self._current_features()
        if feats is None:
            return
        scores = self.engine.score(feats)
        if len(scores) == 0:
            return
        self.set_probability(float(scores[-1]))

        recent = scores[-20:] if len(scores) >= 20 else scores
        if self._debug_log:
            sys.stderr.write(
                f"[p_t] n_windows={len(scores)} latest={scores[-1]:.4f} "
                f"recent_mean={float(recent.mean()):.4f} "
                f"min={float(recent.min()):.4f} max={float(recent.max()):.4f}\n"
            )
        self.probability_debug.configure(
            text=(f"n_w={len(scores)}  "
                  f"recent: min={float(recent.min()):.2f} "
                  f"mean={float(recent.mean()):.2f} "
                  f"max={float(recent.max()):.2f}")
        )

    def _start_auth(self) -> None:
        """Pick a bank file, load saved τ if present (else live-calibrate), enter auth."""
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
            bank_npz = np.load(path)
            self.engine.set_bank_from_windows(bank_npz["windows"])
        except Exception as exc:  # noqa: BLE001
            self.engine_status.configure(text=f"Failed to load bank: {exc}")
            return

        n_windows = bank_npz["windows"].shape[0]
        if n_windows < 5:
            self.engine_status.configure(
                text=f"Warning: bank has only {n_windows} windows (< 5) — accuracy may be reduced. Re-enrol with more typing."
            )
            self.root.update_idletasks()

        if "cal_tau" in bank_npz.files:
            # Fast path: trust the calibration saved at enrollment time.
            cal = {k[4:]: bank_npz[k].item() if bank_npz[k].ndim == 0 else bank_npz[k]
                   for k in bank_npz.files if k.startswith("cal_")}
            self._threshold = self._threshold_from_cal(cal)
            self.threshold_label.configure(text=self._format_cal_label(cal, mode="saved"))
            status = f"Authenticating · bank: {Path(path).name}"
        else:
            # No saved calibration → run the Aalto-proxy live calibration (~15s).
            self.engine_status.configure(text=f"Calibrating threshold for {Path(path).name} …")
            self.root.update_idletasks()
            try:
                cal = calibrate_threshold(self.engine, bank_npz["windows"], n_users=20)
                self._threshold = self._threshold_from_cal(cal)
                self.threshold_label.configure(text=self._format_cal_label(cal, mode="Aalto proxy"))
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
        """Multi-session enrollment + self-calibration state machine.

        idle → enroll: begin session 1. Button = "Add session".
        enroll, sessions < ENROLL_SESSIONS: append session on each press.
            Once all collected, button = "Finish enrollment".
        enroll, finish: build bank, save, transition to verify mode. Button = "Finish calibration".
        verify: user types ≥ VERIFY_KEYS of genuine traffic. On press, score against the
            saved bank for the *real* genuine distribution, calibrate τ vs Aalto impostors,
            re-save bank with cal metadata, return to idle.
        """
        if self.engine is None:
            return

        # ── verify mode: finalize self-calibration ────────────────────────────
        if self._mode == "verify":
            self._finalize_calibration()
            return

        # ── idle → start enrollment ───────────────────────────────────────────
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

        # ── enroll, all sessions captured → finalize bank + enter verify ──────
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

            # Stash for verification → we re-save with calibration on finalize.
            self._verify_bank_path = save_path
            self._verify_bank_windows = all_windows
            self._mode = "verify"
            self.enroll_button.configure(text="Finish calibration")
            self.engine_status.configure(
                text=f"Bank saved · type ≥ {VERIFY_KEYS} keys of your normal typing, "
                     f"then click 'Finish calibration'"
            )
            self._clear_stream()
            self._update_counter()
            return

        # ── enroll, mid-flight: capture current stream as a session ───────────
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

    def _finalize_calibration(self) -> None:
        """Self-calibrate τ using the user's own verification stream + Aalto impostors."""
        feats = self._current_features()
        n = 0 if feats is None else len(feats)
        if feats is None or n < VERIFY_KEYS:
            self.engine_status.configure(
                text=f"Need at least {VERIFY_KEYS} verification keys (got {n}) — keep typing"
            )
            return

        self.engine_status.configure(text="Calibrating with your genuine typing …")
        self.root.update_idletasks()
        try:
            cal = calibrate_threshold(
                self.engine, self._verify_bank_windows,
                n_users=20, real_genuine_events=feats,
            )
            self._threshold = self._threshold_from_cal(cal)
            save_bank(self._verify_bank_windows,
                      path=self._verify_bank_path, calibration=cal)
            self.threshold_label.configure(text=self._format_cal_label(
                cal, mode=f"self-calibrated, n={cal['n_genuine']}"))
            self.engine_status.configure(
                text=f"Self-calibrated · saved to {self._verify_bank_path.name} "
                     f"· 'Start auth' to score"
            )
        except Exception as exc:  # noqa: BLE001
            self.engine_status.configure(
                text=f"Calibration failed: {exc} — bank still saved without τ"
            )

        self._verify_bank_path = None
        self._verify_bank_windows = None
        self._mode = "idle"
        self.enroll_button.configure(text="Save enrollment")
        self._clear_stream()

    def _reset_all(self) -> None:
        """Hard reset: live buffer + enrollment/verify state + return to idle mode."""
        self._enrollment_sessions = []
        self._verify_bank_path = None
        self._verify_bank_windows = None
        if self._mode in ("enroll", "verify"):
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
    parser.add_argument("--debug", action="store_true",
                        help="log per-window p_t to stderr (diagnostics)")
    args = parser.parse_args()

    root = tk.Tk()
    app = KeyboardAuthApp(root, debug_log=args.debug)

    if args.demo:
        app.set_probability(0.23)
        app.set_keystrokes(_demo_rows())

    root.mainloop()


if __name__ == "__main__":
    main()
