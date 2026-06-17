"""Keystroke authentication desktop UI.

A Tk window split into two areas:
  - a top probability panel for the current authentication score
  - a bottom table of captured keystrokes (hold / flight times)

The view owns presentation and the enroll/auth flow; keystroke-timing capture
lives in `app.recorder`, model scoring in `app.engine`, bank I/O in `app.bank_io`.
"""

from __future__ import annotations

import datetime
import json
import re
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import filedialog, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from app.bank_io import build_bank_windows, load_session_csv, save_session_csv
from app.engine import (
    BANK_DIR,
    AuthEngine,
    holdflight_to_features,
    keystrokes_to_features,
)
from app.keymap import event_key_name
from app.recorder import KeystrokeRecorder

# Keystrokes needed to fill one detector window (must match the model's seq_len).
SEQ_LEN          = 25
# Bank size: K windows spread across one enrollment session.
BANK_K           = 5
# Single-session enrollment: one typing session, split into BANK_K spread
# (non-overlapping) windows for the bank. Needs ≥ BANK_K*SEQ_LEN keys plus margin.
ENROLL_KEYS      = 150
# Decision threshold. Per-user: calibrated at enrollment to the CAL_PERCENTILE-th
# percentile of the user's own genuine p_t, so ~(100 - CAL_PERCENTILE)% of genuine
# windows land above τ regardless of the score distribution's shape. (μ+2σ assumed a
# Gaussian tail and over-tightened τ, false-flagging genuine users on their own data.)
# DEFAULT_THRESHOLD is the population operating point (tau_op), used if uncalibrated.
DEFAULT_THRESHOLD = 0.59
CAL_PERCENTILE    = 99.0
# Live-decision smoothing: EMA over p_t, and require SUSTAIN_N consecutive smoothed
# windows above threshold before flagging — a single spiky window won't flip status.
EMA_ALPHA        = 0.15
SUSTAIN_N        = 3


class KeyboardAuthApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Keyboard Authentication")
        self.root.geometry("1080x740")
        self.root.minsize(820, 560)
        # Scrolling plot window: keep the last PLOT_WINDOW points but track the
        # absolute event index so the x-axis keeps advancing past the window size.
        self._probability_history: deque[float] = deque(maxlen=120)
        self._plot_index = 0

        # Keystroke-timing capture for the live stream (see app.recorder).
        self.recorder = KeystrokeRecorder()
        # Capture mode: "idle" (typing does nothing), "auth" (live scoring vs the
        # loaded bank), or "enroll" (collecting keystrokes for a new bank).
        self._mode = "idle"
        # Per-user decision threshold (calibrated at enrollment; default until then).
        self._threshold: float = DEFAULT_THRESHOLD
        # Display name of the loaded bank's user (parsed from the filename).
        self._user_name: str = "genuine"
        # Live-decision smoothing state (see _update_live_score).
        self._ema: float | None = None
        self._sustain: int = 0

        self._configure_style()
        self._build_layout()
        self.set_probability(0.0)
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
        style.configure("Subtle.TLabel", background="#111827", foreground="#9ca3af", font=("TkDefaultFont", 12))
        style.configure("Value.TLabel", background="#111827", foreground="#f9fafb", font=("TkDefaultFont", 28, "bold"))
        style.configure("Status.TLabel", background="#111827", foreground="#34d399", font=("TkDefaultFont", 13, "bold"))
        style.configure("Verdict.TLabel", background="#1f2937", foreground="#9ca3af", font=("TkDefaultFont", 22, "bold"), padding=14, anchor="center")
        style.configure("Input.TEntry", fieldbackground="#0b1220", foreground="#f9fafb", insertcolor="#f9fafb", font=("TkDefaultFont", 18))
        style.configure("Big.TButton", font=("TkDefaultFont", 15, "bold"), padding=(18, 12))
        # Enroll-button states: amber while still collecting keys, green once the
        # profile can be built from the typed history (see _refresh_enroll_button).
        style.configure("Collect.TButton", font=("TkDefaultFont", 15, "bold"), padding=(18, 12),
                        background="#b45309", foreground="#fde68a")
        style.map("Collect.TButton", background=[("active", "#92400e")])
        style.configure("Ready.TButton", font=("TkDefaultFont", 15, "bold"), padding=(18, 12),
                        background="#15803d", foreground="#dcfce7")
        style.map("Ready.TButton", background=[("active", "#166534")])
        style.configure("App.Treeview", font=("TkDefaultFont", 13), rowheight=32)
        style.configure("App.Treeview.Heading", font=("TkDefaultFont", 12, "bold"))
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

        self.probability_status = ttk.Label(self.probability_panel, text="Awaiting input",
                                             style="Verdict.TLabel", anchor="center")
        self.probability_status.grid(row=2, column=0, sticky="ew", pady=(2, 8))

        self.probability_scores = ttk.Label(self.probability_panel, text="—", style="Value.TLabel")
        self.probability_scores.grid(row=3, column=0, sticky="w", pady=(0, 4))

        plot_frame = ttk.Frame(self.probability_panel, style="Panel.TFrame")
        plot_frame.grid(row=4, column=0, sticky="nsew", pady=(16, 0))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self.probability_figure = Figure(figsize=(7.2, 2.8), dpi=100, facecolor="#111827")
        self.probability_figure.subplots_adjust(left=0.04, right=0.99, top=0.94, bottom=0.22)
        self.probability_ax = self.probability_figure.add_subplot(111)
        self.probability_ax.set_facecolor("#0b1220")
        self.probability_ax.set_ylim(0.0, 1.0)
        self.probability_ax.set_xlim(0, 119)
        self.probability_ax.set_ylabel("Probability", color="#e5e7eb", fontsize=14)
        self.probability_ax.set_xlabel("Keystroke events", color="#e5e7eb", fontsize=14)
        self.probability_ax.tick_params(colors="#cbd5e1", labelsize=12)
        for spine in self.probability_ax.spines.values():
            spine.set_color("#475569")
        self.probability_ax.grid(True, color="#334155", alpha=0.35, linewidth=0.8)
        self.probability_line, = self.probability_ax.plot([], [], color="#38bdf8", linewidth=2.4, zorder=3)
        # Red overlay drawn on top of the segments that rise above τ (set in set_probability).
        self.probability_line_hi, = self.probability_ax.plot([], [], color="#f87171", linewidth=2.8, zorder=4)
        # Translucent danger zone above τ — repositioned/shown once a bank is loaded.
        self.impostor_zone = self.probability_ax.axhspan(self._threshold, 1.0, color="#ef4444", alpha=0.12, zorder=0)
        self.impostor_zone.set_visible(False)
        self.tau_line = self.probability_ax.axhline(
            self._threshold, color="#f87171", linewidth=1.4, linestyle="--", alpha=0.85,
        )
        self.tau_line.set_visible(False)

        self.probability_canvas = FigureCanvasTkAgg(self.probability_figure, master=plot_frame)
        canvas_widget = self.probability_canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, sticky="nsew")

        # Single form row: Name (~10%) and Type here (~90%) share the same y.
        form_row = ttk.Frame(self.probability_panel, style="Panel.TFrame")
        form_row.grid(row=5, column=0, sticky="ew", pady=(18, 0))
        form_row.columnconfigure(1, weight=1)     # name entry  → ~5%
        form_row.columnconfigure(3, weight=19)    # type entry  → ~95%

        name_label = ttk.Label(form_row, text="Name:", style="Subtle.TLabel",
                               font=("TkDefaultFont", 18), anchor="w")
        name_label.grid(row=0, column=0, sticky="w", padx=(0, 10))

        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(form_row, textvariable=self.name_var, style="Input.TEntry",
                                    font=("TkDefaultFont", 18), width=6)
        self.name_entry.grid(row=0, column=1, sticky="ew", ipady=8)

        capture_label = ttk.Label(form_row, text="Type here:", style="Subtle.TLabel",
                                   font=("TkDefaultFont", 18), anchor="w")
        capture_label.grid(row=0, column=2, sticky="w", padx=(16, 10))

        self.capture_var = tk.StringVar()
        self.capture_entry = ttk.Entry(form_row, textvariable=self.capture_var, style="Input.TEntry",
                                        font=("TkDefaultFont", 18))
        self.capture_entry.grid(row=0, column=3, sticky="ew", ipady=8)
        self.capture_entry.focus_set()
        self.capture_entry.bind("<KeyPress>", self._on_key_press)
        self.capture_entry.bind("<KeyRelease>", self._on_key_release)

        controls = ttk.Frame(self.probability_panel, style="Panel.TFrame")
        controls.grid(row=7, column=0, sticky="ew", pady=(14, 0))
        # Weighted spacer columns on each side center the button group.
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(4, weight=1)

        self.auth_button = ttk.Button(controls, text="Authenticate", style="Big.TButton",
                                      command=self._start_auth)
        self.auth_button.grid(row=0, column=1)

        self.enroll_button = ttk.Button(controls, text="Create profile", style="Big.TButton",
                                        command=self._save_enrollment)
        self.enroll_button.grid(row=0, column=2, padx=(10, 0))

        self.clear_button = ttk.Button(controls, text="Reset", style="Big.TButton",
                                       command=self._reset_all)
        self.clear_button.grid(row=0, column=3, padx=(10, 0))

        self.engine_status = ttk.Label(self.probability_panel, text="", style="Subtle.TLabel")
        self.engine_status.grid(row=8, column=0, sticky="w", pady=(8, 0))

        style = ttk.Style(self.root)
        style.configure("Counter.TLabel", background="#111827", foreground="#fbbf24",
                         font=("TkDefaultFont", 13, "bold"))
        self.counter_label = ttk.Label(self.probability_panel, text="", style="Counter.TLabel")
        self.counter_label.grid(row=9, column=0, sticky="w", pady=(4, 0))

        table_section = ttk.Frame(container, style="App.TFrame")
        table_section.grid(row=1, column=0, sticky="nsew", pady=(0, 0))
        table_section.rowconfigure(1, weight=1)
        table_section.columnconfigure(0, weight=1)

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
                        flagged: bool | None = None) -> None:
        threshold = self._threshold if threshold is None else threshold
        probability = max(0.0, min(1.0, float(probability)))
        # `flagged` overrides the simple p≥τ test (live scoring passes the smoothed,
        # sustained decision); fall back to the instantaneous test otherwise.
        if flagged is None:
            flagged = probability >= threshold
        self._probability_history.append(probability)
        self._plot_index += 1

        y_values = list(self._probability_history)
        # Absolute event indices for the points currently in the window, so the
        # x-axis scrolls forward (e.g. 200..320) instead of resetting at 0..120.
        x_start = self._plot_index - len(y_values)
        x_values = list(range(x_start, self._plot_index))
        self.probability_line.set_data(x_values, y_values)
        self.probability_ax.set_xlim(x_start, max(x_start + 1, self._plot_index - 1))
        engine = getattr(self, "engine", None)
        if self._mode == "auth" and engine is not None and engine.has_bank:
            self.tau_line.set_ydata([threshold, threshold])
            self.tau_line.set_visible(True)
            # Red overlay on the above-τ segments, plus the shaded danger zone.
            above = [v if v >= threshold else float("nan") for v in y_values]
            self.probability_line_hi.set_data(x_values, above)
            self.impostor_zone.set_bounds(x_start, threshold, max(1, len(y_values)), 1.0 - threshold)
            self.impostor_zone.set_visible(True)
        else:
            self.tau_line.set_visible(False)
            self.probability_line_hi.set_data([], [])
            self.impostor_zone.set_visible(False)
        self.probability_ax.figure.canvas.draw_idle()

        if self._mode != "auth":
            verdict_text, fg, bg = "Awaiting input", "#9ca3af", "#1f2937"
        elif flagged:
            verdict_text, fg, bg = "✕  IMPOSTOR DETECTED", "#fee2e2", "#b91c1c"
        else:
            verdict_text, fg, bg = f"✓  {self._user_name.upper()} IS TYPING", "#dcfce7", "#15803d"

        self.probability_status.configure(text=verdict_text)
        ttk.Style(self.root).configure("Verdict.TLabel", foreground=fg, background=bg)

        self.probability_scores.configure(text=f"Score = {probability:.2f}")
        ttk.Style(self.root).configure("Value.TLabel", foreground=fg)

    def _on_key_press(self, event: tk.Event) -> str | None:
        self.recorder.on_press(event.keycode, time.perf_counter())
        return None

    def _on_key_release(self, event: tk.Event) -> str | None:
        key_name = event_key_name(event)
        row = self.recorder.on_release(event.keycode, key_name, time.perf_counter())
        if row is None:
            return None
        hold_time, flight_time = row

        self.tree.insert("", "end", values=(key_name, f"{hold_time:.1f}", f"{flight_time:.1f}"))
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

        banks = sorted(BANK_DIR.glob("bank_*.csv"))
        if banks:
            latest = banks[-1]
            self._load_bank_csv(latest)
            self.engine_status.configure(
                text=f"Model ready · {latest.name} loaded · click 'Authenticate' to score")
        else:
            self.engine_status.configure(
                text="Model ready · no profile yet — click 'Create profile' to build one")

    @staticmethod
    def _user_name_from_path(path: Path) -> str:
        """Extract the user name from a `bank_{name}_{YYYYMMDD}_{HHMM}.csv` filename.

        Strips the `bank_` prefix and the trailing `_date_time` stamp; if no name
        part remains (older nameless banks), falls back to 'genuine'.
        """
        rest = path.stem[len("bank_"):] if path.stem.startswith("bank_") else path.stem
        name = re.sub(r"_?\d{8}_\d{4}$", "", rest).strip("_")
        return name or "genuine"

    def _load_bank_csv(self, path: Path) -> None:
        """Load a session CSV as the active bank (rebuilds windows from the session).

        Also seeds the live stream with the enrolled user's own events so auth scores
        from the first live keystroke with a warm accumulator (see set_stream_prefix),
        and sets the per-user threshold (from the .thr.json sidecar, else recomputed).
        """
        self._user_name = self._user_name_from_path(path)
        session = load_session_csv(path)                       # (n, 3) hold,flight,code
        feats = holdflight_to_features(session[:, 0], session[:, 1], session[:, 2])
        windows = build_bank_windows(feats, seq_len=SEQ_LEN, k=BANK_K)
        self.engine.set_bank_from_windows(windows)
        self.engine.set_stream_prefix(feats)

        thr_path = path.with_suffix(".thr.json")
        if thr_path.exists():
            self._threshold = float(json.loads(thr_path.read_text())["threshold"])
        else:
            self._threshold = self._calibrate_threshold(feats)

    def _calibrate_threshold(self, feats) -> float:
        """Per-user threshold = CAL_PERCENTILE-th percentile of the user's genuine p_t.

        Requires the bank + stream prefix already set to this user's session, so
        engine.score(feats) is the seeded genuine self-score (see set_stream_prefix).
        Targets ~(100 - CAL_PERCENTILE)% per-window genuine false-reject directly,
        without assuming the score distribution is Gaussian.
        """
        cal = self.engine.score(feats)
        if len(cal) == 0:
            return DEFAULT_THRESHOLD
        return float(np.percentile(cal, CAL_PERCENTILE))

    def _current_features(self):
        """Build the model feature stream from the logged keystrokes, or None."""
        if not len(self.recorder):
            return None
        return keystrokes_to_features(
            self.recorder.press_ms, self.recorder.release_ms, self.recorder.key_ids)

    def _update_counter(self) -> None:
        n = len(self.recorder)
        if self._mode == "enroll":
            self.counter_label.configure(text=f"Keys: {n} / {ENROLL_KEYS}")
        else:
            self.counter_label.configure(text=f"Keys typed: {n}")
        self._refresh_enroll_button(n)

    def _refresh_enroll_button(self, n: int) -> None:
        """Reflect profile-building progress in the enroll button's label and frame.

        idle: neutral 'Create profile'. While enrolling, the button switches to the
        amber 'Collect' frame and counts keys, turning green once enough history has
        accumulated to build the user profile.
        """
        if self._mode != "enroll":
            self.enroll_button.configure(text="Create profile", style="Big.TButton")
        elif n < ENROLL_KEYS:
            self.enroll_button.configure(text=f"Recording… {n}/{ENROLL_KEYS}", style="Collect.TButton")
        else:
            self.enroll_button.configure(text="Save profile ✓", style="Ready.TButton")

    def _update_live_score(self) -> None:
        """Score the full stream and update the probability panel (auth mode only).

        engine.score() runs the full detector (encoder + CUSUM accumulator) over all
        windows; scores[-1] is the latest window's p_t. We smooth p_t with an EMA and
        only flag once SUSTAIN_N consecutive smoothed windows exceed the per-user τ, so
        a single spiky window doesn't flip the status.
        """
        if self.engine is None or self._mode != "auth" or not self.engine.has_bank:
            return
        feats = self._current_features()
        if feats is None:
            return
        scores = self.engine.score(feats)
        if len(scores) == 0:
            return
        raw = float(scores[-1])
        self._ema = raw if self._ema is None else EMA_ALPHA * raw + (1 - EMA_ALPHA) * self._ema
        self._sustain = self._sustain + 1 if self._ema >= self._threshold else 0
        flagged = self._sustain >= SUSTAIN_N
        self.set_probability(self._ema, flagged=flagged)

    def _start_auth(self) -> None:
        """Pick a bank file, load it, and enter auth (fixed threshold)."""
        if self.engine is None:
            return
        path = filedialog.askopenfilename(
            title="Select enrollment bank",
            initialdir=str(BANK_DIR),
            filetypes=[("Bank files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        path = Path(path)
        try:
            self._load_bank_csv(path)
        except Exception as exc:  # noqa: BLE001
            self.engine_status.configure(text=f"Failed to load bank: {exc}")
            return

        self._mode = "auth"
        self._clear_stream()
        self.engine_status.configure(text=f"Authenticating · bank: {path.name}")
        self._update_counter()

    def _save_enrollment(self) -> None:
        """Single-session enrollment: one typing session → saved bank CSV.

        idle → enroll: start collecting keystrokes.
        enroll → finish: once ≥ ENROLL_KEYS typed, build BANK_K spread windows
            from this one session, save the session CSV, return to idle.
        """
        if self.engine is None:
            return

        # ── idle → start enrollment ───────────────────────────────────────────
        if self._mode != "enroll":
            self._mode = "enroll"
            self._clear_stream()
            self.engine_status.configure(
                text=f"Type at least {ENROLL_KEYS} keys of your normal typing, "
                     f"then click 'Save profile'"
            )
            self._update_counter()
            return

        # ── enroll → finalize: build + save bank from this one session ───
        feats = self._current_features()
        n = 0 if feats is None else len(feats)
        if feats is None or n < ENROLL_KEYS:
            self.engine_status.configure(
                text=f"Need at least {ENROLL_KEYS} keys (got {n}) — keep typing"
            )
            return

        # Persist the FULL session as a plain key,hold,flight CSV (normalization-
        # agnostic, reusable as an experiment dataset sample). The recorder derives
        # hold/flight in the model's convention so reloading reproduces these feats.
        hold_ms, flight_ms = self.recorder.session_hold_flight_ms()
        windows = build_bank_windows(feats, seq_len=SEQ_LEN, k=BANK_K)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        # Sanitize the typed name into a filename-safe slug (fallback if blank).
        name = "".join(c if c.isalnum() else "_" for c in self.name_var.get().strip())
        name = name.strip("_") or "user"
        BANK_DIR.mkdir(parents=True, exist_ok=True)
        save_path = BANK_DIR / f"bank_{name}_{ts}.csv"
        save_session_csv(save_path, self.recorder.key_names, hold_ms, flight_ms)
        self.engine.set_bank_from_windows(windows)
        self.engine.set_stream_prefix(feats)

        # Per-user calibration: threshold = CAL_PERCENTILE-th percentile of this
        # user's genuine p_t. Saved next to the bank so 'Start auth' loads it without
        # recomputing.
        self._threshold = self._calibrate_threshold(feats)
        save_path.with_suffix(".thr.json").write_text(json.dumps(
            {"threshold": self._threshold, "percentile": CAL_PERCENTILE}, indent=2))
        self.engine_status.configure(
            text=f"Profile saved to {save_path.name} · τ={self._threshold:.2f} · click 'Authenticate' to score"
        )

        self._mode = "idle"
        self._clear_stream()

    def _reset_all(self) -> None:
        """Hard reset: live buffer + enrollment state + return to idle mode."""
        self._mode = "idle"
        self._clear_stream()

    def _clear_stream(self) -> None:
        """Reset the live keystroke log, table, and probability history (keeps the bank)."""
        self.recorder.clear()
        self._probability_history.clear()
        self._plot_index = 0
        self._ema = None
        self._sustain = 0
        self.capture_var.set("")
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.empty_label.grid()
        self.set_probability(0.0)
        self._update_counter()


def main() -> None:
    root = tk.Tk()
    KeyboardAuthApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
