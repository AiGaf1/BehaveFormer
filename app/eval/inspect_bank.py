"""Print the contents of a saved bank npz as a readable table.

Usage:
    python -m app.eval.inspect_bank [path/to/bank.npz]

Without an argument, inspects all banks in app/banks/.
"""

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

_KEYCODE_NAMES: dict[int, str] = {
    8: "BackSpace", 9: "Tab", 13: "Enter",
    16: "Shift", 17: "Ctrl", 18: "Alt", 20: "CapsLock",
    27: "Escape", 32: "Space",
    33: "PgUp", 34: "PgDn", 35: "End", 36: "Home",
    37: "Left", 38: "Up", 39: "Right", 40: "Down",
    45: "Insert", 46: "Delete",
    186: ";", 187: "=", 188: ",", 189: "-", 190: ".", 191: "/", 192: "`",
    219: "[", 220: "\\", 221: "]", 222: "'", 229: "IME",
}


def _label(keycode: int) -> str:
    if keycode in _KEYCODE_NAMES:
        return _KEYCODE_NAMES[keycode]
    if 65 <= keycode <= 90:
        return chr(keycode)
    if 48 <= keycode <= 57:
        return chr(keycode)
    return f"#{keycode}"


def inspect(path: Path) -> None:
    data = np.load(path)
    windows = data["windows"]       # (K, L, 3): [hold_s, flight_s, key_id]
    K, L, _ = windows.shape

    znorm = ""
    if "cal_znorm_mu" in data.files:
        znorm = f"  znorm: μ={float(data['cal_znorm_mu']):.3f}  σ={float(data['cal_znorm_sigma']):.3f}"

    print(f"=== {path.name}  ({K} enrollment windows × {L} keystrokes){znorm} ===")
    print(f"  {'Win':>4}  {'Pos':>3}  {'Key':<12}  {'Code':>5}  {'Hold(ms)':>9}  {'Flight(ms)':>10}")
    print("  " + "-" * 56)
    for w in range(K):
        for pos in range(L):
            hold_s, flight_s, kid = windows[w, pos]
            code = int(kid) - 1          # features store key_id = raw_keycode + 1
            print(f"  {w:>4}  {pos:>3}  {_label(code):<12}  {code:>5}  "
                  f"{hold_s * 1000:>9.1f}  {flight_s * 1000:>10.1f}")
    print()


def main() -> None:
    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        paths = sorted((Path(__file__).resolve().parents[1] / "banks").glob("*.npz"))

    if not paths:
        print("No bank files found.")
        return

    for p in paths:
        inspect(p)


if __name__ == "__main__":
    main()
