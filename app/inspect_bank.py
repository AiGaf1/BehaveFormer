"""Print enrollment bank NPZ as a keystroke table (letter | hold_s | flight_s).

Usage:
    python app/inspect_bank.py app/user_bank.npz
    python app/inspect_bank.py app/user_bank.npz --window 2
"""

import argparse
import numpy as np


def _char(key_id: float) -> str:
    # Banks store key_code + 1 (compute_features reserves 0 for padding), so undo it.
    code = int(key_id) - 1
    return chr(code).lower() if 32 <= code < 127 else f"[{code}]"


def print_bank(path: str, window_idx: int | None = None) -> None:
    windows = np.load(path, allow_pickle=True)["windows"]  # (K, L, F)
    K, L, F = windows.shape
    print(f"{path}  —  {K} windows × {L} keystrokes\n")

    indices = [window_idx] if window_idx is not None else range(K)
    for k in indices:
        print(f"  Window {k}")
        print(f"  {'letter':>6}  {'hold_s':>8}  {'flight_s':>9}")
        print(f"  {'-'*6}  {'-'*8}  {'-'*9}")
        for row in windows[k]:
            char   = _char(row[2]) if F > 2 else "?"
            hold   = row[0]
            flight = row[1] if F > 1 else 0.0
            print(f"  {char:>6}  {hold:>8.4f}  {flight:>9.4f}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to .npz bank file")
    parser.add_argument("--window", "-w", type=int, default=None,
                        help="Print only this window index (default: all)")
    args = parser.parse_args()
    print_bank(args.path, args.window)
