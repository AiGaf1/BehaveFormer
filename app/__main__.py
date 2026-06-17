"""Entry point: `python -m app` launches the keystroke authentication UI."""

import sys
from pathlib import Path

# Add project root to path to support direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ui import main

if __name__ == "__main__":
    main()
