#!/usr/bin/env python3
"""Launch the feedforward comparison GUI."""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fitting.feedforward_gui import FeedforwardComparisonGUI


def main() -> None:
    root = tk.Tk()
    _app = FeedforwardComparisonGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
