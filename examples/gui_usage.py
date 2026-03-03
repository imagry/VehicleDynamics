#!/usr/bin/env python3
"""Example of using the GUI for parameter fitting.

This example shows how to launch the GUI application for interactive parameter fitting.
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fitting.gui import FittingGUI


def main() -> None:
    """Launch the fitting GUI."""
    root = tk.Tk()
    app = FittingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
