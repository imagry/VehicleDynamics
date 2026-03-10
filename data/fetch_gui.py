"""Tkinter GUI for downloading trip data from S3."""

from __future__ import annotations

import logging
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

# Allow running this file directly: python data/fetch_gui.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.fetch import DEFAULT_FILES, FetchTripsConfig, TripFetcher


class _TextWidgetHandler(logging.Handler):
    """Log handler that appends records to a Tk text widget."""

    def __init__(self, text_widget: ScrolledText) -> None:
        super().__init__()
        self._text_widget = text_widget

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)

        def _append() -> None:
            self._text_widget.configure(state="normal")
            self._text_widget.insert(tk.END, msg + "\n")
            self._text_widget.see(tk.END)
            self._text_widget.configure(state="disabled")

        self._text_widget.after(0, _append)


class FetchTripsGUI:
    """GUI wrapper for :class:`data.fetch.TripFetcher`."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Trip Fetcher")
        self.root.geometry("980x720")

        self._running = False

        self.car_var = tk.StringVar(value="NiroEV")
        self.start_var = tk.StringVar(value="2024-01-01")
        self.end_var = tk.StringVar(value="2024-01-01")
        self.dest_var = tk.StringVar(value=str(Path("data/raw/trips")))
        self.vehicle_id_var = tk.StringVar(value="")
        self.max_gb_var = tk.StringVar(value="")
        self.bucket_var = tk.StringVar(value="trips-backup")
        self.root_prefix_var = tk.StringVar(value="trips_metadata")
        self.log_level_var = tk.StringVar(value="INFO")
        self.overwrite_var = tk.BooleanVar(value=False)
        self.dry_run_var = tk.BooleanVar(value=False)

        self._build_ui()
        self._configure_logger()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        form = ttk.LabelFrame(frame, text="Fetch Configuration", padding=10)
        form.pack(fill=tk.X)

        ttk.Label(form, text="Car type:").grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(form, textvariable=self.car_var, width=24).grid(row=0, column=1, sticky=tk.W, padx=4, pady=4)

        ttk.Label(form, text="Start date (YYYY-MM-DD):").grid(row=0, column=2, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(form, textvariable=self.start_var, width=16).grid(row=0, column=3, sticky=tk.W, padx=4, pady=4)

        ttk.Label(form, text="End date (YYYY-MM-DD):").grid(row=0, column=4, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(form, textvariable=self.end_var, width=16).grid(row=0, column=5, sticky=tk.W, padx=4, pady=4)

        ttk.Label(form, text="Destination:").grid(row=1, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(form, textvariable=self.dest_var, width=58).grid(row=1, column=1, columnspan=4, sticky=tk.EW, padx=4, pady=4)
        ttk.Button(form, text="Browse", command=self._browse_dest).grid(row=1, column=5, sticky=tk.W, padx=4, pady=4)

        ttk.Label(form, text="Vehicle ID (optional):").grid(row=2, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(form, textvariable=self.vehicle_id_var, width=24).grid(row=2, column=1, sticky=tk.W, padx=4, pady=4)

        ttk.Label(form, text="Max GB (optional):").grid(row=2, column=2, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(form, textvariable=self.max_gb_var, width=16).grid(row=2, column=3, sticky=tk.W, padx=4, pady=4)

        ttk.Label(form, text="Log level:").grid(row=2, column=4, sticky=tk.W, padx=4, pady=4)
        ttk.Combobox(
            form,
            textvariable=self.log_level_var,
            values=["DEBUG", "INFO", "WARNING", "ERROR"],
            state="readonly",
            width=14,
        ).grid(row=2, column=5, sticky=tk.W, padx=4, pady=4)

        ttk.Label(form, text="Bucket:").grid(row=3, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(form, textvariable=self.bucket_var, width=24).grid(row=3, column=1, sticky=tk.W, padx=4, pady=4)

        ttk.Label(form, text="Root prefix:").grid(row=3, column=2, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(form, textvariable=self.root_prefix_var, width=24).grid(row=3, column=3, sticky=tk.W, padx=4, pady=4)

        ttk.Checkbutton(form, text="Overwrite existing trip folders", variable=self.overwrite_var).grid(
            row=3, column=4, columnspan=2, sticky=tk.W, padx=4, pady=4
        )
        ttk.Checkbutton(form, text="Dry run (no downloads)", variable=self.dry_run_var).grid(
            row=4, column=4, columnspan=2, sticky=tk.W, padx=4, pady=4
        )

        files_frame = ttk.LabelFrame(frame, text="Required Files (one per line, blank = defaults)", padding=10)
        files_frame.pack(fill=tk.BOTH, expand=False, pady=(10, 0))
        self.files_text = ScrolledText(files_frame, height=8, wrap=tk.WORD)
        self.files_text.pack(fill=tk.BOTH, expand=True)
        self.files_text.insert("1.0", "\n".join(DEFAULT_FILES))

        actions = ttk.Frame(frame)
        actions.pack(fill=tk.X, pady=(10, 0))
        self.run_button = ttk.Button(actions, text="Run Fetch", command=self._start_fetch)
        self.run_button.pack(side=tk.LEFT)
        ttk.Button(actions, text="Clear Log", command=self._clear_log).pack(side=tk.LEFT, padx=(8, 0))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(actions, textvariable=self.status_var).pack(side=tk.LEFT, padx=(12, 0))

        log_frame = ttk.LabelFrame(frame, text="Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.log_text = ScrolledText(log_frame, height=14, wrap=tk.WORD, state="disabled")
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _configure_logger(self) -> None:
        self.logger = logging.getLogger("data.fetch_gui")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)

        text_handler = _TextWidgetHandler(self.log_text)
        text_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%H:%M:%S"))
        self.logger.addHandler(text_handler)

    def _browse_dest(self) -> None:
        selected = filedialog.askdirectory(initialdir=str(Path(self.dest_var.get()).parent))
        if selected:
            self.dest_var.set(selected)

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state="disabled")

    def _parse_files(self) -> list[str] | None:
        raw = self.files_text.get("1.0", tk.END)
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        return lines if lines else None

    def _build_config(self) -> FetchTripsConfig:
        car = self.car_var.get().strip()
        if not car:
            raise ValueError("Car type is required")

        max_gb_raw = self.max_gb_var.get().strip()
        max_gb = None
        if max_gb_raw:
            max_gb = float(max_gb_raw)
            if max_gb <= 0:
                raise ValueError("Max GB must be positive")

        vehicle_id = self.vehicle_id_var.get().strip() or None

        return FetchTripsConfig(
            car=car,
            start=self.start_var.get().strip(),
            end=self.end_var.get().strip(),
            dest=Path(self.dest_var.get().strip()),
            files=self._parse_files(),
            vehicle_id=vehicle_id,
            max_gb=max_gb,
            overwrite=self.overwrite_var.get(),
            dry_run=self.dry_run_var.get(),
            bucket=self.bucket_var.get().strip() or "trips-backup",
            root_prefix=self.root_prefix_var.get().strip() or "trips_metadata",
        )

    def _start_fetch(self) -> None:
        if self._running:
            return

        try:
            config = self._build_config()
        except Exception as exc:
            messagebox.showerror("Invalid Configuration", str(exc))
            return

        level_name = self.log_level_var.get().strip().upper() or "INFO"
        self.logger.setLevel(getattr(logging, level_name, logging.INFO))

        self._running = True
        self.run_button.configure(state="disabled")
        self.status_var.set("Running...")

        thread = threading.Thread(target=self._run_fetch_worker, args=(config,), daemon=True)
        thread.start()

    def _run_fetch_worker(self, config: FetchTripsConfig) -> None:
        try:
            self.logger.info("Starting fetch job")
            fetcher = TripFetcher(config=config, logger=self.logger)
            fetcher.run()
            self.root.after(0, lambda: self.status_var.set("Done"))
            self.logger.info("Fetch job finished")
        except Exception as exc:  # pragma: no cover - UI error path
            self.logger.exception("Fetch failed")
            self.root.after(0, lambda: messagebox.showerror("Fetch Error", str(exc)))
            self.root.after(0, lambda: self.status_var.set("Error"))
        finally:
            self.root.after(0, self._on_fetch_finished)

    def _on_fetch_finished(self) -> None:
        self._running = False
        self.run_button.configure(state="normal")


def main() -> None:
    root = tk.Tk()
    FetchTripsGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
