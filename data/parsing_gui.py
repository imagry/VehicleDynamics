"""Tkinter GUI for parsing raw trip folders into datasets."""

from __future__ import annotations

import logging
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

# Allow running this file directly: python data/parsing_gui.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.parsing import TripDatasetParser, TripParserConfig


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


class TripParsingGUI:
    """GUI wrapper for :class:`data.parsing.TripDatasetParser`."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Trip Parser")
        self.root.geometry("980x680")

        self._running = False

        self.root_var = tk.StringVar(value=str(Path("data/raw/trips")))
        self.car_var = tk.StringVar(value="NiroEV")
        self.vehicle_id_var = tk.StringVar(value="")
        self.dt_var = tk.StringVar(value="0.005")
        self.out_dir_var = tk.StringVar(value="")
        self.out_file_var = tk.StringVar(value="all_trips_data.pt")
        self.smooth_speed_var = tk.BooleanVar(value=False)
        self.allow_all_modes_var = tk.BooleanVar(value=False)
        self.log_level_var = tk.StringVar(value="INFO")

        self._build_ui()
        self._configure_logger()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        form = ttk.LabelFrame(frame, text="Parsing Configuration", padding=10)
        form.pack(fill=tk.X)

        ttk.Label(form, text="Raw trips root:").grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(form, textvariable=self.root_var, width=64).grid(row=0, column=1, columnspan=4, sticky=tk.EW, padx=4, pady=4)
        ttk.Button(form, text="Browse", command=self._browse_root).grid(row=0, column=5, sticky=tk.W, padx=4, pady=4)

        ttk.Label(form, text="Car model:").grid(row=1, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(form, textvariable=self.car_var, width=24).grid(row=1, column=1, sticky=tk.W, padx=4, pady=4)

        ttk.Label(form, text="Vehicle ID (optional):").grid(row=1, column=2, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(form, textvariable=self.vehicle_id_var, width=24).grid(row=1, column=3, sticky=tk.W, padx=4, pady=4)

        ttk.Label(form, text="dt (seconds):").grid(row=1, column=4, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(form, textvariable=self.dt_var, width=10).grid(row=1, column=5, sticky=tk.W, padx=4, pady=4)

        ttk.Label(form, text="Output directory:").grid(row=2, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(form, textvariable=self.out_dir_var, width=64).grid(row=2, column=1, columnspan=4, sticky=tk.EW, padx=4, pady=4)
        ttk.Button(form, text="Browse", command=self._browse_out_dir).grid(row=2, column=5, sticky=tk.W, padx=4, pady=4)

        ttk.Label(form, text="Output file:").grid(row=3, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(form, textvariable=self.out_file_var, width=32).grid(row=3, column=1, sticky=tk.W, padx=4, pady=4)

        ttk.Label(form, text="Log level:").grid(row=3, column=2, sticky=tk.W, padx=4, pady=4)
        ttk.Combobox(
            form,
            textvariable=self.log_level_var,
            values=["DEBUG", "INFO", "WARNING", "ERROR"],
            state="readonly",
            width=14,
        ).grid(row=3, column=3, sticky=tk.W, padx=4, pady=4)

        ttk.Checkbutton(form, text="Smooth speed", variable=self.smooth_speed_var).grid(
            row=4, column=0, columnspan=2, sticky=tk.W, padx=4, pady=4
        )
        ttk.Checkbutton(form, text="Allow all driving modes", variable=self.allow_all_modes_var).grid(
            row=4, column=2, columnspan=2, sticky=tk.W, padx=4, pady=4
        )

        actions = ttk.Frame(frame)
        actions.pack(fill=tk.X, pady=(10, 0))
        self.run_button = ttk.Button(actions, text="Run Parsing", command=self._start_parse)
        self.run_button.pack(side=tk.LEFT)
        ttk.Button(actions, text="Clear Log", command=self._clear_log).pack(side=tk.LEFT, padx=(8, 0))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(actions, textvariable=self.status_var).pack(side=tk.LEFT, padx=(12, 0))

        hint_frame = ttk.LabelFrame(frame, text="Note", padding=8)
        hint_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(
            hint_frame,
            text=(
                "If Output directory is blank, it defaults to "
                "data/processed/<car_model>/<vehicle_id_or_unknown_vehicle>."
            ),
        ).pack(anchor=tk.W)

        log_frame = ttk.LabelFrame(frame, text="Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.log_text = ScrolledText(log_frame, height=16, wrap=tk.WORD, state="disabled")
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _configure_logger(self) -> None:
        self.logger = logging.getLogger("data.parsing_gui")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)

        text_handler = _TextWidgetHandler(self.log_text)
        text_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%H:%M:%S"))
        self.logger.addHandler(text_handler)

    def _browse_root(self) -> None:
        selected = filedialog.askdirectory(initialdir=str(Path(self.root_var.get()).parent))
        if selected:
            self.root_var.set(selected)

    def _browse_out_dir(self) -> None:
        initial = self.out_dir_var.get().strip()
        selected = filedialog.askdirectory(initialdir=initial if initial else str(Path("data/processed")))
        if selected:
            self.out_dir_var.set(selected)

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state="disabled")

    def _build_config(self) -> TripParserConfig:
        car_model = self.car_var.get().strip()
        if not car_model:
            raise ValueError("Car model is required")

        dt = float(self.dt_var.get().strip())
        if dt <= 0:
            raise ValueError("dt must be positive")

        vehicle_id = self.vehicle_id_var.get().strip() or None

        out_dir_raw = self.out_dir_var.get().strip()
        if out_dir_raw:
            out_dir = Path(out_dir_raw)
        else:
            out_dir = Path("data/processed") / car_model / (vehicle_id or "unknown_vehicle")

        out_file = self.out_file_var.get().strip() or "all_trips_data.pt"

        config = TripParserConfig(
            root_folder=Path(self.root_var.get().strip()),
            car_model=car_model,
            vehicle_id=vehicle_id,
            dt=dt,
            out_dir=out_dir,
            out_file=out_file,
            smooth_speed=self.smooth_speed_var.get(),
        )
        config.require_driving_mode = not self.allow_all_modes_var.get()
        return config

    def _start_parse(self) -> None:
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

        thread = threading.Thread(target=self._run_parse_worker, args=(config,), daemon=True)
        thread.start()

    def _run_parse_worker(self, config: TripParserConfig) -> None:
        try:
            self.logger.info("Starting parse job")
            parser = TripDatasetParser(config)
            output_path = parser.save()
            self.root.after(0, lambda: self.status_var.set("Done"))
            self.logger.info("Saved dataset to %s", output_path)
        except Exception as exc:  # pragma: no cover - UI error path
            self.logger.exception("Parsing failed")
            self.root.after(0, lambda: messagebox.showerror("Parsing Error", str(exc)))
            self.root.after(0, lambda: self.status_var.set("Error"))
        finally:
            self.root.after(0, self._on_parse_finished)

    def _on_parse_finished(self) -> None:
        self._running = False
        self.run_button.configure(state="normal")


def main() -> None:
    root = tk.Tk()
    TripParsingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
