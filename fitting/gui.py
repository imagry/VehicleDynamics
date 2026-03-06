"""GUI application for vehicle parameter fitting configuration and execution."""

from __future__ import annotations

import json
import logging
import threading
import time
import tkinter as tk
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Dict, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy import signal

from fitting.fitter import FitterConfig, TripSegment, VehicleParamFitter

matplotlib.use("TkAgg")

LOGGER = logging.getLogger(__name__)

# Parameter groups for DC motor model
PARAM_GROUPS_DC = {
    "Body": ["mass", "drag_area", "rolling_coeff"],
    "Motor": ["motor_V_max", "motor_R", "motor_K", "motor_b", "motor_J", "motor_gamma_throttle", "motor_throttle_tau", "motor_min_current_A"],
    "Motor Limits": ["motor_T_max", "motor_P_max"],
    "Drivetrain": ["gear_ratio", "eta_gb"],
    "Brake": ["brake_T_max", "brake_tau", "brake_p", "brake_kappa", "mu"],
    "Wheel": ["wheel_radius", "wheel_inertia"],
}

# Parameter groups for polynomial motor model
PARAM_GROUPS_POLY = {
    "Body": ["mass", "drag_area", "rolling_coeff"],
    "Motor": ["motor_V_max", "motor_gamma_throttle", "motor_throttle_tau", "motor_min_current_A"],
    "Polynomial Coefficients": [
        "poly_c_00", "poly_c_10", "poly_c_01", "poly_c_20", "poly_c_11", "poly_c_02",
        "poly_c_30", "poly_c_21", "poly_c_12", "poly_c_03"
    ],
    "Drivetrain": ["gear_ratio", "eta_gb"],
    "Brake": ["brake_T_max", "brake_tau", "brake_p", "brake_kappa", "mu"],
    "Wheel": ["wheel_radius", "wheel_inertia"],
}

# Parameter display names and units
PARAM_DISPLAY = {
    "mass": ("Mass", "kg"),
    "drag_area": ("Drag Area", "m²"),
    "rolling_coeff": ("Rolling Coefficient", ""),
    "motor_V_max": ("Motor V_max", "V"),
    "motor_R": ("Motor R", "Ω"),
    "motor_K": ("Motor K", "Nm/A"),
    "motor_b": ("Motor b", "Nm·s/rad"),
    "motor_J": ("Motor J", "kg·m²"),
    "motor_gamma_throttle": ("Throttle γ", ""),
    "motor_throttle_tau": ("Throttle τ", "s"),
    "motor_min_current_A": ("Motor I_min", "A"),
    "motor_T_max": ("Motor T_max", "Nm"),
    "motor_P_max": ("Motor P_max", "W"),
    "poly_c_00": ("c₀₀ (constant)", ""),
    "poly_c_10": ("c₁₀ (V)", ""),
    "poly_c_01": ("c₀₁ (ω)", ""),
    "poly_c_20": ("c₂₀ (V²)", ""),
    "poly_c_11": ("c₁₁ (V·ω)", ""),
    "poly_c_02": ("c₀₂ (ω²)", ""),
    "poly_c_30": ("c₃₀ (V³)", ""),
    "poly_c_21": ("c₂₁ (V²·ω)", ""),
    "poly_c_12": ("c₁₂ (V·ω²)", ""),
    "poly_c_03": ("c₀₃ (ω³)", ""),
    "gear_ratio": ("Gear Ratio", ""),
    "eta_gb": ("Gearbox Efficiency", ""),
    "brake_T_max": ("Brake T_max", "Nm"),
    "brake_tau": ("Brake τ", "s"),
    "brake_p": ("Brake p", ""),
    "brake_kappa": ("Brake κ", ""),
    "mu": ("Friction μ", ""),
    "wheel_radius": ("Wheel Radius", "m"),
    "wheel_inertia": ("Wheel Inertia", "kg·m²"),
}


class FittingGUI:
    """Main GUI application for vehicle parameter fitting."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Vehicle Parameter Fitting")
        self.root.geometry("1400x900")

        # Default config for initial values
        self.default_config = FitterConfig()
        
        # Settings file path
        self.settings_file = Path(__file__).parent / "gui_settings.json"

        # Data paths
        self.data_dir = Path(__file__).parent.parent / "data" / "processed"
        self.results_dir = Path(__file__).parent / "results"

        # Fitting state
        self.fitting_thread: Optional[threading.Thread] = None
        self.is_fitting = False
        self.current_fitter: Optional[VehicleParamFitter] = None  # Store fitter for param names
        self.validation_segment = None  # Store a segment for real data comparison
        self.example_segment_key: Optional[Tuple[str, int]] = None

        # Update throttling
        self.last_update_time = 0.0
        self.update_interval = 0.5  # Minimum seconds between plot updates
        self.pending_params: Optional[np.ndarray] = None
        self.pending_loss: Optional[float] = None
        self.update_pending = False

        # Plot line objects for efficient updates
        self.throttle_lines = {}
        self.throttle_power_lines = {}
        self.throttle_current_lines = {}
        self.brake_lines = {}
        self.val_lines = {}
        self.val_act_lines = {}
        self.val_power_lines = {}
        self.val_current_lines = {}

        # Create main layout
        self._create_widgets()
        
        # Load saved settings if available
        self._load_settings()

    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main container with paned windows for resizing
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel: Configuration (scrollable)
        left_container = ttk.Frame(main_paned)
        main_paned.add(left_container, weight=1)

        left_canvas = tk.Canvas(left_container)
        left_scrollbar = ttk.Scrollbar(left_container, orient=tk.VERTICAL, command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        left_frame = ttk.Frame(left_canvas)
        left_window_id = left_canvas.create_window((0, 0), window=left_frame, anchor="nw")

        def _on_left_frame_configure(_event):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))

        def _on_left_container_configure(event):
            left_canvas.itemconfigure(left_window_id, width=event.width)

        left_frame.bind("<Configure>", _on_left_frame_configure)
        left_container.bind("<Configure>", _on_left_container_configure)

        # Right panel: Simulation preview
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        # Left panel: Top section (dataset and name)
        top_left = ttk.LabelFrame(left_frame, text="Fitting Configuration", padding=10)
        top_left.pack(fill=tk.X, padx=5, pady=5)

        # Dataset selection
        ttk.Label(top_left, text="Dataset:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.dataset_var = tk.StringVar()
        self.dataset_combo = ttk.Combobox(
            top_left, textvariable=self.dataset_var, state="readonly", width=50
        )
        self.dataset_combo.grid(row=0, column=1, columnspan=2, sticky=tk.EW, pady=5, padx=5)
        self._populate_datasets()

        # Example segment selection
        ttk.Label(top_left, text="Example Segment:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.example_segment_mode_var = tk.StringVar(value="Longest")
        self.example_segment_mode_combo = ttk.Combobox(
            top_left,
            textvariable=self.example_segment_mode_var,
            state="readonly",
            values=["Longest", "Random", "Index"],
            width=12,
        )
        self.example_segment_mode_combo.grid(row=6, column=1, sticky=tk.W, pady=5, padx=5)
        self.example_segment_mode_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_example_segment_mode_changed())

        self.example_segment_index_var = tk.StringVar(value="0")
        self.example_segment_index_entry = ttk.Entry(
            top_left, textvariable=self.example_segment_index_var, width=10
        )
        self.example_segment_index_entry.grid(row=6, column=2, sticky=tk.W, pady=5, padx=5)
        self._on_example_segment_mode_changed()

        # Fitting name
        ttk.Label(top_left, text="Fitting Name:").grid(row=1, column=0, sticky=tk.W, pady=5)
        default_name = datetime.now().strftime("fit_%Y%m%d_%H%M%S")
        self.name_var = tk.StringVar(value=default_name)
        self.name_entry = ttk.Entry(top_left, textvariable=self.name_var, width=50)
        self.name_entry.grid(row=1, column=1, columnspan=2, sticky=tk.EW, pady=5, padx=5)

        # Barrier function controls
        self.use_barrier_var = tk.BooleanVar(value=False)
        barrier_check = ttk.Checkbutton(
            top_left,
            text="Use Barrier Functions",
            variable=self.use_barrier_var,
        )
        barrier_check.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5, padx=5)
        
        ttk.Label(top_left, text="Barrier μ:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.barrier_mu_var = tk.StringVar(value="0.01")
        barrier_mu_entry = ttk.Entry(top_left, textvariable=self.barrier_mu_var, width=15)
        barrier_mu_entry.grid(row=3, column=1, sticky=tk.W, pady=5, padx=5)
        
        ttk.Label(
            top_left,
            text="(keeps parameters away from boundaries)",
            font=("TkDefaultFont", 8),
            foreground="gray",
        ).grid(row=3, column=2, sticky=tk.W, padx=5)

        # Motor model type selector
        ttk.Label(top_left, text="Motor Model:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.motor_model_var = tk.StringVar(value="dc")
        motor_model_frame = ttk.Frame(top_left)
        motor_model_frame.grid(row=4, column=1, columnspan=2, sticky=tk.W, pady=5, padx=5)
        ttk.Radiobutton(
            motor_model_frame, text="DC Motor", variable=self.motor_model_var, value="dc",
            command=self._on_motor_model_changed
        ).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            motor_model_frame, text="Polynomial Map", variable=self.motor_model_var, value="polynomial",
            command=self._on_motor_model_changed
        ).pack(side=tk.LEFT, padx=5)
        
        # Fit DC from map option (only shown for polynomial model)
        self.fit_dc_from_map_var = tk.BooleanVar(value=False)
        self.fit_dc_check = ttk.Checkbutton(
            top_left,
            text="Fit DC parameters from map after optimization",
            variable=self.fit_dc_from_map_var,
        )
        self.fit_dc_check.grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=5, padx=5)
        self.fit_dc_check.grid_remove()  # Hidden by default

        top_left.columnconfigure(1, weight=1)

        # Optimization settings
        opt_frame = ttk.LabelFrame(left_frame, text="Training & Optimization", padding=10)
        opt_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Max Iterations
        ttk.Label(opt_frame, text="Max Iterations:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.max_iter_var = tk.StringVar(value="50")
        ttk.Entry(opt_frame, textvariable=self.max_iter_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Min/Max Segment Length
        ttk.Label(opt_frame, text="Min Segment Length:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.min_segment_length_var = tk.StringVar(value="50")
        ttk.Entry(opt_frame, textvariable=self.min_segment_length_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(opt_frame, text="Max Segment Length:").grid(row=1, column=2, sticky=tk.W, pady=2, padx=(10, 0))
        self.max_segment_length_var = tk.StringVar(value="100")
        ttk.Entry(opt_frame, textvariable=self.max_segment_length_var, width=10).grid(row=1, column=3, sticky=tk.W, padx=5)

        self.use_whole_trips_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            opt_frame,
            text="Use whole trips (no segmenting)",
            variable=self.use_whole_trips_var,
        ).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=2)

        self.filter_zero_speed_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            opt_frame,
            text="Filter zero-speed segments",
            variable=self.filter_zero_speed_var,
        ).grid(row=2, column=2, columnspan=2, sticky=tk.W, pady=2, padx=(10, 0))

        self.disable_segment_filter_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            opt_frame,
            text="Disable segment filtering (raw trips)",
            variable=self.disable_segment_filter_var,
        ).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # Batching settings
        ttk.Label(opt_frame, text="Segments per Batch:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.segments_per_batch_var = tk.StringVar(value="16")
        ttk.Entry(opt_frame, textvariable=self.segments_per_batch_var, width=10).grid(row=4, column=1, sticky=tk.W, padx=5)

        ttk.Label(opt_frame, text="Epochs:").grid(row=4, column=2, sticky=tk.W, pady=2, padx=(10, 0))
        self.epochs_var = tk.StringVar(value="1")
        ttk.Entry(opt_frame, textvariable=self.epochs_var, width=10).grid(row=4, column=3, sticky=tk.W, padx=5)

        self.random_batch_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            opt_frame,
            text="Random fixed-length batches",
            variable=self.random_batch_var,
        ).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=2)

        ttk.Label(opt_frame, text="Batch segment length:").grid(row=5, column=2, sticky=tk.W, pady=2, padx=(10, 0))
        self.random_segment_length_var = tk.StringVar(value="100")
        ttk.Entry(opt_frame, textvariable=self.random_segment_length_var, width=10).grid(row=5, column=3, sticky=tk.W, padx=5)

        ttk.Label(opt_frame, text="Random batches/epoch:").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.random_batches_per_epoch_var = tk.StringVar(value="10")
        ttk.Entry(opt_frame, textvariable=self.random_batches_per_epoch_var, width=10).grid(row=6, column=1, sticky=tk.W, padx=5)

        self.debug_batch_progress_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            opt_frame,
            text="Debug batch progress",
            variable=self.debug_batch_progress_var,
        ).grid(row=6, column=2, columnspan=1, sticky=tk.W, pady=2, padx=(10, 0))

        ttk.Label(opt_frame, text="Progress step (%):").grid(row=6, column=3, sticky=tk.W, pady=2)
        self.debug_batch_progress_step_var = tk.StringVar(value="10")
        ttk.Entry(opt_frame, textvariable=self.debug_batch_progress_step_var, width=6).grid(row=6, column=4, sticky=tk.W, padx=5)

        self.fixed_length_val_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            opt_frame,
            text="Fixed-length validation",
            variable=self.fixed_length_val_var,
        ).grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=2)

        ttk.Label(opt_frame, text="Validation fraction:").grid(row=7, column=2, sticky=tk.W, pady=2, padx=(10, 0))
        self.validation_fraction_var = tk.StringVar(value="0.1")
        ttk.Entry(opt_frame, textvariable=self.validation_fraction_var, width=10).grid(row=7, column=3, sticky=tk.W, padx=5)

        ttk.Label(opt_frame, text="Validation split seed:").grid(row=7, column=4, sticky=tk.W, pady=2, padx=(10, 0))
        self.validation_seed_var = tk.StringVar(value="")
        ttk.Entry(opt_frame, textvariable=self.validation_seed_var, width=10).grid(row=7, column=5, sticky=tk.W, padx=5)

        ttk.Label(opt_frame, text="Random batch max iter:").grid(row=8, column=0, sticky=tk.W, pady=2)
        self.random_batch_max_iter_var = tk.StringVar(value="5")
        ttk.Entry(opt_frame, textvariable=self.random_batch_max_iter_var, width=10).grid(row=8, column=1, sticky=tk.W, padx=5)
        
        # Loss Weights
        ttk.Label(opt_frame, text="Speed Loss Weight:").grid(row=9, column=0, sticky=tk.W, pady=2)
        self.speed_weight_var = tk.StringVar(value="1.0")
        ttk.Entry(opt_frame, textvariable=self.speed_weight_var, width=10).grid(row=9, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(opt_frame, text="Accel Loss Weight:").grid(row=9, column=2, sticky=tk.W, pady=2, padx=(10, 0))
        self.accel_weight_var = tk.StringVar(value="0.0")
        ttk.Entry(opt_frame, textvariable=self.accel_weight_var, width=10).grid(row=9, column=3, sticky=tk.W, padx=5)

        ttk.Label(opt_frame, text="Brake Loss Boost:").grid(row=10, column=0, sticky=tk.W, pady=2)
        self.brake_loss_boost_var = tk.StringVar(value="0.0")
        ttk.Entry(opt_frame, textvariable=self.brake_loss_boost_var, width=10).grid(row=10, column=1, sticky=tk.W, padx=5)

        ttk.Label(opt_frame, text="Full-stop loss cap (fraction):").grid(row=10, column=2, sticky=tk.W, pady=2, padx=(10, 0))
        self.full_stop_loss_cap_var = tk.StringVar(value="0.0")
        ttk.Entry(opt_frame, textvariable=self.full_stop_loss_cap_var, width=10).grid(row=10, column=3, sticky=tk.W, padx=5)

        self.mask_negative_speed_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            opt_frame,
            text="Mask loss for negative GT speeds",
            variable=self.mask_negative_speed_var,
        ).grid(row=10, column=6, columnspan=2, sticky=tk.W, pady=2, padx=(10, 0))

        # ExtendedPlant parity controls
        self.use_extended_plant_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            opt_frame,
            text="Use ExtendedPlant (RL)",
            variable=self.use_extended_plant_var,
        ).grid(row=11, column=0, columnspan=2, sticky=tk.W, pady=2)

        ttk.Label(opt_frame, text="Plant Substeps:").grid(row=11, column=2, sticky=tk.W, pady=2, padx=(10, 0))
        self.plant_substeps_var = tk.StringVar(value="2")
        ttk.Entry(opt_frame, textvariable=self.plant_substeps_var, width=10).grid(row=11, column=3, sticky=tk.W, padx=5)

        self.apply_lpf_to_fitting_data_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            opt_frame,
            text="Apply LPF to fitting data (accel: 2Hz, speed: 5Hz)",
            variable=self.apply_lpf_to_fitting_data_var,
        ).grid(row=13, column=4, columnspan=2, sticky=tk.W, pady=2, padx=(10, 0))

        # Actuator smoothing/deadband
        ttk.Label(opt_frame, text="Actuator Smoothing α:").grid(row=12, column=0, sticky=tk.W, pady=2)
        self.actuator_smoothing_var = tk.StringVar(value="0.2")
        ttk.Entry(opt_frame, textvariable=self.actuator_smoothing_var, width=10).grid(row=12, column=1, sticky=tk.W, padx=5)

        ttk.Label(opt_frame, text="Actuator Deadband (%):").grid(row=12, column=2, sticky=tk.W, pady=2, padx=(10, 0))
        self.actuator_deadband_var = tk.StringVar(value="1.0")
        ttk.Entry(opt_frame, textvariable=self.actuator_deadband_var, width=10).grid(row=12, column=3, sticky=tk.W, padx=5)

        ttk.Label(opt_frame, text="Max |Accel| (m/s²):").grid(row=13, column=0, sticky=tk.W, pady=2)
        self.max_accel_var = tk.StringVar(value=str(self.default_config.max_accel))
        ttk.Entry(opt_frame, textvariable=self.max_accel_var, width=10,).grid(row=13, column=1, sticky=tk.W, padx=5)

        self.use_param_scaling_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            opt_frame,
            text="Scale params (0-1)",
            variable=self.use_param_scaling_var,
        ).grid(row=13, column=2, columnspan=2, sticky=tk.W, pady=2)

        ttk.Label(opt_frame, text="Optimization Mode:").grid(row=14, column=0, sticky=tk.W, pady=2)
        self.optimization_mode_var = tk.StringVar(value="joint")
        self.optimization_mode_combo = ttk.Combobox(
            opt_frame,
            textvariable=self.optimization_mode_var,
            state="readonly",
            values=["joint", "sequential"],
            width=12,
        )
        self.optimization_mode_combo.grid(row=14, column=1, sticky=tk.W, padx=5)
        self.optimization_mode_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_optimization_mode_changed())

        ttk.Label(opt_frame, text="Optimizer:").grid(row=15, column=0, sticky=tk.W, pady=2)
        self.optimizer_method_var = tk.StringVar(value="L-BFGS-B")
        self.optimizer_method_combo = ttk.Combobox(
            opt_frame,
            textvariable=self.optimizer_method_var,
            state="readonly",
            values=["L-BFGS-B", "TNC", "SLSQP", "Powell"],
            width=10,
        )
        self.optimizer_method_combo.grid(row=15, column=1, sticky=tk.W, padx=5)

        self.overfit_longest_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            opt_frame,
            text="Overfit longest training trip first",
            variable=self.overfit_longest_var,
        ).grid(row=16, column=0, columnspan=2, sticky=tk.W, pady=2)

        ttk.Label(opt_frame, text="Overfit epochs:").grid(row=16, column=2, sticky=tk.W, pady=2, padx=(10, 0))
        self.overfit_longest_epochs_var = tk.StringVar(value="1")
        ttk.Entry(opt_frame, textvariable=self.overfit_longest_epochs_var, width=10).grid(row=16, column=3, sticky=tk.W, padx=5)

        ttk.Label(opt_frame, text="Phase Order:").grid(row=14, column=2, sticky=tk.W, pady=2, padx=(10, 0))
        self.phase_order_var = tk.StringVar(value="throttle -> brake")
        self.phase_order_combo = ttk.Combobox(
            opt_frame,
            textvariable=self.phase_order_var,
            state="readonly",
            values=["throttle -> brake", "brake -> throttle"],
            width=18,
        )
        self.phase_order_combo.grid(row=14, column=3, sticky=tk.W, padx=5)

        self._on_optimization_mode_changed()

        # Left panel: Parameters (scrollable)
        params_frame = ttk.LabelFrame(left_frame, text="Parameters", padding=10)
        params_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create scrollable canvas for parameters
        canvas = tk.Canvas(params_frame)
        scrollbar = ttk.Scrollbar(params_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Store parameter entry widgets and scrollable frame reference
        self.param_entries: Dict[str, Dict[str, tk.StringVar]] = {}
        self.scrollable_frame = scrollable_frame
        self.param_canvas = canvas

        # Create parameter input fields (will be updated when motor model changes)
        self._create_parameter_fields()

        # Left panel: Buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        self.update_sim_btn = ttk.Button(
            button_frame, text="Update Simulation", command=self._update_simulation
        )
        self.update_sim_btn.pack(side=tk.LEFT, padx=5)

        self.val_rmse_btn = ttk.Button(
            button_frame, text="Validation RMSE", command=self._compute_validation_rmse
        )
        self.val_rmse_btn.pack(side=tk.LEFT, padx=5)

        self.full_state_btn = ttk.Button(
            button_frame, text="Full-State Throttle", command=self._open_constant_throttle_full_state_window
        )
        self.full_state_btn.pack(side=tk.LEFT, padx=5)

        self.full_state_brake_btn = ttk.Button(
            button_frame, text="Full-State Brake", command=self._open_constant_brake_full_state_window
        )
        self.full_state_brake_btn.pack(side=tk.LEFT, padx=5)
        
        # Separator
        ttk.Separator(button_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Settings buttons
        self.save_settings_btn = ttk.Button(
            button_frame, text="Save as Default", command=self._save_settings
        )
        self.save_settings_btn.pack(side=tk.LEFT, padx=5)

        self.load_settings_btn = ttk.Button(
            button_frame, text="Load Default", command=self._load_settings
        )
        self.load_settings_btn.pack(side=tk.LEFT, padx=5)
        
        # Separator
        ttk.Separator(button_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        self.start_fitting_btn = ttk.Button(
            button_frame, text="Start Fitting", command=self._start_fitting
        )
        self.start_fitting_btn.pack(side=tk.LEFT, padx=5)

        self.abort_fitting_btn = ttk.Button(
            button_frame, text="Abort", command=self._abort_fitting, state=tk.DISABLED
        )
        self.abort_fitting_btn.pack(side=tk.LEFT, padx=5)

        self.advance_phase_btn = ttk.Button(
            button_frame, text="Advance Phase", command=self._advance_phase, state=tk.DISABLED
        )
        self.advance_phase_btn.pack(side=tk.LEFT, padx=5)

        # Progress bar and status
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = ttk.Label(button_frame, textvariable=self.progress_var)
        self.progress_label.pack(side=tk.LEFT, padx=10)

        self.progress_bar = ttk.Progressbar(
            button_frame, mode="indeterminate", length=200
        )
        self.progress_bar.pack(side=tk.LEFT, padx=5)

        # Right panel: Simulation plots
        plot_frame = ttk.LabelFrame(right_frame, text="Simulation Preview", padding=10)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add navigation toolbar frame first (at the top)
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)

        # Create matplotlib figure with 4x2 layout (4 rows, 2 columns)
        # Left column: Braking dynamics (1 plot)
        # Right column: Validation segment (4 plots with shared x-axis)
        self.fig = Figure(figsize=(16, 12), dpi=100)
        
        # Left column: Braking dynamics and throttle dynamics (3 plots)
        self.ax2 = self.fig.add_subplot(4, 2, 1)  # Brake speed
        self.ax1 = self.fig.add_subplot(4, 2, 3)  # Throttle speed
        self.ax1_power = self.fig.add_subplot(4, 2, 5)  # Throttle power
        self.ax1_current = self.fig.add_subplot(4, 2, 7)  # Throttle current
        
        # Right column: Validation segment (with shared x-axis for all 4 plots)
        self.ax3 = self.fig.add_subplot(4, 2, 2)
        self.ax4 = self.fig.add_subplot(4, 2, 4, sharex=self.ax3)
        self.ax5 = self.fig.add_subplot(4, 2, 6, sharex=self.ax3)
        self.ax6 = self.fig.add_subplot(4, 2, 8, sharex=self.ax3)

        self.ax2.set_title("Braking Dynamics (from 20 m/s)")
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("Speed (m/s)")
        self.ax2.grid(True, alpha=0.3)

        self.ax1.set_title("Throttle Dynamics - Speed (from 0 m/s)")
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Speed (m/s)")
        self.ax1.grid(True, alpha=0.3)

        self.ax1_power.set_title("Throttle Dynamics - Power")
        self.ax1_power.set_xlabel("Time (s)")
        self.ax1_power.set_ylabel("Power (W)")
        self.ax1_power.grid(True, alpha=0.3)

        self.ax1_current.set_title("Throttle Dynamics - Current")
        self.ax1_current.set_xlabel("Time (s)")
        self.ax1_current.set_ylabel("Current (A)")
        self.ax1_current.grid(True, alpha=0.3)

        self.ax3.set_title("Validation Segment Comparison")
        self.ax3.set_xlabel("Time (s)")
        self.ax3.set_ylabel("Speed (m/s)")
        self.ax3.grid(True, alpha=0.3)

        self.ax4.set_title("Validation Segment Actuations")
        self.ax4.set_xlabel("Time (s)")
        self.ax4.set_ylabel("Command (%)")
        self.ax4.grid(True, alpha=0.3)

        self.ax5.set_title("Validation Motor Power")
        self.ax5.set_xlabel("Time (s)")
        self.ax5.set_ylabel("Power (W)")
        self.ax5.grid(True, alpha=0.3)

        self.ax6.set_title("Validation Motor Current")
        self.ax6.set_xlabel("Time (s)")
        self.ax6.set_ylabel("Current (A)")
        self.ax6.grid(True, alpha=0.3)

        self.fig.tight_layout()

        # Embed in tkinter
        self.canvas_plot = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas_plot.draw()
        self.canvas_plot.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar for zoom/pan functionality
        self.toolbar = NavigationToolbar2Tk(self.canvas_plot, toolbar_frame)
        self.toolbar.update()

    def _on_motor_model_changed(self):
        """Called when motor model type changes - update parameter fields."""
        # Clear existing parameter fields
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.param_entries.clear()
        
        # Recreate parameter fields for new model
        self._create_parameter_fields()
        
        # Show/hide fit DC checkbox
        if self.motor_model_var.get() == "polynomial":
            self.fit_dc_check.grid()
        else:
            self.fit_dc_check.grid_remove()

    def _simulate_segment_with_state(self, params: np.ndarray, segment, fitter):
        """Simulate segment and extract motor state (current, power, voltage, i_limit)."""
        from simulation.dynamics import ExtendedPlant
        
        # Build ExtendedPlant params
        ext_params = fitter._build_extended_plant_params(params)
        plant = ExtendedPlant(ext_params)
        
        n = segment.length
        v_sim = np.zeros(n)
        current = np.zeros(n)
        power = np.zeros(n)
        voltage = np.zeros(n)
        i_limit = np.zeros(n)
        
        plant.reset(speed=segment.initial_speed, position=0.0)
        # Extract initial state
        state = plant.state
        v_sim[0] = state.speed
        current[0] = state.motor_current
        voltage[0] = state.V_cmd
        power[0] = voltage[0] * current[0]
        i_limit[0] = state.i_limit
        
        substeps = max(int(fitter.config.extended_plant_substeps), 1)
        for t in range(n - 1):
            throttle = float(segment.throttle[t]) / 100.0
            brake = float(segment.brake[t]) / 100.0
            grade = float(segment.grade[t])

            brake_active = brake * 100.0 > fitter.config.brake_deadband_pct
            if brake_active:
                action = -brake
            else:
                action = throttle
            action = float(np.clip(action, -1.0, 1.0))
            
            # Step the plant
            plant.step(
                action=action,
                grade_rad=grade,
                dt=segment.dt,
                substeps=substeps,
            )
            
            # Extract state after step
            state = plant.state
            v_sim[t + 1] = state.speed
            current[t + 1] = state.motor_current
            voltage[t + 1] = state.V_cmd
            power[t + 1] = voltage[t + 1] * current[t + 1]
            i_limit[t + 1] = state.i_limit
        
        # Get motor parameters for limits (for I_max and P_max display)
        motor_T_max = ext_params.motor.T_max if ext_params.motor.T_max is not None else 0.0
        motor_P_max = ext_params.motor.P_max if ext_params.motor.P_max is not None else 0.0
        motor_R = ext_params.motor.R
        motor_K = ext_params.motor.K_t
        
        # Calculate I_max from T_max (for display purposes)
        I_max = motor_T_max / motor_K if motor_K > 1e-6 else 0.0
        
        return v_sim, current, power, voltage, I_max, motor_P_max, i_limit

    def _on_optimization_mode_changed(self):
        """Enable/disable phase order selector based on optimization mode."""
        if self.optimization_mode_var.get() == "sequential":
            self.phase_order_combo.config(state="readonly")
        else:
            self.phase_order_combo.config(state=tk.DISABLED)

    def _on_example_segment_mode_changed(self):
        """Enable index entry only when Index mode is selected."""
        if self.example_segment_mode_var.get() == "Index":
            self.example_segment_index_entry.config(state=tk.NORMAL)
        else:
            self.example_segment_index_entry.config(state=tk.DISABLED)
        self.example_segment_key = None
    
    def _create_parameter_fields(self):
        """Create parameter input fields based on current motor model type."""
        # Get appropriate parameter groups
        if self.motor_model_var.get() == "polynomial":
            param_groups = PARAM_GROUPS_POLY
        else:
            param_groups = PARAM_GROUPS_DC
        
        row = 0
        for group_name, param_names in param_groups.items():
            # Group header
            ttk.Label(
                self.scrollable_frame, text=group_name, font=("TkDefaultFont", 10, "bold")
            ).grid(row=row, column=0, columnspan=4, sticky=tk.W, pady=(10, 5))
            row += 1

            # Column headers (only show once)
            if row == 1:
                ttk.Label(self.scrollable_frame, text="Parameter").grid(
                    row=row, column=0, sticky=tk.W, padx=5
                )
                ttk.Label(self.scrollable_frame, text="Initial").grid(
                    row=row, column=1, sticky=tk.W, padx=5
                )
                ttk.Label(self.scrollable_frame, text="Min").grid(
                    row=row, column=2, sticky=tk.W, padx=5
                )
                ttk.Label(self.scrollable_frame, text="Max").grid(
                    row=row, column=3, sticky=tk.W, padx=5
                )
                row += 1

            # Parameter rows
            for param_name in param_names:
                display_name, unit = PARAM_DISPLAY.get(param_name, (param_name, ""))
                label_text = f"{display_name}"
                if unit:
                    label_text += f" ({unit})"

                ttk.Label(self.scrollable_frame, text=label_text).grid(
                    row=row, column=0, sticky=tk.W, padx=5, pady=2
                )

                # Get default values
                try:
                    init_val = getattr(self.default_config, f"{param_name}_init")
                    bounds = getattr(self.default_config, f"{param_name}_bounds")
                    min_val, max_val = bounds
                except AttributeError:
                    # Use defaults if not found
                    init_val = 0.0
                    min_val, max_val = -100.0, 100.0

                if init_val is None:
                    init_val = 0.0

                # Entry variables
                init_var = tk.StringVar(value=str(init_val))
                min_var = tk.StringVar(value=str(min_val))
                max_var = tk.StringVar(value=str(max_val))

                # Entry widgets
                ttk.Entry(self.scrollable_frame, textvariable=init_var, width=12).grid(
                    row=row, column=1, padx=5, pady=2
                )
                ttk.Entry(self.scrollable_frame, textvariable=min_var, width=12).grid(
                    row=row, column=2, padx=5, pady=2
                )
                ttk.Entry(self.scrollable_frame, textvariable=max_var, width=12).grid(
                    row=row, column=3, padx=5, pady=2
                )

                self.param_entries[param_name] = {
                    "init": init_var,
                    "min": min_var,
                    "max": max_var,
                }

                row += 1
        
        # Update canvas scroll region
        self.scrollable_frame.update_idletasks()
        self.param_canvas.configure(scrollregion=self.param_canvas.bbox("all"))

    def _populate_datasets(self):
        """Scan data/processed for .pt files and populate dropdown."""
        datasets = []
        if self.data_dir.exists():
            for pt_file in self.data_dir.rglob("*.pt"):
                # Get relative path from data/processed
                try:
                    rel_path = pt_file.relative_to(self.data_dir.parent.parent)
                    datasets.append(str(rel_path))
                except ValueError:
                    # Skip if path resolution fails
                    continue

        datasets.sort()
        self.dataset_combo["values"] = datasets
        if datasets:
            self.dataset_combo.current(0)
        else:
            # Show warning if no datasets found
            self.dataset_combo.set("No datasets found")

    def _get_params_from_gui(self) -> Optional[Dict[str, Dict[str, float]]]:
        """Extract parameter values from GUI entries."""
        params = {}
        for param_name, entries in self.param_entries.items():
            try:
                init_val = float(entries["init"].get())
                min_val = float(entries["min"].get())
                max_val = float(entries["max"].get())

                if min_val > max_val:
                    messagebox.showerror(
                        "Validation Error",
                        f"Parameter {param_name}: min ({min_val}) must be <= max ({max_val})",
                    )
                    return None
                
                # If min == max, parameter is fixed (not optimized)
                # Ensure initial value matches the fixed value
                if min_val == max_val:
                    # Update initial value to match fixed value
                    entries["init"].set(str(min_val))
                    init_val = min_val

                params[param_name] = {
                    "init": init_val,
                    "min": min_val,
                    "max": max_val,
                }
            except ValueError:
                messagebox.showerror(
                    "Validation Error", f"Invalid number for parameter {param_name}"
                )
                return None

        return params

    def _update_simulation(self):
        """Update simulation plots with current parameter values."""
        params_dict = self._get_params_from_gui()
        if params_dict is None:
            return

        # Convert to parameter array for simulation
        motor_model_type = self.motor_model_var.get()
        param_array = self._params_dict_to_array(params_dict, use_init=True, motor_model_type=motor_model_type)

        # Clear existing lines to force recreation
        self.throttle_lines.clear()
        self.throttle_power_lines.clear()
        self.throttle_current_lines.clear()
        self.brake_lines.clear()
        self.val_lines.clear()
        self.val_act_lines.clear()
        self.val_power_lines.clear()
        self.val_current_lines.clear()
        self.ax1.clear()
        self.ax1_power.clear()
        self.ax1_current.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.ax5.clear()
        self.ax6.clear()
        self.validation_segment = None
        self.example_segment_key = None

        # Load a validation segment for preview if available
        self._ensure_validation_segment_loaded()

        # Update plots (will recreate lines)
        self._update_simulation_plots_callback(param_array)

    def _open_constant_throttle_full_state_window(self) -> None:
        """Open a detailed full-state viewer for constant-throttle simulation."""
        if self.motor_model_var.get() != "dc":
            messagebox.showwarning(
                "Full-State Viewer",
                "Full-state viewer requires DC motor model.\nPlease switch Motor Model to 'DC Motor'.",
            )
            return

        window = tk.Toplevel(self.root)
        window.title("Constant Throttle Full-State Explorer")
        window.geometry("1800x1100")

        controls = ttk.LabelFrame(window, text="Simulation Controls", padding=8)
        controls.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        ttk.Label(controls, text="Throttle (%):").grid(row=0, column=0, sticky=tk.W, padx=4)
        throttle_var = tk.DoubleVar(value=50.0)
        throttle_entry_var = tk.StringVar(value="50.0")

        throttle_scale = ttk.Scale(
            controls,
            from_=0.0,
            to=100.0,
            variable=throttle_var,
            orient=tk.HORIZONTAL,
            length=300,
        )
        throttle_scale.grid(row=0, column=1, sticky=tk.W, padx=4)

        throttle_entry = ttk.Entry(controls, textvariable=throttle_entry_var, width=8)
        throttle_entry.grid(row=0, column=2, sticky=tk.W, padx=4)

        ttk.Label(controls, text="Initial speed (m/s):").grid(row=0, column=3, sticky=tk.W, padx=(16, 4))
        initial_speed_var = tk.StringVar(value="0.0")
        ttk.Entry(controls, textvariable=initial_speed_var, width=8).grid(row=0, column=4, sticky=tk.W, padx=4)

        ttk.Label(controls, text="Road grade (deg):").grid(row=0, column=5, sticky=tk.W, padx=(16, 4))
        grade_deg_var = tk.StringVar(value="0.0")
        ttk.Entry(controls, textvariable=grade_deg_var, width=8).grid(row=0, column=6, sticky=tk.W, padx=4)

        ttk.Label(controls, text="dt (s):").grid(row=0, column=7, sticky=tk.W, padx=(16, 4))
        dt_var = tk.StringVar(value="0.1")
        ttk.Entry(controls, textvariable=dt_var, width=8).grid(row=0, column=8, sticky=tk.W, padx=4)

        ttk.Label(controls, text="Horizon (s):").grid(row=0, column=9, sticky=tk.W, padx=(16, 4))
        horizon_var = tk.StringVar(value="200.0")
        ttk.Entry(controls, textvariable=horizon_var, width=8).grid(row=0, column=10, sticky=tk.W, padx=4)

        status_var = tk.StringVar(value="Ready")
        ttk.Label(controls, textvariable=status_var).grid(row=1, column=0, columnspan=9, sticky=tk.W, padx=4, pady=(8, 0))

        plot_container = ttk.Frame(window)
        plot_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        toolbar_frame = ttk.Frame(plot_container)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)

        fig = Figure(figsize=(18, 12), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=plot_container)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        def _sync_scale_to_entry(_event=None):
            try:
                value = float(throttle_entry_var.get())
            except ValueError:
                return
            value = float(np.clip(value, 0.0, 100.0))
            throttle_var.set(value)
            throttle_entry_var.set(f"{value:.2f}")

        def _sync_entry_from_scale(*_args):
            throttle_entry_var.set(f"{float(throttle_var.get()):.2f}")

        throttle_var.trace_add("write", _sync_entry_from_scale)
        throttle_entry.bind("<Return>", _sync_scale_to_entry)
        throttle_entry.bind("<FocusOut>", _sync_scale_to_entry)

        def _run_full_state_simulation():
            try:
                params_dict = self._get_params_from_gui()
                if params_dict is None:
                    return

                config = self._create_fitter_config()
                if config.motor_model_type != "dc":
                    messagebox.showwarning(
                        "Full-State Viewer",
                        "Full-state viewer requires DC motor model.\nPlease switch Motor Model to 'DC Motor'.",
                    )
                    return

                fitter = VehicleParamFitter(config)
                param_array = self._params_dict_to_array(
                    params_dict,
                    use_init=True,
                    motor_model_type=config.motor_model_type,
                )

                throttle_pct = float(np.clip(float(throttle_var.get()) / 100.0, 0.0, 1.0))
                initial_speed = float(initial_speed_var.get())
                grade_deg = float(grade_deg_var.get())
                dt = max(float(dt_var.get()), 1e-3)
                horizon_s = max(float(horizon_var.get()), 0.1)

                status_var.set(f"Running full simulation ({horizon_s:.1f}s)...")
                window.update_idletasks()

                time_arr, signals = self._simulate_constant_throttle_full_state(
                    params=param_array,
                    fitter=fitter,
                    throttle_pct=throttle_pct,
                    horizon_s=horizon_s,
                    dt=dt,
                    initial_speed=initial_speed,
                    grade_deg=grade_deg,
                )

                self._plot_constant_throttle_full_state(
                    fig=fig,
                    time_arr=time_arr,
                    signals=signals,
                    throttle_pct=throttle_pct,
                    horizon_s=horizon_s,
                    dt=dt,
                    initial_speed=initial_speed,
                    grade_deg=grade_deg,
                )
                canvas.draw_idle()
                status_var.set(f"Done ({horizon_s:.1f}s)")
            except Exception as e:
                LOGGER.exception("Failed full-state simulation view")
                status_var.set("Error")
                messagebox.showerror("Full-State Viewer Error", f"Failed to run simulation:\n{e}")

        run_btn = ttk.Button(controls, text="Run Full Simulation", command=_run_full_state_simulation)
        run_btn.grid(row=1, column=9, columnspan=2, sticky=tk.E, padx=4, pady=(8, 0))

        _run_full_state_simulation()

    def _open_constant_brake_full_state_window(self) -> None:
        """Open a detailed full-state viewer for constant-brake simulation."""
        if self.motor_model_var.get() != "dc":
            messagebox.showwarning(
                "Full-State Viewer",
                "Full-state viewer requires DC motor model.\nPlease switch Motor Model to 'DC Motor'.",
            )
            return

        window = tk.Toplevel(self.root)
        window.title("Constant Brake Full-State Explorer")
        window.geometry("1800x1100")

        controls = ttk.LabelFrame(window, text="Simulation Controls", padding=8)
        controls.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        ttk.Label(controls, text="Brake (%):").grid(row=0, column=0, sticky=tk.W, padx=4)
        brake_var = tk.DoubleVar(value=50.0)
        brake_entry_var = tk.StringVar(value="50.0")

        brake_scale = ttk.Scale(
            controls,
            from_=0.0,
            to=100.0,
            variable=brake_var,
            orient=tk.HORIZONTAL,
            length=300,
        )
        brake_scale.grid(row=0, column=1, sticky=tk.W, padx=4)

        brake_entry = ttk.Entry(controls, textvariable=brake_entry_var, width=8)
        brake_entry.grid(row=0, column=2, sticky=tk.W, padx=4)

        ttk.Label(controls, text="Initial speed (m/s):").grid(row=0, column=3, sticky=tk.W, padx=(16, 4))
        initial_speed_var = tk.StringVar(value="20.0")
        ttk.Entry(controls, textvariable=initial_speed_var, width=8).grid(row=0, column=4, sticky=tk.W, padx=4)

        ttk.Label(controls, text="Road grade (deg):").grid(row=0, column=5, sticky=tk.W, padx=(16, 4))
        grade_deg_var = tk.StringVar(value="0.0")
        ttk.Entry(controls, textvariable=grade_deg_var, width=8).grid(row=0, column=6, sticky=tk.W, padx=4)

        ttk.Label(controls, text="dt (s):").grid(row=0, column=7, sticky=tk.W, padx=(16, 4))
        dt_var = tk.StringVar(value="0.1")
        ttk.Entry(controls, textvariable=dt_var, width=8).grid(row=0, column=8, sticky=tk.W, padx=4)

        ttk.Label(controls, text="Horizon (s):").grid(row=0, column=9, sticky=tk.W, padx=(16, 4))
        horizon_var = tk.StringVar(value="120.0")
        ttk.Entry(controls, textvariable=horizon_var, width=8).grid(row=0, column=10, sticky=tk.W, padx=4)

        status_var = tk.StringVar(value="Ready")
        ttk.Label(controls, textvariable=status_var).grid(row=1, column=0, columnspan=9, sticky=tk.W, padx=4, pady=(8, 0))

        plot_container = ttk.Frame(window)
        plot_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        toolbar_frame = ttk.Frame(plot_container)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)

        fig = Figure(figsize=(18, 12), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=plot_container)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        def _sync_scale_to_entry(_event=None):
            try:
                value = float(brake_entry_var.get())
            except ValueError:
                return
            value = float(np.clip(value, 0.0, 100.0))
            brake_var.set(value)
            brake_entry_var.set(f"{value:.2f}")

        def _sync_entry_from_scale(*_args):
            brake_entry_var.set(f"{float(brake_var.get()):.2f}")

        brake_var.trace_add("write", _sync_entry_from_scale)
        brake_entry.bind("<Return>", _sync_scale_to_entry)
        brake_entry.bind("<FocusOut>", _sync_scale_to_entry)

        def _run_full_state_simulation():
            try:
                params_dict = self._get_params_from_gui()
                if params_dict is None:
                    return

                config = self._create_fitter_config()
                if config.motor_model_type != "dc":
                    messagebox.showwarning(
                        "Full-State Viewer",
                        "Full-state viewer requires DC motor model.\nPlease switch Motor Model to 'DC Motor'.",
                    )
                    return

                fitter = VehicleParamFitter(config)
                param_array = self._params_dict_to_array(
                    params_dict,
                    use_init=True,
                    motor_model_type=config.motor_model_type,
                )

                brake_pct = float(np.clip(float(brake_var.get()) / 100.0, 0.0, 1.0))
                initial_speed = float(initial_speed_var.get())
                grade_deg = float(grade_deg_var.get())
                dt = max(float(dt_var.get()), 1e-3)
                horizon_s = max(float(horizon_var.get()), 0.1)

                status_var.set(f"Running full simulation ({horizon_s:.1f}s)...")
                window.update_idletasks()

                time_arr, signals = self._simulate_constant_brake_full_state(
                    params=param_array,
                    fitter=fitter,
                    brake_pct=brake_pct,
                    horizon_s=horizon_s,
                    dt=dt,
                    initial_speed=initial_speed,
                    grade_deg=grade_deg,
                )

                self._plot_constant_action_full_state(
                    fig=fig,
                    time_arr=time_arr,
                    signals=signals,
                    command_name="brake",
                    command_pct=brake_pct,
                    horizon_s=horizon_s,
                    dt=dt,
                    initial_speed=initial_speed,
                    grade_deg=grade_deg,
                )
                canvas.draw_idle()
                status_var.set(f"Done ({horizon_s:.1f}s)")
            except Exception as e:
                LOGGER.exception("Failed full-state brake simulation view")
                status_var.set("Error")
                messagebox.showerror("Full-State Viewer Error", f"Failed to run simulation:\n{e}")

        run_btn = ttk.Button(controls, text="Run Full Simulation", command=_run_full_state_simulation)
        run_btn.grid(row=1, column=9, columnspan=2, sticky=tk.E, padx=4, pady=(8, 0))

        _run_full_state_simulation()

    def _simulate_constant_action_full_state(
        self,
        params: np.ndarray,
        fitter: VehicleParamFitter,
        action: float,
        horizon_s: float,
        dt: float,
        initial_speed: float,
        grade_deg: float,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Run detailed full-state simulation under a constant action for the requested horizon."""
        from simulation.dynamics import ExtendedPlant

        n_steps = max(int(horizon_s / max(dt, 1e-9)), 2)
        time_arr = np.arange(n_steps, dtype=np.float64) * dt

        ext_params = fitter._build_extended_plant_params(params)
        plant = ExtendedPlant(ext_params)
        plant.reset(speed=initial_speed, position=0.0)

        state_keys = [
            "speed",
            "position",
            "acceleration",
            "wheel_speed",
            "brake_torque",
            "slip_ratio",
            "action",
            "motor_current",
            "motor_omega",
            "back_emf_voltage",
            "V_cmd",
            "i_limit",
            "drive_torque",
            "tire_force",
            "drag_force",
            "rolling_force",
            "grade_force",
            "net_force",
            "held_by_brakes",
            "coupling_enabled",
        ]
        signals = {key: np.zeros(n_steps, dtype=np.float64) for key in state_keys}

        def _write_state(idx: int, state) -> None:
            signals["speed"][idx] = state.speed
            signals["position"][idx] = state.position
            signals["acceleration"][idx] = state.acceleration
            signals["wheel_speed"][idx] = state.wheel_speed
            signals["brake_torque"][idx] = state.brake_torque
            signals["slip_ratio"][idx] = state.slip_ratio
            signals["action"][idx] = state.action
            signals["motor_current"][idx] = state.motor_current
            signals["motor_omega"][idx] = state.motor_omega
            signals["back_emf_voltage"][idx] = state.back_emf_voltage
            signals["V_cmd"][idx] = state.V_cmd
            signals["i_limit"][idx] = state.i_limit
            signals["drive_torque"][idx] = state.drive_torque
            signals["tire_force"][idx] = state.tire_force
            signals["drag_force"][idx] = state.drag_force
            signals["rolling_force"][idx] = state.rolling_force
            signals["grade_force"][idx] = state.grade_force
            signals["net_force"][idx] = state.net_force
            signals["held_by_brakes"][idx] = 1.0 if state.held_by_brakes else 0.0
            signals["coupling_enabled"][idx] = 1.0 if state.coupling_enabled else 0.0

        _write_state(0, plant.state)

        action = float(np.clip(action, -1.0, 1.0))
        grade_rad = np.deg2rad(grade_deg)
        substeps = max(int(fitter.config.extended_plant_substeps), 1)

        for t in range(n_steps - 1):
            state = plant.step(action=action, grade_rad=grade_rad, dt=dt, substeps=substeps)
            _write_state(t + 1, state)

        wheel_radius = max(ext_params.wheel.radius, 1e-9)
        signals["wheel_omega"] = signals["wheel_speed"] / wheel_radius
        signals["electrical_power"] = signals["V_cmd"] * signals["motor_current"]
        signals["tractive_power"] = signals["tire_force"] * signals["speed"]
        signals["drag_power"] = signals["drag_force"] * np.abs(signals["speed"])
        signals["rolling_power"] = signals["rolling_force"] * np.abs(signals["speed"])
        signals["grade_power"] = signals["grade_force"] * signals["speed"]
        signals["net_power"] = signals["net_force"] * signals["speed"]
        i_limit_static = (
            (ext_params.motor.T_max / max(ext_params.motor.K_t, 1e-9))
            if ext_params.motor.T_max is not None
            else (ext_params.motor.V_max / max(ext_params.motor.R, 1e-9))
        )
        signals["i_limit_static"] = np.full(n_steps, float(i_limit_static), dtype=np.float64)

        return time_arr, signals

    def _simulate_constant_throttle_full_state(
        self,
        params: np.ndarray,
        fitter: VehicleParamFitter,
        throttle_pct: float,
        horizon_s: float,
        dt: float,
        initial_speed: float,
        grade_deg: float,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Run detailed full-state simulation under constant throttle for the requested horizon."""
        return self._simulate_constant_action_full_state(
            params=params,
            fitter=fitter,
            action=float(np.clip(throttle_pct, 0.0, 1.0)),
            horizon_s=horizon_s,
            dt=dt,
            initial_speed=initial_speed,
            grade_deg=grade_deg,
        )

    def _simulate_constant_brake_full_state(
        self,
        params: np.ndarray,
        fitter: VehicleParamFitter,
        brake_pct: float,
        horizon_s: float,
        dt: float,
        initial_speed: float,
        grade_deg: float,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Run detailed full-state simulation under constant brake for the requested horizon."""
        return self._simulate_constant_action_full_state(
            params=params,
            fitter=fitter,
            action=-float(np.clip(brake_pct, 0.0, 1.0)),
            horizon_s=horizon_s,
            dt=dt,
            initial_speed=initial_speed,
            grade_deg=grade_deg,
        )

    def _plot_constant_action_full_state(
        self,
        fig: Figure,
        time_arr: np.ndarray,
        signals: Dict[str, np.ndarray],
        command_name: str,
        command_pct: float,
        horizon_s: float,
        dt: float,
        initial_speed: float,
        grade_deg: float,
    ) -> None:
        """Render detailed full-state plots for constant-action simulation."""
        fig.clear()
        axes = fig.subplots(5, 2, sharex=True)
        ax = axes.flatten()

        ax[0].plot(time_arr, signals["speed"], label="Vehicle Speed")
        ax[0].plot(time_arr, signals["wheel_speed"], label="Wheel Linear Speed", alpha=0.8)
        ax[0].set_ylabel("Speed (m/s)")
        ax[0].set_title("Kinematics")
        ax[0].legend(fontsize=8)
        ax[0].grid(True, alpha=0.3)

        ax[1].plot(time_arr, signals["position"], label="Position", color="tab:purple")
        ax[1].set_ylabel("Position (m)")
        ax[1].set_title("Position")
        ax[1].legend(fontsize=8)
        ax[1].grid(True, alpha=0.3)

        ax[2].plot(time_arr, signals["acceleration"], label="Acceleration", color="tab:green")
        ax[2].set_ylabel("Acceleration (m/s²)")
        ax[2].set_title("Acceleration")
        ax[2].legend(fontsize=8)
        ax[2].grid(True, alpha=0.3)

        ax[3].plot(time_arr, signals["motor_current"], label="Motor Current", color="tab:green")
        ax[3].plot(time_arr, signals["i_limit"], label="Dynamic Current Limit", color="tab:red", linestyle="--")
        if "i_limit_static" in signals:
            ax[3].plot(time_arr, signals["i_limit_static"], label="Static Current Limit", color="tab:purple", linestyle=":")
        ax[3].set_ylabel("Current (A)")
        ax[3].set_title("Motor Current Limits")
        ax[3].legend(fontsize=8)
        ax[3].grid(True, alpha=0.3)

        ax[4].plot(time_arr, signals["V_cmd"], label="Command Voltage", color="tab:blue")
        ax[4].plot(time_arr, signals["back_emf_voltage"], label="Back-EMF Voltage", color="tab:orange")
        ax[4].set_ylabel("Voltage (V)")
        ax[4].set_title("Electrical Voltages")
        ax[4].legend(fontsize=8)
        ax[4].grid(True, alpha=0.3)

        ax[5].plot(time_arr, signals["motor_omega"], label="Motor ω", color="tab:cyan")
        ax[5].plot(time_arr, signals["wheel_omega"], label="Wheel ω", color="tab:brown")
        ax[5].set_ylabel("Angular Speed")
        ax[5].set_title("Rotational Speeds")
        ax[5].legend(fontsize=8)
        ax[5].grid(True, alpha=0.3)

        ax[6].plot(time_arr, signals["drive_torque"], label="Drive Torque (Wheel)", color="tab:cyan")
        ax[6].plot(time_arr, signals["brake_torque"], label="Brake Torque", color="tab:red")
        ax[6].set_ylabel("Torque (Nm)")
        ax[6].set_title("Torque Channels (Wheel Side)")
        ax[6].legend(fontsize=8)
        ax[6].grid(True, alpha=0.3)

        ax[7].plot(time_arr, signals["tire_force"], label="Tire")
        ax[7].plot(time_arr, signals["drag_force"], label="Drag")
        ax[7].plot(time_arr, signals["rolling_force"], label="Rolling")
        ax[7].plot(time_arr, signals["grade_force"], label="Grade")
        ax[7].plot(time_arr, signals["net_force"], label="Net", linewidth=2, color="k")
        ax[7].set_ylabel("Force (N)")
        ax[7].set_title("Longitudinal Forces")
        ax[7].legend(fontsize=8, ncol=3)
        ax[7].grid(True, alpha=0.3)

        ax[8].plot(time_arr, signals["electrical_power"], label="Electrical P", color="tab:blue")
        ax[8].plot(time_arr, signals["tractive_power"], label="Traction P", color="tab:green")
        ax[8].plot(time_arr, signals["drag_power"], label="Drag Loss P", color="tab:red")
        ax[8].plot(time_arr, signals["rolling_power"], label="Rolling Loss P", color="tab:orange")
        ax[8].plot(time_arr, signals["grade_power"], label="Grade P", color="tab:brown")
        ax[8].plot(time_arr, signals["net_power"], label="Net P", color="k", linewidth=2)
        ax[8].set_ylabel("Power (W)")
        ax[8].set_title("Power Flow")
        ax[8].legend(fontsize=8, ncol=3)
        ax[8].grid(True, alpha=0.3)

        final_speed = float(signals["speed"][-1])
        max_speed = float(np.max(signals["speed"]))
        distance = float(signals["position"][-1])
        peak_current = float(np.max(signals["motor_current"]))
        peak_drive_torque = float(np.max(signals["drive_torque"]))
        peak_brake_torque = float(np.max(signals["brake_torque"]))
        max_force = float(np.max(np.abs(signals["net_force"])))
        summary = (
            f"Summary\n"
            f"{command_name.capitalize()} command: {command_pct * 100.0:.2f}%\n"
            f"Final speed: {final_speed:.3f} m/s\n"
            f"Max speed: {max_speed:.3f} m/s\n"
            f"Distance: {distance:.1f} m\n"
            f"Peak current: {peak_current:.2f} A\n"
            f"Peak drive torque: {peak_drive_torque:.2f} Nm\n"
            f"Peak brake torque: {peak_brake_torque:.2f} Nm\n"
            f"Max |net force|: {max_force:.2f} N\n"
            f"Horizon: {horizon_s:.1f} s\n"
            f"dt: {dt:.3f} s\n"
            f"Initial speed: {initial_speed:.3f} m/s\n"
            f"Road grade: {grade_deg:.3f} deg"
        )
        ax[9].axis("off")
        ax[9].text(0.02, 0.98, summary, va="top", ha="left", fontsize=10)

        ax[8].set_xlabel("Time (s)")
        fig.suptitle(
            f"Constant-{command_name.capitalize()} Full-State Simulation ({horizon_s:.1f} s) | "
            f"{command_name}={command_pct * 100.0:.2f}%",
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))

    def _plot_constant_throttle_full_state(
        self,
        fig: Figure,
        time_arr: np.ndarray,
        signals: Dict[str, np.ndarray],
        throttle_pct: float,
        horizon_s: float,
        dt: float,
        initial_speed: float,
        grade_deg: float,
    ) -> None:
        """Render detailed full-state plots for constant-throttle simulation."""
        self._plot_constant_action_full_state(
            fig=fig,
            time_arr=time_arr,
            signals=signals,
            command_name="throttle",
            command_pct=float(np.clip(throttle_pct, 0.0, 1.0)),
            horizon_s=horizon_s,
            dt=dt,
            initial_speed=initial_speed,
            grade_deg=grade_deg,
        )

    def _compute_validation_rmse(self) -> None:
        """Open validation analysis window with detailed plots and statistics."""
        try:
            params_dict = self._get_params_from_gui()
            if params_dict is None:
                return

            motor_model_type = self.motor_model_var.get()
            param_array = self._params_dict_to_array(
                params_dict, use_init=True, motor_model_type=motor_model_type
            )

            config = self._create_fitter_config()
            
            # Reuse existing fitter if available and config matches (faster - segments already loaded)
            # Otherwise create new fitter
            if (self.current_fitter is not None 
                and self.current_fitter._trips is not None 
                and self.current_fitter.val_segments):
                # Reuse existing fitter and segments (much faster)
                fitter = self.current_fitter
                val_segments = fitter.val_segments
            else:
                # Create new fitter and load segments
                fitter = VehicleParamFitter(config)
                dataset_rel = self.dataset_var.get()
                if not dataset_rel:
                    messagebox.showerror("Validation Error", "Please select a dataset")
                    return
                dataset_path = Path(__file__).parent.parent / dataset_rel
                trips = fitter.load_trip_data(dataset_path)
                if not trips:
                    messagebox.showerror("Validation Error", "No trips found in dataset")
                    return
                dt = fitter._estimate_dt(trips)
                segments = fitter._create_segments(trips, dt)
                if not segments:
                    messagebox.showerror("Validation Error", "No valid segments created")
                    return

                # Split into train/val using same method as optimization
                rng = np.random.default_rng(config.validation_split_seed if config.validation_split_seed is not None else 42)
                n_total = len(segments)
                n_val = max(1, int(n_total * config.validation_fraction))
                n_val = min(n_val, n_total - 1) if n_total > 1 else 0
                n_train = n_total - n_val
                perm = rng.permutation(n_total)
                val_segments = [segments[i] for i in perm[n_train:]]

            if not val_segments:
                messagebox.showerror("Validation Error", "No validation segments available")
                return

            # Open validation analysis menu
            self._show_validation_analysis_window(fitter, param_array, val_segments)
        except Exception:
            LOGGER.exception("Failed to compute validation RMSE")
            messagebox.showerror("Validation Error", "Failed to compute validation RMSE")

    def _show_validation_analysis_window(
        self,
        fitter: VehicleParamFitter,
        param_array: np.ndarray,
        val_segments: list,
    ) -> None:
        """Open a menu window to select which validation segments to visualize."""
        # Get dt from segments
        dt = val_segments[0].dt if val_segments else 0.1

        # Segment lengths in seconds
        segment_lengths_sec = [5.0, 10.0, 15.0, 25.0, 50.0]
        segment_lengths_timesteps = [int(length_sec / dt) for length_sec in segment_lengths_sec]

        # Precompute analysis data for each segment length
        analysis_data = {}
        for length_sec, length_ts in zip(segment_lengths_sec, segment_lengths_timesteps):
            rng = np.random.default_rng(42)
            fixed_segments = fitter._sample_fixed_length_batch(
                val_segments,
                min(100, len(val_segments) * 2),
                length_ts,
                rng,
            )
            if not fixed_segments:
                continue

            segment_errors = []
            for seg in fixed_segments:
                # Completely filter out segments where more than 10% of samples are near zero speed
                zero_speed_eps = 0.1  # m/s threshold for zero speed
                zero_frac = float(np.sum(np.abs(seg.speed) < zero_speed_eps)) / max(len(seg.speed), 1)
                if zero_frac > 0.10:
                    continue

                v_sim, a_sim = fitter._simulate_segment(param_array, seg)
                v_err = v_sim - seg.speed
                a_err = a_sim - seg.acceleration
                v_rmse = np.sqrt(np.mean(v_err ** 2))
                a_rmse = np.sqrt(np.mean(a_err ** 2))
                v_mae = np.mean(np.abs(v_err))
                a_mae = np.mean(np.abs(a_err))
                segment_errors.append(
                    {
                        "segment": seg,
                        "v_sim": v_sim,
                        "a_sim": a_sim,
                        "v_err": v_err,
                        "a_err": a_err,
                        "v_rmse": v_rmse,
                        "a_rmse": a_rmse,
                        "v_mae": v_mae,
                        "a_mae": a_mae,
                    }
                )

            if not segment_errors:
                continue

            # Overall statistics
            all_v_err = np.concatenate([se["v_err"] for se in segment_errors])
            all_a_err = np.concatenate([se["a_err"] for se in segment_errors])
            all_v_gt = np.concatenate([se["segment"].speed for se in segment_errors])
            all_grade = np.concatenate([se["segment"].grade for se in segment_errors])

            overall_v_rmse = float(np.sqrt(np.mean(all_v_err ** 2)))
            overall_a_rmse = float(np.sqrt(np.mean(all_a_err ** 2)))
            overall_v_mae = float(np.mean(np.abs(all_v_err)))
            overall_a_mae = float(np.mean(np.abs(all_a_err)))
            overall_v_std = float(np.std(all_v_err))
            overall_a_std = float(np.std(all_a_err))

            # Speed range statistics
            speed_ranges = [
                (0.0, 2.0, "0-2 m/s"),
                (2.0, 5.0, "2-5 m/s"),
                (5.0, 10.0, "5-10 m/s"),
                (10.0, 20.0, "10-20 m/s"),
                (20.0, np.inf, "20+ m/s"),
            ]
            speed_range_stats = []
            for v_min, v_max, label in speed_ranges:
                mask = (all_v_gt >= v_min) & (all_v_gt < v_max)
                if np.sum(mask) > 0:
                    v_err_range = all_v_err[mask]
                    a_err_range = all_a_err[mask]
                    speed_range_stats.append(
                        {
                            "label": label,
                            "count": int(np.sum(mask)),
                            "v_rmse": float(np.sqrt(np.mean(v_err_range ** 2))),
                            "a_rmse": float(np.sqrt(np.mean(a_err_range ** 2))),
                            "v_mae": float(np.mean(np.abs(v_err_range))),
                            "a_mae": float(np.mean(np.abs(a_err_range))),
                            "v_std": float(np.std(v_err_range)),
                            "a_std": float(np.std(a_err_range)),
                        }
                    )

            # Collect per-segment metrics for scatter plots
            segment_v_rmse = [se["v_rmse"] for se in segment_errors]
            segment_a_rmse = [se["a_rmse"] for se in segment_errors]
            segment_avg_grade = [np.mean(se["segment"].grade) for se in segment_errors]
            
            analysis_data[length_sec] = {
                "length_ts": length_ts,
                "segment_errors": segment_errors,
                "overall": {
                    "v_rmse": overall_v_rmse,
                    "a_rmse": overall_a_rmse,
                    "v_mae": overall_v_mae,
                    "a_mae": overall_a_mae,
                    "v_std": overall_v_std,
                    "a_std": overall_a_std,
                },
                "ranges": speed_range_stats,
                "segment_v_rmse": segment_v_rmse,
                "segment_a_rmse": segment_a_rmse,
                "segment_avg_grade": segment_avg_grade,
            }

        if not analysis_data:
            messagebox.showerror("Validation Analysis", "No valid segments for analysis.")
            return

        # Helper to choose a segment based on metric and mode
        # All segments in analysis_data are already filtered to have <=10% zero-speed samples
        def select_segment(length_sec: float, metric: str, mode: str, enforce_zero_speed_filter: bool = False):
            data = analysis_data.get(length_sec)
            if data is None:
                return None
            segment_list = data["segment_errors"]
            if not segment_list:
                return None

            segment_list = sorted(segment_list, key=lambda se: se[metric])
            if mode == "best":
                return segment_list[0]
            if mode == "worst":
                return segment_list[-1]
            if mode == "median":
                return segment_list[len(segment_list) // 2]
            return None

        # Create menu window
        menu = tk.Toplevel(self.root)
        menu.title("Validation Analysis Menu")
        menu.geometry("1000x900")

        info_label = tk.Label(
            menu,
            text="Select which validation segment to display.\n"
            "Each selection opens a new window with speed, acceleration, actuation, and road grade for a single segment.\n"
            "All subplots share the same time axis.",
            justify="left",
            font=("Arial", 12),
        )
        info_label.pack(side=tk.TOP, anchor="w", padx=10, pady=10)
        
        # Add summary table button at the top
        summary_btn = ttk.Button(
            menu,
            text="View Summary Table",
            command=lambda: self._show_validation_summary_table(analysis_data),
            style="Large.TButton"
        )
        summary_btn.pack(side=tk.TOP, padx=10, pady=10)

        container = ttk.Frame(menu)
        container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure larger button font style
        style = ttk.Style()
        style.configure("Large.TButton", font=("Arial", 11))

        # Create controls per segment length
        for length_sec in segment_lengths_sec:
            if length_sec not in analysis_data:
                continue
            data = analysis_data[length_sec]
            # Create frame with larger title label
            frame = ttk.Frame(container)
            frame.pack(side=tk.TOP, fill=tk.X, expand=False, padx=5, pady=5)
            title_label = tk.Label(frame, text=f"Segment length: {length_sec:.1f} s", 
                                  font=("Arial", 12, "bold"))
            title_label.pack(side=tk.TOP, anchor="w", padx=5, pady=5)
            content_frame = ttk.Frame(frame)
            content_frame.pack(side=tk.TOP, fill=tk.X, expand=False)

            overall = data["overall"]
            stats_lines = [
                f"Overall speed RMSE: {overall['v_rmse']:.4f} m/s, MAE: {overall['v_mae']:.4f} m/s",
                f"Overall accel RMSE: {overall['a_rmse']:.4f} m/s², MAE: {overall['a_mae']:.4f} m/s²",
            ]
            stats_label = tk.Label(content_frame, text="\n".join(stats_lines), justify="left", font=("Courier", 12))
            stats_label.pack(side=tk.TOP, anchor="w", padx=5, pady=5)

            btn_row = ttk.Frame(content_frame)
            btn_row.pack(side=tk.TOP, fill=tk.X, pady=5)

            def make_button(text: str, metric: str, mode: str, enforce_zero_speed_filter: bool = False):
                def _on_click(length=length_sec, m=metric, md=mode, zf=enforce_zero_speed_filter):
                    seg_data = select_segment(length, m, md, enforce_zero_speed_filter=zf)
                    if seg_data is None:
                        messagebox.showerror("Validation Analysis", "No suitable segment found for selection.")
                        return
                    self._show_validation_segment_window(dt, length, seg_data)

                btn = ttk.Button(btn_row, text=text, command=_on_click, style="Large.TButton")
                return btn

            # Buttons: best/worst/median by speed RMSE, best/worst/median by accel RMSE
            make_button("Best speed RMSE", "v_rmse", "best", enforce_zero_speed_filter=True).pack(
                side=tk.LEFT, padx=3
            )
            make_button("Worst speed RMSE", "v_rmse", "worst").pack(side=tk.LEFT, padx=3)
            make_button("Median speed RMSE", "v_rmse", "median").pack(side=tk.LEFT, padx=3)
            make_button("Best accel RMSE", "a_rmse", "best").pack(side=tk.LEFT, padx=8)
            make_button("Worst accel RMSE", "a_rmse", "worst").pack(side=tk.LEFT, padx=3)
            make_button("Median accel RMSE", "a_rmse", "median").pack(side=tk.LEFT, padx=3)

    def _show_validation_segment_window(
        self,
        dt: float,
        length_sec: float,
        seg_data: Dict[str, object],
    ) -> None:
        """Open a window showing speed, acceleration, and actuation for a single segment."""
        seg = seg_data["segment"]
        t = np.arange(len(seg.speed)) * dt

        # Low-pass filter GT speed (slightly - higher cutoff than acceleration)
        if len(seg.speed) > 3:
            nyquist = 0.5 / dt
            cutoff = 5.0  # Higher cutoff for speed (less filtering)
            normal_cutoff = cutoff / nyquist
            if normal_cutoff < 1.0:
                b, a = signal.butter(2, normal_cutoff, btype="low")
                v_gt_filtered = signal.filtfilt(b, a, seg.speed)
            else:
                v_gt_filtered = seg.speed
        else:
            v_gt_filtered = seg.speed

        # Low-pass filter GT acceleration
        if len(seg.acceleration) > 3:
            nyquist = 0.5 / dt
            cutoff = 2.0
            normal_cutoff = cutoff / nyquist
            if normal_cutoff < 1.0:
                b, a = signal.butter(2, normal_cutoff, btype="low")
                a_gt_filtered = signal.filtfilt(b, a, seg.acceleration)
            else:
                a_gt_filtered = seg.acceleration
        else:
            a_gt_filtered = seg.acceleration

        # Create figure with a single column of subplots sharing x-axis
        fig = Figure(figsize=(12, 12))
        fig.suptitle(
            f"Validation Segment ({length_sec:.1f} s) - "
            f"v_RMSE={seg_data['v_rmse']:.4f} m/s, v_MAE={seg_data['v_mae']:.4f} m/s, "
            f"a_RMSE={seg_data['a_rmse']:.4f} m/s², a_MAE={seg_data['a_mae']:.4f} m/s²",
            fontsize=12,
        )

        ax_speed = fig.add_subplot(4, 1, 1)
        ax_accel = fig.add_subplot(4, 1, 2, sharex=ax_speed)
        ax_act = fig.add_subplot(4, 1, 3, sharex=ax_speed)
        ax_grade = fig.add_subplot(4, 1, 4, sharex=ax_speed)

        # Speed
        ax_speed.plot(t, v_gt_filtered, "b-", label="GT Speed (filtered)", linewidth=2)
        ax_speed.plot(t, seg_data["v_sim"], "r--", label="Sim Speed", linewidth=2)
        ax_speed.set_ylabel("Speed (m/s)")
        ax_speed.legend(loc="best", fontsize=9)
        ax_speed.grid(True, alpha=0.3)
        ax_speed.set_title(f"Speed - RMSE: {seg_data['v_rmse']:.4f} m/s, MAE: {seg_data['v_mae']:.4f} m/s", fontsize=10)

        # Acceleration
        ax_accel.plot(t, a_gt_filtered, "g-", label="GT Accel (filtered)", linewidth=2)
        ax_accel.plot(t, seg_data["a_sim"], "m--", label="Sim Accel", linewidth=2)
        ax_accel.set_ylabel("Acceleration (m/s²)")
        ax_accel.legend(loc="best", fontsize=9)
        ax_accel.grid(True, alpha=0.3)
        ax_accel.set_title(f"Acceleration - RMSE: {seg_data['a_rmse']:.4f} m/s², MAE: {seg_data['a_mae']:.4f} m/s²", fontsize=10)

        # Actuation
        ax_act.plot(t, seg.throttle, "orange", label="Throttle (%)", linewidth=2)
        ax_act.plot(t, seg.brake, "red", label="Brake (%)", linewidth=2)
        ax_act.set_ylabel("Actuation (%)")
        ax_act.set_xlabel("Time (s)")
        ax_act.set_ylim([0, 100])
        ax_act.legend(loc="best", fontsize=9)
        ax_act.grid(True, alpha=0.3)
        ax_act.set_title("Actuation", fontsize=10)

        # Road Grade
        grade_deg = np.degrees(seg.grade)  # Convert from radians to degrees
        ax_grade.plot(t, grade_deg, "brown", label="Road Grade", linewidth=2)
        ax_grade.set_ylabel("Grade (deg)")
        ax_grade.set_xlabel("Time (s)")
        ax_grade.legend(loc="best", fontsize=9)
        ax_grade.grid(True, alpha=0.3)
        ax_grade.set_title("Road Grade", fontsize=10)

        for ax in (ax_speed, ax_accel, ax_act, ax_grade):
            for label in ax.get_xticklabels():
                label.set_rotation(0)

        # Create window for this segment
        window = tk.Toplevel(self.root)
        window.title("Validation Segment")
        window.geometry("1200x1100")

        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(canvas, window)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def _show_validation_summary_table(self, analysis_data: Dict) -> None:
        """Show a summary window with subplots showing error metrics and RMSE vs grade."""
        window = tk.Toplevel(self.root)
        window.title("Validation Summary")
        window.geometry("3200x1200")  # Much wider to match figure
        
        # Create matplotlib figure - much wider to prevent overlap
        fig = Figure(figsize=(32, 12))  # Much wider to prevent value overlap
        fig.suptitle("Validation Summary - Error Metrics by Segment Length and Speed Range", 
                     fontsize=14, fontweight='bold')
        
        # Create subplots: 3 rows x 3 columns
        # Row 1: Speed RMSE, Speed MAE, Speed STD by speed range
        # Row 2: Accel RMSE, Accel MAE, Accel STD by speed range  
        # Row 3: Overall metrics, RMSE vs Grade scatter
        # Adjust margins to use more horizontal space
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(3, 3, figure=fig, hspace=0.6, wspace=0.6, 
                              left=0.05, right=0.98, top=0.93, bottom=0.08)
        
        segment_lengths = sorted(analysis_data.keys())
        speed_ranges = ["0-2 m/s", "2-5 m/s", "5-10 m/s", "10-20 m/s"]
        x_pos = np.arange(len(speed_ranges))
        width = 0.12  # Narrower bar width to prevent overlap
        colors = plt.cm.viridis(np.linspace(0, 1, len(segment_lengths)))
        
        # Row 1: Speed metrics
        # Speed RMSE
        ax1 = fig.add_subplot(gs[0, 0])
        for i, length_sec in enumerate(segment_lengths):
            data = analysis_data[length_sec]
            ranges = data["ranges"]
            range_dict = {r["label"]: r for r in ranges}
            values = [range_dict.get(label, {}).get("v_rmse", 0.0) for label in speed_ranges]
            bars = ax1.bar(x_pos + i * width, values, width, label=f"{length_sec:.0f}s", 
                          color=colors[i], alpha=0.8)
            # Add value labels on bars (vertically)
            for j, (bar, val) in enumerate(zip(bars, values)):
                if val > 0:
                    # Place text in the middle of the bar, rotated vertically
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2,
                            f'{val:.3f}', ha='center', va='center', fontsize=8, rotation=90)
        ax1.set_xlabel('Speed Range')
        ax1.set_ylabel('Speed RMSE (m/s)')
        ax1.set_title('Speed RMSE by Speed Range')
        ax1.set_xticks(x_pos + width * (len(segment_lengths) - 1) / 2)
        ax1.set_xticklabels([r.replace(' m/s', '') for r in speed_ranges], rotation=45, ha='right')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Speed MAE
        ax2 = fig.add_subplot(gs[0, 1])
        for i, length_sec in enumerate(segment_lengths):
            data = analysis_data[length_sec]
            ranges = data["ranges"]
            range_dict = {r["label"]: r for r in ranges}
            values = [range_dict.get(label, {}).get("v_mae", 0.0) for label in speed_ranges]
            bars = ax2.bar(x_pos + i * width, values, width, label=f"{length_sec:.0f}s", 
                          color=colors[i], alpha=0.8)
            for j, (bar, val) in enumerate(zip(bars, values)):
                if val > 0:
                    # Place text in the middle of the bar, rotated vertically
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2,
                            f'{val:.3f}', ha='center', va='center', fontsize=8, rotation=90)
        ax2.set_xlabel('Speed Range')
        ax2.set_ylabel('Speed MAE (m/s)')
        ax2.set_title('Speed MAE by Speed Range')
        ax2.set_xticks(x_pos + width * (len(segment_lengths) - 1) / 2)
        ax2.set_xticklabels([r.replace(' m/s', '') for r in speed_ranges], rotation=45, ha='right')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Speed STD
        ax3 = fig.add_subplot(gs[0, 2])
        for i, length_sec in enumerate(segment_lengths):
            data = analysis_data[length_sec]
            ranges = data["ranges"]
            range_dict = {r["label"]: r for r in ranges}
            values = [range_dict.get(label, {}).get("v_std", 0.0) for label in speed_ranges]
            bars = ax3.bar(x_pos + i * width, values, width, label=f"{length_sec:.0f}s", 
                          color=colors[i], alpha=0.8)
            for j, (bar, val) in enumerate(zip(bars, values)):
                if val > 0:
                    # Place text in the middle of the bar, rotated vertically
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2,
                            f'{val:.3f}', ha='center', va='center', fontsize=8, rotation=90)
        ax3.set_xlabel('Speed Range')
        ax3.set_ylabel('Speed STD (m/s)')
        ax3.set_title('Speed STD by Speed Range')
        ax3.set_xticks(x_pos + width * (len(segment_lengths) - 1) / 2)
        ax3.set_xticklabels([r.replace(' m/s', '') for r in speed_ranges], rotation=45, ha='right')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Row 2: Acceleration metrics
        # Accel RMSE
        ax4 = fig.add_subplot(gs[1, 0])
        for i, length_sec in enumerate(segment_lengths):
            data = analysis_data[length_sec]
            ranges = data["ranges"]
            range_dict = {r["label"]: r for r in ranges}
            values = [range_dict.get(label, {}).get("a_rmse", 0.0) for label in speed_ranges]
            bars = ax4.bar(x_pos + i * width, values, width, label=f"{length_sec:.0f}s", 
                          color=colors[i], alpha=0.8)
            for j, (bar, val) in enumerate(zip(bars, values)):
                if val > 0:
                    # Place text in the middle of the bar, rotated vertically
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2,
                            f'{val:.3f}', ha='center', va='center', fontsize=8, rotation=90)
        ax4.set_xlabel('Speed Range')
        ax4.set_ylabel('Accel RMSE (m/s²)')
        ax4.set_title('Accel RMSE by Speed Range')
        ax4.set_xticks(x_pos + width * (len(segment_lengths) - 1) / 2)
        ax4.set_xticklabels([r.replace(' m/s', '') for r in speed_ranges], rotation=45, ha='right')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Accel MAE
        ax5 = fig.add_subplot(gs[1, 1])
        for i, length_sec in enumerate(segment_lengths):
            data = analysis_data[length_sec]
            ranges = data["ranges"]
            range_dict = {r["label"]: r for r in ranges}
            values = [range_dict.get(label, {}).get("a_mae", 0.0) for label in speed_ranges]
            bars = ax5.bar(x_pos + i * width, values, width, label=f"{length_sec:.0f}s", 
                          color=colors[i], alpha=0.8)
            for j, (bar, val) in enumerate(zip(bars, values)):
                if val > 0:
                    # Place text in the middle of the bar, rotated vertically
                    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2,
                            f'{val:.3f}', ha='center', va='center', fontsize=8, rotation=90)
        ax5.set_xlabel('Speed Range')
        ax5.set_ylabel('Accel MAE (m/s²)')
        ax5.set_title('Accel MAE by Speed Range')
        ax5.set_xticks(x_pos + width * (len(segment_lengths) - 1) / 2)
        ax5.set_xticklabels([r.replace(' m/s', '') for r in speed_ranges], rotation=45, ha='right')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Accel STD
        ax6 = fig.add_subplot(gs[1, 2])
        for i, length_sec in enumerate(segment_lengths):
            data = analysis_data[length_sec]
            ranges = data["ranges"]
            range_dict = {r["label"]: r for r in ranges}
            values = [range_dict.get(label, {}).get("a_std", 0.0) for label in speed_ranges]
            bars = ax6.bar(x_pos + i * width, values, width, label=f"{length_sec:.0f}s", 
                          color=colors[i], alpha=0.8)
            for j, (bar, val) in enumerate(zip(bars, values)):
                if val > 0:
                    # Place text in the middle of the bar, rotated vertically
                    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2,
                            f'{val:.3f}', ha='center', va='center', fontsize=8, rotation=90)
        ax6.set_xlabel('Speed Range')
        ax6.set_ylabel('Accel STD (m/s²)')
        ax6.set_title('Accel STD by Speed Range')
        ax6.set_xticks(x_pos + width * (len(segment_lengths) - 1) / 2)
        ax6.set_xticklabels([r.replace(' m/s', '') for r in speed_ranges], rotation=45, ha='right')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Row 3: Overall metrics (spans entire bottom row)
        # Overall metrics bar chart
        ax7 = fig.add_subplot(gs[2, :])  # Span all 3 columns
        metrics = ['v_RMSE', 'v_MAE', 'v_STD', 'a_RMSE', 'a_MAE', 'a_STD']
        x_metrics = np.arange(len(metrics))
        for i, length_sec in enumerate(segment_lengths):
            data = analysis_data[length_sec]
            overall = data["overall"]
            values = [
                overall['v_rmse'], overall['v_mae'], overall['v_std'],
                overall['a_rmse'], overall['a_mae'], overall['a_std']
            ]
            bars = ax7.bar(x_metrics + i * width, values, width, label=f"{length_sec:.0f}s", 
                          color=colors[i], alpha=0.8)
            for j, (bar, val) in enumerate(zip(bars, values)):
                if val > 0:
                    # Place text in the middle of the bar, rotated vertically
                    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2,
                            f'{val:.3f}', ha='center', va='center', fontsize=8, rotation=90)
        ax7.set_xlabel('Metric')
        ax7.set_ylabel('Value')
        ax7.set_title('Overall Metrics by Segment Length')
        ax7.set_xticks(x_metrics + width * (len(segment_lengths) - 1) / 2)
        ax7.set_xticklabels(metrics, rotation=45, ha='right')
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3, axis='y')
        
        # Create canvas and display
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        toolbar = NavigationToolbar2Tk(canvas, window)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def _ensure_validation_segment_loaded(self) -> None:
        """Load a validation segment for preview when fitting is not running."""
        mode = self.example_segment_mode_var.get() if hasattr(self, "example_segment_mode_var") else "Longest"
        idx_val = 0
        if mode == "Index":
            try:
                idx_val = int(self.example_segment_index_var.get())
            except Exception:
                idx_val = 0
        key = (mode, idx_val)
        if self.validation_segment is not None and self.example_segment_key == key:
            return

        try:
            dataset_rel = self.dataset_var.get()
            if not dataset_rel:
                return
            dataset_path = Path(__file__).parent.parent / dataset_rel
            if not dataset_path.exists():
                return

            config = self._create_fitter_config()
            config.max_segment_length = max(config.max_segment_length, 1_000_000)
            fitter = VehicleParamFitter(config)
            trips = fitter.load_trip_data(dataset_path)
            if not trips:
                return
            dt = fitter._estimate_dt(trips)
            segments = fitter._create_segments(trips, dt)
            if not segments:
                return

            if mode == "Random":
                rng = np.random.default_rng()
                display_segment = segments[int(rng.integers(0, len(segments)))]
            elif mode == "Index":
                index = max(0, min(idx_val, len(segments) - 1))
                display_segment = segments[index]
            else:
                display_segment = max(segments, key=lambda s: s.length)
            self.validation_segment = display_segment
            self.example_segment_key = key
        except Exception:
            LOGGER.exception("Failed to load validation segment for preview")

    def _simulate_throttle_response(
        self,
        params: np.ndarray,
        dt: float = 0.1,
        duration: float = 40.0,
        fitter: Optional[VehicleParamFitter] = None,
    ) -> Tuple[Dict[float, Tuple[np.ndarray, np.ndarray]], Dict[float, Tuple[np.ndarray, np.ndarray]], Dict[float, Tuple[np.ndarray, np.ndarray]]]:
        """Simulate throttle response from 0 m/s.
        
        Returns:
            (speed_results, power_results, current_results) where each is a dict mapping throttle_pct to (time, values)
        """
        # Create fitter with current motor model type if not provided
        if fitter is None:
            motor_model_type = self.motor_model_var.get()
            config = FitterConfig(motor_model_type=motor_model_type)
            fitter = VehicleParamFitter(config)
        
        speed_results = {}
        power_results = {}
        current_results = {}

        throttle_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for throttle_pct in throttle_values:
            time, speed, power, current = self._simulate_preview_profile(
                params=params,
                fitter=fitter,
                dt=dt,
                duration=duration,
                initial_speed=0.0,
                throttle_pct=throttle_pct,
                brake_pct=0.0,
            )
            speed_results[throttle_pct] = (time, speed)
            power_results[throttle_pct] = (time, power)
            current_results[throttle_pct] = (time, current)

        return speed_results, power_results, current_results

    def _simulate_brake_response(
        self,
        params: np.ndarray,
        dt: float = 0.1,
        duration: float = 40.0,
        initial_speed: float = 20.0,
        fitter: Optional[VehicleParamFitter] = None,
    ) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
        """Simulate brake response from initial speed."""
        # Create fitter with current motor model type if not provided
        if fitter is None:
            motor_model_type = self.motor_model_var.get()
            config = FitterConfig(motor_model_type=motor_model_type)
            fitter = VehicleParamFitter(config)
        results = {}

        brake_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        for brake_pct in brake_values:
            time, speed, _power, _current = self._simulate_preview_profile(
                params=params,
                fitter=fitter,
                dt=dt,
                duration=duration,
                initial_speed=initial_speed,
                throttle_pct=0.0,
                brake_pct=brake_pct,
            )
            results[brake_pct] = (time, speed)

        return results

    def _simulate_preview_profile(
        self,
        params: np.ndarray,
        fitter: VehicleParamFitter,
        dt: float,
        duration: float,
        initial_speed: float,
        throttle_pct: float,
        brake_pct: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate a constant-input preview profile and return time, speed, power, and current."""
        n_steps = max(int(duration / dt), 2)
        time = np.arange(n_steps) * dt

        if fitter.config.motor_model_type == "dc":
            try:
                from simulation.dynamics import ExtendedPlant

                ext_params = fitter._build_extended_plant_params(params)
                plant = ExtendedPlant(ext_params)
                plant.reset(speed=initial_speed, position=0.0)

                speed = np.zeros(n_steps)
                power = np.zeros(n_steps)
                current = np.zeros(n_steps)

                state = plant.state
                speed[0] = state.speed
                current[0] = state.motor_current
                power[0] = state.V_cmd * state.motor_current

                throttle_cmd = float(np.clip(throttle_pct, 0.0, 1.0))
                brake_cmd = float(np.clip(brake_pct, 0.0, 1.0))
                brake_active = brake_cmd * 100.0 > fitter.config.brake_deadband_pct
                action = -brake_cmd if brake_active else throttle_cmd
                action = float(np.clip(action, -1.0, 1.0))
                substeps = max(int(fitter.config.extended_plant_substeps), 1)

                for t in range(n_steps - 1):
                    state = plant.step(
                        action=action,
                        grade_rad=0.0,
                        dt=dt,
                        substeps=substeps,
                    )
                    speed[t + 1] = state.speed
                    current[t + 1] = state.motor_current
                    power[t + 1] = state.V_cmd * state.motor_current

                return time, speed, power, current
            except Exception as e:
                LOGGER.warning(f"Failed full ExtendedPlant preview simulation: {e}")

        speed_measured = np.full(n_steps, float(initial_speed), dtype=np.float64)
        segment = TripSegment(
            trip_id="preview_profile",
            speed=speed_measured,
            acceleration=np.zeros(n_steps, dtype=np.float64),
            throttle=np.full(n_steps, float(np.clip(throttle_pct * 100.0, 0.0, 100.0)), dtype=np.float64),
            brake=np.full(n_steps, float(np.clip(brake_pct * 100.0, 0.0, 100.0)), dtype=np.float64),
            grade=np.zeros(n_steps, dtype=np.float64),
            dt=float(dt),
        )
        speed, _ = fitter._simulate_segment(params, segment)
        power = np.zeros(n_steps)
        current = np.zeros(n_steps)
        return time, speed, power, current

    def _simulate_preview_profile_with_state(
        self,
        params: np.ndarray,
        fitter: VehicleParamFitter,
        dt: float,
        duration: float,
        initial_speed: float,
        throttle_pct: float,
        brake_pct: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Simulate a constant-input preview profile and return time, speed, power, current and state traces."""
        n_steps = max(int(duration / dt), 2)
        time = np.arange(n_steps) * dt

        if fitter.config.motor_model_type == "dc":
            try:
                from simulation.dynamics import ExtendedPlant

                ext_params = fitter._build_extended_plant_params(params)
                plant = ExtendedPlant(ext_params)
                plant.reset(speed=initial_speed, position=0.0)

                speed = np.zeros(n_steps)
                power = np.zeros(n_steps)
                current = np.zeros(n_steps)
                state_traces = {
                    "drive_torque": np.zeros(n_steps),
                    "brake_torque": np.zeros(n_steps),
                    "tire_force": np.zeros(n_steps),
                    "drag_force": np.zeros(n_steps),
                    "rolling_force": np.zeros(n_steps),
                    "grade_force": np.zeros(n_steps),
                    "net_force": np.zeros(n_steps),
                }

                state = plant.state
                speed[0] = state.speed
                current[0] = state.motor_current
                power[0] = state.V_cmd * state.motor_current
                state_traces["drive_torque"][0] = state.drive_torque
                state_traces["brake_torque"][0] = state.brake_torque
                state_traces["tire_force"][0] = state.tire_force
                state_traces["drag_force"][0] = state.drag_force
                state_traces["rolling_force"][0] = state.rolling_force
                state_traces["grade_force"][0] = state.grade_force
                state_traces["net_force"][0] = state.net_force

                throttle_cmd = float(np.clip(throttle_pct, 0.0, 1.0))
                brake_cmd = float(np.clip(brake_pct, 0.0, 1.0))
                brake_active = brake_cmd * 100.0 > fitter.config.brake_deadband_pct
                action = -brake_cmd if brake_active else throttle_cmd
                action = float(np.clip(action, -1.0, 1.0))
                substeps = max(int(fitter.config.extended_plant_substeps), 1)

                for t in range(n_steps - 1):
                    state = plant.step(
                        action=action,
                        grade_rad=0.0,
                        dt=dt,
                        substeps=substeps,
                    )
                    speed[t + 1] = state.speed
                    current[t + 1] = state.motor_current
                    power[t + 1] = state.V_cmd * state.motor_current
                    state_traces["drive_torque"][t + 1] = state.drive_torque
                    state_traces["brake_torque"][t + 1] = state.brake_torque
                    state_traces["tire_force"][t + 1] = state.tire_force
                    state_traces["drag_force"][t + 1] = state.drag_force
                    state_traces["rolling_force"][t + 1] = state.rolling_force
                    state_traces["grade_force"][t + 1] = state.grade_force
                    state_traces["net_force"][t + 1] = state.net_force

                return time, speed, power, current, state_traces
            except Exception as e:
                LOGGER.warning(f"Failed full ExtendedPlant state preview simulation: {e}")

        time_basic, speed, power, current = self._simulate_preview_profile(
            params=params,
            fitter=fitter,
            dt=dt,
            duration=duration,
            initial_speed=initial_speed,
            throttle_pct=throttle_pct,
            brake_pct=brake_pct,
        )
        zero_states = {
            "drive_torque": np.zeros_like(time_basic),
            "brake_torque": np.zeros_like(time_basic),
            "tire_force": np.zeros_like(time_basic),
            "drag_force": np.zeros_like(time_basic),
            "rolling_force": np.zeros_like(time_basic),
            "grade_force": np.zeros_like(time_basic),
            "net_force": np.zeros_like(time_basic),
        }
        return time_basic, speed, power, current, zero_states

    def _params_dict_to_array(
        self, params_dict: Dict[str, Dict[str, float]], use_init: bool = True, motor_model_type: str = "dc"
    ) -> np.ndarray:
        """Convert parameter dictionary to array in correct order."""
        # Create temporary config to get param names
        temp_config = FitterConfig(motor_model_type=motor_model_type)
        temp_fitter = VehicleParamFitter(temp_config)
        param_names = temp_fitter.PARAM_NAMES
        param_array = np.zeros(len(param_names))

        for i, param_name in enumerate(param_names):
            if param_name in params_dict:
                if use_init:
                    param_array[i] = params_dict[param_name]["init"]
                else:
                    # Use midpoint of bounds
                    min_val = params_dict[param_name]["min"]
                    max_val = params_dict[param_name]["max"]
                    param_array[i] = (min_val + max_val) / 2.0

        return param_array

    def _validate_inputs(self) -> bool:
        """Validate all inputs before starting fitting."""
        # Check dataset
        dataset_path = self.dataset_var.get()
        if not dataset_path:
            messagebox.showerror("Validation Error", "Please select a dataset")
            return False

        full_path = Path(__file__).parent.parent / dataset_path
        if not full_path.exists():
            messagebox.showerror("Validation Error", f"Dataset not found: {full_path}")
            return False

        # Check name
        fitting_name = self.name_var.get().strip()
        if not fitting_name:
            messagebox.showerror("Validation Error", "Please enter a fitting name")
            return False

        # Validate parameters
        params_dict = self._get_params_from_gui()
        if params_dict is None:
            return False

        return True

    def _create_fitter_config(self) -> FitterConfig:
        """Create FitterConfig from GUI inputs."""
        params_dict = self._get_params_from_gui()
        if params_dict is None:
            raise ValueError("Invalid parameters")

        config_kwargs = {}

        # Add all parameter initial values and bounds
        for param_name, values in params_dict.items():
            if param_name in {"motor_T_max", "motor_P_max"} and values["init"] <= 0:
                config_kwargs[f"{param_name}_init"] = None
            else:
                config_kwargs[f"{param_name}_init"] = values["init"]
            config_kwargs[f"{param_name}_bounds"] = (values["min"], values["max"])

        # Add barrier function settings
        config_kwargs["use_barrier"] = self.use_barrier_var.get()
        try:
            barrier_mu = float(self.barrier_mu_var.get())
            if barrier_mu <= 0:
                raise ValueError("Barrier μ must be positive")
            config_kwargs["barrier_mu"] = barrier_mu
        except ValueError as e:
            raise ValueError(f"Invalid barrier μ value: {e}")

        # Add motor model settings
        config_kwargs["motor_model_type"] = self.motor_model_var.get()
        config_kwargs["fit_dc_from_map"] = self.fit_dc_from_map_var.get()

        # Add training and optimization settings
        try:
            config_kwargs["max_iter"] = int(self.max_iter_var.get())
            config_kwargs["min_segment_length"] = int(self.min_segment_length_var.get())
            config_kwargs["max_segment_length"] = int(self.max_segment_length_var.get())
            config_kwargs["segments_per_batch"] = int(self.segments_per_batch_var.get())
            config_kwargs["num_epochs"] = int(self.epochs_var.get())
            config_kwargs["speed_loss_weight"] = float(self.speed_weight_var.get())
            config_kwargs["accel_loss_weight"] = float(self.accel_weight_var.get())
            config_kwargs["brake_loss_boost"] = float(self.brake_loss_boost_var.get())
            config_kwargs["full_stop_loss_cap_fraction"] = float(self.full_stop_loss_cap_var.get())
            config_kwargs["mask_negative_gt_speed"] = bool(self.mask_negative_speed_var.get())
            config_kwargs["apply_lpf_to_fitting_data"] = bool(self.apply_lpf_to_fitting_data_var.get())
            config_kwargs["use_whole_trips"] = bool(self.use_whole_trips_var.get())
            config_kwargs["filter_zero_speed_segments"] = bool(self.filter_zero_speed_var.get())
            config_kwargs["disable_segment_filtering"] = bool(self.disable_segment_filter_var.get())
            config_kwargs["use_random_segment_batches"] = bool(self.random_batch_var.get())
            config_kwargs["random_segment_length"] = int(self.random_segment_length_var.get())
            config_kwargs["random_batches_per_epoch"] = int(self.random_batches_per_epoch_var.get())
            config_kwargs["debug_batch_progress"] = bool(self.debug_batch_progress_var.get())
            config_kwargs["debug_batch_progress_step"] = float(self.debug_batch_progress_step_var.get()) / 100.0
            config_kwargs["random_batch_max_iter"] = int(self.random_batch_max_iter_var.get())
            config_kwargs["validation_fraction"] = float(self.validation_fraction_var.get())
            seed_raw = self.validation_seed_var.get().strip()
            config_kwargs["validation_split_seed"] = int(seed_raw) if seed_raw else None
            config_kwargs["use_fixed_length_validation"] = bool(self.fixed_length_val_var.get())
            config_kwargs["use_extended_plant"] = bool(self.use_extended_plant_var.get())
            config_kwargs["extended_plant_substeps"] = int(self.plant_substeps_var.get())
            config_kwargs["actuator_smoothing_alpha"] = float(self.actuator_smoothing_var.get())
            config_kwargs["actuator_deadband_pct"] = float(self.actuator_deadband_var.get())
            config_kwargs["max_accel"] = float(self.max_accel_var.get())
            config_kwargs["use_param_scaling"] = bool(self.use_param_scaling_var.get())
            config_kwargs["optimizer_method"] = str(self.optimizer_method_var.get()).strip()
            config_kwargs["use_overfit_longest_trip"] = bool(self.overfit_longest_var.get())
            config_kwargs["overfit_longest_trip_epochs"] = int(self.overfit_longest_epochs_var.get())

            optimization_mode = self.optimization_mode_var.get()
            if optimization_mode not in {"joint", "sequential"}:
                raise ValueError("Optimization mode must be 'joint' or 'sequential'")
            config_kwargs["optimization_mode"] = optimization_mode
            config_kwargs["pause_between_phases"] = optimization_mode == "sequential"

            phase_order_raw = self.phase_order_var.get()
            phase_map = {
                "throttle -> brake": ["throttle", "brake"],
                "brake -> throttle": ["brake", "throttle"],
            }
            if phase_order_raw not in phase_map:
                raise ValueError("Invalid phase order selection")
            config_kwargs["phase_order"] = phase_map[phase_order_raw]
            
            if config_kwargs["min_segment_length"] > config_kwargs["max_segment_length"]:
                raise ValueError("Min segment length must be <= max segment length")
            if config_kwargs["min_segment_length"] <= 0:
                raise ValueError("Min segment length must be positive")
            if config_kwargs["max_iter"] <= 0:
                raise ValueError("Max iterations must be positive")
            if config_kwargs["extended_plant_substeps"] <= 0:
                raise ValueError("Plant substeps must be positive")
            if not (0.0 <= config_kwargs["actuator_smoothing_alpha"] <= 1.0):
                raise ValueError("Actuator smoothing α must be between 0 and 1")
            if config_kwargs["actuator_deadband_pct"] < 0:
                raise ValueError("Actuator deadband must be non-negative")
            if config_kwargs["max_accel"] <= 0:
                raise ValueError("Max accel must be positive")
            if config_kwargs["brake_loss_boost"] < 0:
                raise ValueError("Brake loss boost must be non-negative")
            if config_kwargs["random_segment_length"] <= 0:
                raise ValueError("Batch segment length must be positive")
            if config_kwargs["random_batches_per_epoch"] <= 0:
                raise ValueError("Random batches per epoch must be positive")
            if config_kwargs["debug_batch_progress_step"] <= 0 or config_kwargs["debug_batch_progress_step"] > 1.0:
                raise ValueError("Progress step (%) must be in (0, 100]")
            if config_kwargs["random_batch_max_iter"] <= 0:
                raise ValueError("Random batch max iter must be positive")
            if not (0.0 < config_kwargs["validation_fraction"] < 1.0):
                raise ValueError("Validation fraction must be in (0, 1)")
                
        except ValueError as e:
            raise ValueError(f"Invalid optimization setting: {e}")

        return FitterConfig(**config_kwargs)

    def _start_fitting(self):
        """Start the fitting process in a background thread."""
        if not self._validate_inputs():
            return

        if self.is_fitting:
            messagebox.showwarning("Warning", "Fitting already in progress")
            return

        # Disable button and show progress
        self.start_fitting_btn.config(state=tk.DISABLED)
        self.abort_fitting_btn.config(state=tk.NORMAL)
        if self.optimization_mode_var.get() == "sequential":
            self.advance_phase_btn.config(state=tk.NORMAL)
        else:
            self.advance_phase_btn.config(state=tk.DISABLED)
        self.is_fitting = True
        self.validation_segment = None
        self.val_lines.clear()
        self.ax3.clear()
        self.progress_var.set("Starting fitting...")
        self.progress_bar.start()

        # Start background thread
        self.fitting_thread = threading.Thread(target=self._run_fitting, daemon=True)
        self.fitting_thread.start()

    def _run_fitting(self):
        """Run fitting in background thread."""
        try:
            # Get inputs
            dataset_path = Path(__file__).parent.parent / self.dataset_var.get()
            fitting_name = self.name_var.get().strip()
            output_dir = self.results_dir / fitting_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create config
            config = self._create_fitter_config()

            # Save config
            config_path = output_dir / "config.json"
            config_dict = {
                "dataset": str(dataset_path),
                "fitting_name": fitting_name,
                **asdict(config),
            }
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)

            # Update status
            self.root.after(0, lambda: self.progress_var.set("Loading data..."))

            # Create fitter and run
            fitter = VehicleParamFitter(config)
            self.current_fitter = fitter  # Store for param names access
            output_params_path = output_dir / "fitted_params.json"

            self.root.after(0, lambda: self.progress_var.set("Fitting parameters..."))

            fitted = fitter.fit(
                dataset_path,
                verbose=True,
                log_path=output_dir / "fitting_checkpoint.json",
                progress_callback=self._progress_callback
            )
            
            self.current_fitter = None  # Clear after fitting

            # Save results
            fitted.save(output_params_path)

            # Save simulation preview plots
            # Extract parameters based on model type
            if config.motor_model_type == "polynomial":
                # Polynomial model - extract from dict
                param_dict = fitted.to_dict()
                motor_model_type = param_dict.get("motor_model_type", "polynomial")
                temp_config = FitterConfig(motor_model_type=motor_model_type)
                temp_fitter = VehicleParamFitter(temp_config)
                param_names = temp_fitter.PARAM_NAMES
                param_array = np.array([param_dict.get(name, 0.0) for name in param_names])
            else:
                # DC model - use FittedVehicleParams structure
                param_array = np.array([
                    fitted.mass, fitted.drag_area, fitted.rolling_coeff,
                    fitted.motor_V_max, fitted.motor_R,
                    fitted.motor_K, fitted.motor_b, fitted.motor_J, fitted.motor_gamma_throttle, fitted.motor_throttle_tau,
                    fitted.motor_min_current_A,
                    fitted.motor_T_max if fitted.motor_T_max is not None else (fitted.motor_K * (fitted.motor_V_max / max(fitted.motor_R, 1e-4))),
                    fitted.motor_P_max if fitted.motor_P_max is not None else 0.0,
                    fitted.gear_ratio, fitted.eta_gb,
                    fitted.brake_T_max, fitted.brake_tau, fitted.brake_p,
                    fitted.brake_kappa, fitted.mu,
                    fitted.wheel_radius, fitted.wheel_inertia,
                ])

            # Use the model type from config (what was actually fitted)
            # Temporarily set motor model type for simulations
            original_model_type = self.motor_model_var.get()
            self.motor_model_var.set(config.motor_model_type)
            
            throttle_speed, throttle_power, throttle_current = self._simulate_throttle_response(param_array, fitter=fitter)
            brake_data = self._simulate_brake_response(param_array, fitter=fitter)

            # Representative full-state profiles for preview force/torque traces
            state_time, _state_speed, _state_power, _state_current, throttle_state = self._simulate_preview_profile_with_state(
                params=param_array,
                fitter=fitter,
                dt=0.1,
                duration=40.0,
                initial_speed=0.0,
                throttle_pct=1.0,
                brake_pct=0.0,
            )
            brake_state_time, _brake_state_speed, _brake_state_power, _brake_state_current, brake_state = self._simulate_preview_profile_with_state(
                params=param_array,
                fitter=fitter,
                dt=0.1,
                duration=40.0,
                initial_speed=20.0,
                throttle_pct=0.0,
                brake_pct=1.0,
            )
            
            # Restore original model type
            self.motor_model_var.set(original_model_type)

            # Create and save preview plots
            preview_fig = Figure(figsize=(10, 20), dpi=100)
            ax1 = preview_fig.add_subplot(6, 1, 1)
            ax2 = preview_fig.add_subplot(6, 1, 2)
            ax3 = preview_fig.add_subplot(6, 1, 3)
            ax4 = preview_fig.add_subplot(6, 1, 4)
            ax5 = preview_fig.add_subplot(6, 1, 5)
            ax6 = preview_fig.add_subplot(6, 1, 6)

            for throttle, (time, speed) in throttle_speed.items():
                ax1.plot(time, speed, label=f"Throttle {throttle:.1f}")
            ax1.set_title("Throttle Dynamics (from 0 m/s)")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Speed (m/s)")
            ax1.legend(ncol=2, fontsize=8, loc="upper right")
            ax1.grid(True, alpha=0.3)

            for brake, (time, speed) in brake_data.items():
                ax2.plot(time, speed, label=f"Brake {brake:.1f}")
            ax2.set_title("Braking Dynamics (from 20 m/s)")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Speed (m/s)")
            ax2.legend(ncol=2, fontsize=8, loc="upper right")
            ax2.grid(True, alpha=0.3)

            segment = self.validation_segment
            if segment is not None:
                # Simulate validation segment
                v_sim, _ = fitter._simulate_segment(param_array, segment)
                time_arr = np.arange(len(v_sim)) * segment.dt
                # No downsampling - show complete data
                v_sim_ds = v_sim
                v_gt_ds = segment.speed
                th_ds = segment.throttle
                br_ds = segment.brake
                
                ax3.plot(time_arr, v_gt_ds, "k--", label="Ground Truth", alpha=0.6)
                ax3.plot(time_arr, v_sim_ds, "r-", label="Simulated")
                ax3.set_title(f"Validation: {segment.trip_id}")
                ax3.set_xlabel("Time (s)")
                ax3.set_ylabel("Speed (m/s)")
                ax3.legend(fontsize=8, loc="upper right")
                ax3.grid(True, alpha=0.3)

                ax4.plot(time_arr, th_ds, "g-", label="Throttle")
                ax4.plot(time_arr, br_ds, "b-", label="Brake")
                ax4.set_title("Validation Segment Actuations")
                ax4.set_xlabel("Time (s)")
                ax4.set_ylabel("Command (%)")
                ax4.legend(fontsize=8, loc="upper right")
                ax4.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, "No validation segment data available", 
                        ha="center", va="center", transform=ax3.transAxes)
                ax4.text(0.5, 0.5, "No validation segment data available", 
                        ha="center", va="center", transform=ax4.transAxes)

            if config.motor_model_type == "dc":
                ax5.plot(state_time, throttle_state["drive_torque"], "c-", label="Drive Torque @ Wheel (Throttle=1.0)")
                ax5.plot(brake_state_time, brake_state["brake_torque"], "r-", label="Brake Torque (Brake=1.0)")
                ax5.set_title("Full-State Torques (Wheel Side)")
                ax5.set_xlabel("Time (s)")
                ax5.set_ylabel("Torque (Nm)")
                ax5.legend(fontsize=8, loc="upper right")
                ax5.grid(True, alpha=0.3)

                ax6.plot(state_time, throttle_state["tire_force"], "b-", label="Tire")
                ax6.plot(state_time, throttle_state["drag_force"], "g-", label="Drag")
                ax6.plot(state_time, throttle_state["rolling_force"], "orange", label="Rolling")
                ax6.plot(state_time, throttle_state["grade_force"], "brown", label="Grade")
                ax6.plot(state_time, throttle_state["net_force"], "k-", label="Net", linewidth=2)
                ax6.set_title("Full-State Forces (Throttle=1.0)")
                ax6.set_xlabel("Time (s)")
                ax6.set_ylabel("Force (N)")
                ax6.legend(fontsize=8, loc="upper right")
                ax6.grid(True, alpha=0.3)
            else:
                ax5.text(0.5, 0.5, "Full-state torque traces require DC motor model", 
                        ha="center", va="center", transform=ax5.transAxes)
                ax5.set_title("Full-State Torques")
                ax5.set_xlabel("Time (s)")
                ax5.set_ylabel("Torque (Nm)")

                ax6.text(0.5, 0.5, "Full-state force traces require DC motor model", 
                        ha="center", va="center", transform=ax6.transAxes)
                ax6.set_title("Full-State Forces")
                ax6.set_xlabel("Time (s)")
                ax6.set_ylabel("Force (N)")

            preview_fig.tight_layout()
            preview_fig.savefig(output_dir / "simulation_preview.png", dpi=150, bbox_inches="tight")

            # Success
            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Success",
                    f"Fitting completed!\nResults saved to:\n{output_dir}",
                ),
            )

        except Exception as e:
            from fitting.fitter import AbortFitting
            if isinstance(e, AbortFitting):
                self.root.after(0, lambda: self.progress_var.set("Fitting aborted"))
            else:
                LOGGER.exception("Fitting failed")
                error_msg = str(e)
                self.root.after(
                    0, lambda msg=error_msg: messagebox.showerror("Fitting Error", f"Fitting failed:\n{msg}")
                )
        finally:
            # Re-enable button
            self.root.after(0, self._fitting_complete)

    def _fitting_complete(self):
        """Called when fitting completes (in main thread)."""
        self.is_fitting = False
        self.start_fitting_btn.config(state=tk.NORMAL)
        self.abort_fitting_btn.config(state=tk.DISABLED)
        self.advance_phase_btn.config(state=tk.DISABLED)
        self.progress_bar.stop()
        self.progress_var.set("Fitting completed - parameters updated")

    def _abort_fitting(self):
        """Request abort of the fitting process."""
        if not self.is_fitting or self.current_fitter is None:
            messagebox.showwarning("Warning", "No fitting in progress")
            return
        self.current_fitter.request_abort()
        self.progress_var.set("Abort requested...")
        self.abort_fitting_btn.config(state=tk.DISABLED)

    def _advance_phase(self):
        """Request moving to the next optimization phase."""
        if not self.is_fitting or self.current_fitter is None:
            messagebox.showwarning("Warning", "No fitting in progress")
            return
        if self.optimization_mode_var.get() != "sequential":
            messagebox.showwarning("Warning", "Phase advance is only available in sequential mode")
            return
        self.current_fitter.request_phase_advance()
        self.progress_var.set("Phase advance requested")

    def _progress_callback(self, best_params: np.ndarray, best_loss: float):
        """Callback called during fitting when new best parameters are found."""
        # Store pending update
        self.pending_params = best_params.copy()
        self.pending_loss = best_loss

        # Throttle updates - only schedule if enough time has passed
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            if not self.update_pending:
                self.update_pending = True
                self.root.after(0, self._process_pending_update)
                self.last_update_time = current_time

    def _process_pending_update(self):
        """Process pending parameter update (called in main thread)."""
        if self.pending_params is None or self.pending_loss is None:
            self.update_pending = False
            return

        params = self.pending_params
        loss = self.pending_loss

        # Update parameter display with current best
        # Get param names from current fitter if available
        if self.current_fitter is not None:
            # Pick a validation segment if we don't have one yet
            if self.validation_segment is None:
                display_segment = self.current_fitter.get_longest_validation_display_segment()
                if display_segment is not None:
                    self.validation_segment = display_segment

            param_names = self.current_fitter.PARAM_NAMES
            for i, param_name in enumerate(param_names):
                if param_name in self.param_entries:
                    # Update the initial value field to show current best
                    self.param_entries[param_name]["init"].set(f"{params[i]:.6f}")

        # Update progress text
        rmse = np.sqrt(loss)
        self.progress_var.set(f"New best found - RMSE: {rmse:.4f} m/s")

        # Update simulation plots
        self._update_simulation_plots_callback(params)

        # Clear pending
        self.pending_params = None
        self.pending_loss = None
        self.update_pending = False

    def _update_simulation_plots_callback(self, params: np.ndarray):
        """Update simulation plots with given parameters (called in main thread)."""
        fitter = self.current_fitter
        if fitter is None:
            try:
                config = self._create_fitter_config()
            except Exception:
                config = FitterConfig(motor_model_type=self.motor_model_var.get())
            fitter = VehicleParamFitter(config)

        # Run simulations
        throttle_speed, throttle_power, throttle_current = self._simulate_throttle_response(params, fitter=fitter)
        brake_data = self._simulate_brake_response(params, fitter=fitter)

        # Throttle speed plot - reuse lines if they exist
        if not self.throttle_lines:
            # First time: create lines and legend
            for throttle, (time_arr, speed) in throttle_speed.items():
                line, = self.ax1.plot(time_arr, speed, label=f"Throttle {throttle:.1f}")
                self.throttle_lines[throttle] = line
            self.ax1.set_title("Throttle Dynamics - Speed (from 0 m/s)")
            self.ax1.set_xlabel("Time (s)")
            self.ax1.set_ylabel("Speed (m/s)")
            self.ax1.legend(ncol=2, fontsize=8, loc="upper right")
            self.ax1.grid(True, alpha=0.3)
        else:
            # Update existing lines
            for throttle, (time_arr, speed) in throttle_speed.items():
                if throttle in self.throttle_lines:
                    self.throttle_lines[throttle].set_data(time_arr, speed)

        # Throttle power plot
        if not self.throttle_power_lines:
            for throttle, (time_arr, power) in throttle_power.items():
                line, = self.ax1_power.plot(time_arr, power, label=f"Throttle {throttle:.1f}")
                self.throttle_power_lines[throttle] = line
            self.ax1_power.set_title("Throttle Dynamics - Power")
            self.ax1_power.set_xlabel("Time (s)")
            self.ax1_power.set_ylabel("Power (W)")
            self.ax1_power.legend(ncol=2, fontsize=8, loc="upper right")
            self.ax1_power.grid(True, alpha=0.3)
        else:
            for throttle, (time_arr, power) in throttle_power.items():
                if throttle in self.throttle_power_lines:
                    self.throttle_power_lines[throttle].set_data(time_arr, power)

        # Throttle current plot
        if not self.throttle_current_lines:
            for throttle, (time_arr, current) in throttle_current.items():
                line, = self.ax1_current.plot(time_arr, current, label=f"Throttle {throttle:.1f}")
                self.throttle_current_lines[throttle] = line
            self.ax1_current.set_title("Throttle Dynamics - Current")
            self.ax1_current.set_xlabel("Time (s)")
            self.ax1_current.set_ylabel("Current (A)")
            self.ax1_current.legend(ncol=2, fontsize=8, loc="upper right")
            self.ax1_current.grid(True, alpha=0.3)
        else:
            for throttle, (time_arr, current) in throttle_current.items():
                if throttle in self.throttle_current_lines:
                    self.throttle_current_lines[throttle].set_data(time_arr, current)

        # Brake plot - reuse lines if they exist
        if not self.brake_lines:
            # First time: create lines and legend
            for brake, (time_arr, speed) in brake_data.items():
                line, = self.ax2.plot(time_arr, speed, label=f"Brake {brake:.1f}")
                self.brake_lines[brake] = line
            self.ax2.set_title("Braking Dynamics (from 20 m/s)")
            self.ax2.set_xlabel("Time (s)")
            self.ax2.set_ylabel("Speed (m/s)")
            self.ax2.legend(ncol=2, fontsize=8, loc="upper right")
            self.ax2.grid(True, alpha=0.3)
        else:
            # Update existing lines
            for brake, (time_arr, speed) in brake_data.items():
                if brake in self.brake_lines:
                    self.brake_lines[brake].set_data(time_arr, speed)

        # Validation segment plot
        segment = self.validation_segment
        if segment is not None:
            # Check if we can use extended plant (DC motor model with state tracking)
            use_extended = (fitter.config.use_extended_plant and 
                           fitter.config.motor_model_type == "dc")
            
            if use_extended:
                # Simulate with state tracking
                try:
                    v_sim, current, power, voltage, I_max, P_max, i_limit = \
                        self._simulate_segment_with_state(params, segment, fitter)
                    time_arr = np.arange(len(v_sim)) * segment.dt
                    has_state_data = True
                except Exception as e:
                    LOGGER.warning(f"Failed to extract state data: {e}")
                    # Fallback to regular simulation
                    v_sim, _ = fitter._simulate_segment(params, segment)
                    time_arr = np.arange(len(v_sim)) * segment.dt
                    has_state_data = False
            else:
                # Regular simulation without state tracking
                v_sim, _ = fitter._simulate_segment(params, segment)
                time_arr = np.arange(len(v_sim)) * segment.dt
                has_state_data = False
            
            # No downsampling - show complete data
            v_sim_ds = v_sim
            v_gt_ds = segment.speed
            th_ds = segment.throttle
            br_ds = segment.brake

            # Speed comparison plot
            if not self.val_lines:
                line_gt, = self.ax3.plot(time_arr, v_gt_ds, "k--", label="Ground Truth", alpha=0.6)
                line_sim, = self.ax3.plot(time_arr, v_sim_ds, "r-", label="Simulated")
                self.val_lines["gt"] = line_gt
                self.val_lines["sim"] = line_sim
                self.ax3.set_title(f"Validation: {segment.trip_id}")
                self.ax3.set_xlabel("Time (s)")
                self.ax3.set_ylabel("Speed (m/s)")
                self.ax3.legend(fontsize=8, loc="upper right")
                self.ax3.grid(True, alpha=0.3)
            else:
                self.val_lines["sim"].set_data(time_arr, v_sim_ds)
                self.val_lines["gt"].set_data(time_arr, v_gt_ds)

            # Validation actuations plot
            if not self.val_act_lines:
                self.ax4.clear()
                line_th, = self.ax4.plot(time_arr, th_ds, "g-", label="Throttle")
                line_br, = self.ax4.plot(time_arr, br_ds, "b-", label="Brake")
                self.val_act_lines["throttle"] = line_th
                self.val_act_lines["brake"] = line_br
                self.ax4.set_title("Validation Segment Actuations")
                self.ax4.set_xlabel("Time (s)")
                self.ax4.set_ylabel("Command (%)")
                self.ax4.legend(fontsize=8, loc="upper right")
                self.ax4.grid(True, alpha=0.3)
            else:
                self.val_act_lines["throttle"].set_data(time_arr, th_ds)
                self.val_act_lines["brake"].set_data(time_arr, br_ds)
            
            # Power plot
            if has_state_data:
                if not self.val_power_lines:
                    self.ax5.clear()
                    line_p, = self.ax5.plot(time_arr, power, "b-", label="Power")
                    self.val_power_lines["power"] = line_p
                    if P_max > 0:
                        line_pmax, = self.ax5.plot(time_arr, np.full_like(time_arr, P_max), 
                                                   "r--", label=f"P_max ({P_max/1000:.1f} kW)", alpha=0.7)
                        self.val_power_lines["pmax"] = line_pmax
                    self.ax5.set_title("Motor Power")
                    self.ax5.set_xlabel("Time (s)")
                    self.ax5.set_ylabel("Power (W)")
                    self.ax5.legend(fontsize=8, loc="upper right")
                    self.ax5.grid(True, alpha=0.3)
                else:
                    self.val_power_lines["power"].set_data(time_arr, power)
                    if "pmax" in self.val_power_lines and P_max > 0:
                        self.val_power_lines["pmax"].set_data(time_arr, np.full_like(time_arr, P_max))
                
                # Current plot
                if not self.val_current_lines:
                    self.ax6.clear()
                    line_i, = self.ax6.plot(time_arr, current, "g-", label="Current")
                    self.val_current_lines["current"] = line_i
                    if I_max > 0:
                        line_imax, = self.ax6.plot(time_arr, np.full_like(time_arr, I_max), 
                                                   "r--", label=f"Static Current Limit ({I_max:.1f} A)", alpha=0.7)
                        self.val_current_lines["imax"] = line_imax
                    line_ilimit, = self.ax6.plot(time_arr, i_limit, "orange", linestyle=":", 
                                                 label="Dynamic Current Limit (Voltage & Speed Dependent)", alpha=0.7)
                    self.val_current_lines["ilimit"] = line_ilimit
                    self.ax6.set_title("Motor Current")
                    self.ax6.set_xlabel("Time (s)")
                    self.ax6.set_ylabel("Current (A)")
                    self.ax6.legend(fontsize=8, loc="upper right")
                    self.ax6.grid(True, alpha=0.3)
                else:
                    self.val_current_lines["current"].set_data(time_arr, current)
                    if "imax" in self.val_current_lines and I_max > 0:
                        self.val_current_lines["imax"].set_data(time_arr, np.full_like(time_arr, I_max))
                    self.val_current_lines["ilimit"].set_data(time_arr, i_limit)
            else:
                # No state data available - show placeholder
                if not self.val_power_lines:
                    self.ax5.set_title("Motor Power")
                    self.ax5.set_xlabel("Time (s)")
                    self.ax5.set_ylabel("Power (W)")
                    self.ax5.text(0.5, 0.5, "State data not available\n(requires DC motor + ExtendedPlant)",
                                 ha="center", va="center", transform=self.ax5.transAxes)
                if not self.val_current_lines:
                    self.ax6.set_title("Motor Current")
                    self.ax6.set_xlabel("Time (s)")
                    self.ax6.set_ylabel("Current (A)")
                    self.ax6.text(0.5, 0.5, "State data not available\n(requires DC motor + ExtendedPlant)",
                                 ha="center", va="center", transform=self.ax6.transAxes)
        else:
            # No segment - show placeholders
            if not self.val_act_lines:
                self.ax4.set_title("Validation Segment Actuations")
                self.ax4.set_xlabel("Time (s)")
                self.ax4.set_ylabel("Command (%)")
                self.ax4.text(0.5, 0.5, "No validation segment data available",
                              ha="center", va="center", transform=self.ax4.transAxes)
            if not self.val_power_lines:
                self.ax5.set_title("Motor Power")
                self.ax5.set_xlabel("Time (s)")
                self.ax5.set_ylabel("Power (W)")
                self.ax5.text(0.5, 0.5, "No validation segment data available",
                             ha="center", va="center", transform=self.ax5.transAxes)
            if not self.val_current_lines:
                self.ax6.set_title("Motor Current")
                self.ax6.set_xlabel("Time (s)")
                self.ax6.set_ylabel("Current (A)")
                self.ax6.text(0.5, 0.5, "No validation segment data available",
                             ha="center", va="center", transform=self.ax6.transAxes)

        # Update axis limits
        self.ax1.relim()
        self.ax1.autoscale()
        self.ax1_power.relim()
        self.ax1_power.autoscale()
        self.ax1_current.relim()
        self.ax1_current.autoscale()
        self.ax2.relim()
        self.ax2.autoscale()
        self.ax3.relim()
        self.ax3.autoscale()
        self.ax4.relim()
        self.ax4.autoscale()
        self.ax5.relim()
        self.ax5.autoscale()
        self.ax6.relim()
        self.ax6.autoscale()

        # Only call tight_layout once initially, then just draw
        if not self.throttle_lines or not self.brake_lines or not self.val_lines or not self.val_act_lines:
            # First time setup - need tight_layout
            self.fig.tight_layout()
        
        self.canvas_plot.draw_idle()  # Use draw_idle for better performance

    def _save_settings(self):
        """Save current GUI settings to a JSON file."""
        try:
            settings = {
                # Dataset
                "dataset": self.dataset_var.get(),
                
                # Motor model and barrier
                "motor_model": self.motor_model_var.get(),
                "use_barrier": self.use_barrier_var.get(),
                "barrier_mu": self.barrier_mu_var.get(),
                "fit_dc_from_map": self.fit_dc_from_map_var.get(),
                
                # Training & Optimization
                "max_iter": self.max_iter_var.get(),
                "min_segment_length": self.min_segment_length_var.get(),
                "max_segment_length": self.max_segment_length_var.get(),
                "apply_lpf_to_fitting_data": self.apply_lpf_to_fitting_data_var.get(),
                "use_whole_trips": self.use_whole_trips_var.get(),
                "filter_zero_speed": self.filter_zero_speed_var.get(),
                "disable_segment_filter": self.disable_segment_filter_var.get(),
                "segments_per_batch": self.segments_per_batch_var.get(),
                "epochs": self.epochs_var.get(),
                "random_batch": self.random_batch_var.get(),
                "random_batches_per_epoch": self.random_batches_per_epoch_var.get(),
                "random_batch_max_iter": self.random_batch_max_iter_var.get(),
                "random_segment_length": self.random_segment_length_var.get(),
                "validation_fraction": self.validation_fraction_var.get(),
                "validation_seed": self.validation_seed_var.get(),
                "optimizer_method": self.optimizer_method_var.get(),
                "fixed_length_val": self.fixed_length_val_var.get(),
                "debug_batch_progress": self.debug_batch_progress_var.get(),
                "debug_batch_progress_step": self.debug_batch_progress_step_var.get(),
                
                # Loss weights
                "speed_weight": self.speed_weight_var.get(),
                "accel_weight": self.accel_weight_var.get(),
                "brake_loss_boost": self.brake_loss_boost_var.get(),
                "full_stop_loss_cap": self.full_stop_loss_cap_var.get(),
                "mask_negative_speed": self.mask_negative_speed_var.get(),
                
                # Plant settings
                "use_extended_plant": self.use_extended_plant_var.get(),
                "plant_substeps": self.plant_substeps_var.get(),
                "actuator_smoothing": self.actuator_smoothing_var.get(),
                "actuator_deadband": self.actuator_deadband_var.get(),
                "max_accel": self.max_accel_var.get(),
                "use_param_scaling": self.use_param_scaling_var.get(),
                
                # Optimization mode
                "optimization_mode": self.optimization_mode_var.get(),
                "phase_order": self.phase_order_var.get(),
                
                # Overfit options
                "overfit_longest": self.overfit_longest_var.get(),
                "overfit_longest_epochs": self.overfit_longest_epochs_var.get(),
                
                # Example segment
                "example_segment_mode": self.example_segment_mode_var.get(),
                "example_segment_index": self.example_segment_index_var.get(),
                
                # Parameter settings
                "parameters": {}
            }
            
            # Save parameter entries
            for param_name, entries in self.param_entries.items():
                settings["parameters"][param_name] = {
                    "init": entries["init"].get(),
                    "min": entries["min"].get(),
                    "max": entries["max"].get(),
                }
            
            # Write to file
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            
            messagebox.showinfo(
                "Settings Saved",
                f"Settings saved to:\n{self.settings_file}\n\n"
                "These settings will be loaded automatically next time."
            )
            
        except Exception as e:
            messagebox.showerror("Error Saving Settings", f"Failed to save settings:\n{str(e)}")
            LOGGER.error(f"Error saving settings: {e}", exc_info=True)

    def _load_settings(self):
        """Load GUI settings from JSON file if it exists."""
        # Fitting name should always use current wall-clock timestamp, not persisted defaults.
        self.name_var.set(datetime.now().strftime("fit_%Y%m%d_%H%M%S"))

        if not self.settings_file.exists():
            return  # No saved settings, use defaults
        
        try:
            with open(self.settings_file, 'r') as f:
                settings = json.load(f)
            
            # Restore dataset
            if "dataset" in settings:
                self.dataset_var.set(settings["dataset"])
            
            # Restore motor model and barrier
            if "motor_model" in settings:
                old_model = self.motor_model_var.get()
                self.motor_model_var.set(settings["motor_model"])
                # Trigger model change if different
                if old_model != settings["motor_model"]:
                    self._on_motor_model_changed()
            if "use_barrier" in settings:
                self.use_barrier_var.set(settings["use_barrier"])
            if "barrier_mu" in settings:
                self.barrier_mu_var.set(settings["barrier_mu"])
            if "fit_dc_from_map" in settings:
                self.fit_dc_from_map_var.set(settings["fit_dc_from_map"])
            
            # Restore training & optimization settings
            if "max_iter" in settings:
                self.max_iter_var.set(settings["max_iter"])
            if "min_segment_length" in settings:
                self.min_segment_length_var.set(settings["min_segment_length"])
            if "max_segment_length" in settings:
                self.max_segment_length_var.set(settings["max_segment_length"])
            if "apply_lpf_to_fitting_data" in settings:
                self.apply_lpf_to_fitting_data_var.set(settings["apply_lpf_to_fitting_data"])
            if "use_whole_trips" in settings:
                self.use_whole_trips_var.set(settings["use_whole_trips"])
            if "filter_zero_speed" in settings:
                self.filter_zero_speed_var.set(settings["filter_zero_speed"])
            if "disable_segment_filter" in settings:
                self.disable_segment_filter_var.set(settings["disable_segment_filter"])
            if "segments_per_batch" in settings:
                self.segments_per_batch_var.set(settings["segments_per_batch"])
            if "epochs" in settings:
                self.epochs_var.set(settings["epochs"])
            if "random_batch" in settings:
                self.random_batch_var.set(settings["random_batch"])
            if "random_batches_per_epoch" in settings:
                self.random_batches_per_epoch_var.set(settings["random_batches_per_epoch"])
            if "random_batch_max_iter" in settings:
                self.random_batch_max_iter_var.set(settings["random_batch_max_iter"])
            if "random_segment_length" in settings:
                self.random_segment_length_var.set(settings["random_segment_length"])
            if "validation_fraction" in settings:
                self.validation_fraction_var.set(settings["validation_fraction"])
            if "validation_seed" in settings:
                self.validation_seed_var.set(settings["validation_seed"])
            if "optimizer_method" in settings:
                self.optimizer_method_var.set(settings["optimizer_method"])
            if "fixed_length_val" in settings:
                self.fixed_length_val_var.set(settings["fixed_length_val"])
            if "debug_batch_progress" in settings:
                self.debug_batch_progress_var.set(settings["debug_batch_progress"])
            if "debug_batch_progress_step" in settings:
                self.debug_batch_progress_step_var.set(settings["debug_batch_progress_step"])
            
            # Restore loss weights
            if "speed_weight" in settings:
                self.speed_weight_var.set(settings["speed_weight"])
            if "accel_weight" in settings:
                self.accel_weight_var.set(settings["accel_weight"])
            if "brake_loss_boost" in settings:
                self.brake_loss_boost_var.set(settings["brake_loss_boost"])
            if "full_stop_loss_cap" in settings:
                self.full_stop_loss_cap_var.set(settings["full_stop_loss_cap"])
            if "mask_negative_speed" in settings:
                self.mask_negative_speed_var.set(settings["mask_negative_speed"])
            
            # Restore plant settings
            if "use_extended_plant" in settings:
                self.use_extended_plant_var.set(settings["use_extended_plant"])
            if "plant_substeps" in settings:
                self.plant_substeps_var.set(settings["plant_substeps"])
            if "actuator_smoothing" in settings:
                self.actuator_smoothing_var.set(settings["actuator_smoothing"])
            if "actuator_deadband" in settings:
                self.actuator_deadband_var.set(settings["actuator_deadband"])
            if "max_accel" in settings:
                self.max_accel_var.set(settings["max_accel"])
            if "use_param_scaling" in settings:
                self.use_param_scaling_var.set(settings["use_param_scaling"])
            
            # Restore optimization mode
            if "optimization_mode" in settings:
                self.optimization_mode_var.set(settings["optimization_mode"])
            if "phase_order" in settings:
                self.phase_order_var.set(settings["phase_order"])
            
            # Restore overfit options
            if "overfit_longest" in settings:
                self.overfit_longest_var.set(settings["overfit_longest"])
            if "overfit_longest_epochs" in settings:
                self.overfit_longest_epochs_var.set(settings["overfit_longest_epochs"])
            
            # Restore example segment
            if "example_segment_mode" in settings:
                self.example_segment_mode_var.set(settings["example_segment_mode"])
            if "example_segment_index" in settings:
                self.example_segment_index_var.set(settings["example_segment_index"])
            
            # Restore parameter settings
            if "parameters" in settings and hasattr(self, 'param_entries'):
                for param_name, values in settings["parameters"].items():
                    if param_name in self.param_entries:
                        if "init" in values:
                            self.param_entries[param_name]["init"].set(values["init"])
                        if "min" in values:
                            self.param_entries[param_name]["min"].set(values["min"])
                        if "max" in values:
                            self.param_entries[param_name]["max"].set(values["max"])
            
            LOGGER.info(f"Loaded settings from {self.settings_file}")
            
        except Exception as e:
            LOGGER.warning(f"Could not load settings: {e}")
            # Don't show error dialog on load failure, just use defaults


def main():
    """Launch the GUI application."""
    root = tk.Tk()
    app = FittingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

