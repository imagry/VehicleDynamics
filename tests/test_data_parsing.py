"""Unit tests for data parsing module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from data.parsing import TripDatasetParser, TripParserConfig


class TestTripParsing:
    """Tests for trip data parsing."""
    
    def test_parse_simple_trip(self) -> None:
        """Test parsing a simple trip with all required sensors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trip_dir = Path(tmpdir) / "test_trip"
            trip_dir.mkdir()
            
            # Create sample CSV files
            time_stamps = np.arange(0, 10, 0.02)  # 10 seconds at 50Hz
            n = len(time_stamps)
            
            # Wheel speeds (km/h)
            left_speed = 50.0 + np.random.randn(n) * 0.5
            right_speed = 50.0 + np.random.randn(n) * 0.5
            
            # Actuators (0-100)
            throttle = np.random.uniform(0, 50, n)
            brake = np.random.uniform(0, 10, n)
            
            # IMU pitch (degrees)
            pitch = np.random.uniform(-2, 2, n)
            
            # Driving mode (7 = autonomous)
            driving_mode = np.full(n, 7)
            
            # Write CSV files
            pd.DataFrame({
                "time_stamp": time_stamps,
                "data_value": left_speed,
            }).to_csv(trip_dir / "rear_left_wheel_speed.csv", index=False)
            
            pd.DataFrame({
                "time_stamp": time_stamps,
                "data_value": right_speed,
            }).to_csv(trip_dir / "rear_right_wheel_speed.csv", index=False)
            
            pd.DataFrame({
                "time_stamp": time_stamps,
                "throttle": throttle,
                "brake": brake,
            }).to_csv(trip_dir / "cruise_control.csv", index=False)
            
            pd.DataFrame({
                "time_stamp": time_stamps,
                "pitch": pitch,
            }).to_csv(trip_dir / "imu.csv", index=False)
            
            pd.DataFrame({
                "time_stamp": time_stamps,
                "data_value": driving_mode,
            }).to_csv(trip_dir / "driving_mode.csv", index=False)
            
            # Car info
            (trip_dir / "car_info.json").write_text('{"car_type": "TestCar"}')
            
            # Parse
            config = TripParserConfig(
                root_folder=Path(tmpdir),
                car_model="TestCar",
                dt=0.02,
                out_dir=Path(tmpdir) / "output",
                require_driving_mode=True,
            )
            
            parser = TripDatasetParser(config)
            segments = parser.parse()
            
            # Should have one segment
            assert len(segments) > 1  # segments dict + metadata
            assert "metadata" in segments
            
            # Check segment data
            seg_keys = [k for k in segments.keys() if k != "metadata"]
            assert len(seg_keys) > 0
            
            seg = segments[seg_keys[0]]
            assert "speed" in seg
            assert "throttle" in seg
            assert "brake" in seg
            assert "angle" in seg
            assert len(seg["speed"]) > 0
