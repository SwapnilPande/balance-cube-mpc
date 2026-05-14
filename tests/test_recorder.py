"""Tests for headless video recording."""
import math
from pathlib import Path

import imageio.v3 as iio
import pytest

from cubli_mpc.config import HardwareConfig, SimConfig
from cubli_mpc.sim.env import CubliEnv
from cubli_mpc.sim.recorder import VideoRecorder


def _make_hw() -> HardwareConfig:
    return HardwareConfig(
        cube_side_length_m=0.10,
        cube_mass_kg=0.40,
        wheel_mass_kg=0.10,
        wheel_radius_m=0.035,
        wheel_thickness_m=0.010,
        motor_max_torque_nm=0.20,
        motor_max_speed_rad_s=600.0,
        motor_torque_constant=0.01,
        edge_bearing_damping=1e-4,
        wheel_bearing_damping=1e-5,
    )


def test_recorder_writes_mp4(tmp_path: Path):
    env = CubliEnv(_make_hw(), SimConfig())
    env.reset(theta0=0.05)
    out = tmp_path / "out.mp4"
    with VideoRecorder(env.model, out, fps=30, width=320, height=240) as rec:
        for _ in range(10):
            for _ in range(10):  # 10 ms per "frame" at dt=1ms
                env.step()
            rec.capture(env.data)
    assert out.exists()
    assert out.stat().st_size > 0


def test_recorder_video_is_readable(tmp_path: Path):
    env = CubliEnv(_make_hw(), SimConfig())
    env.reset(theta0=math.radians(5.0))
    out = tmp_path / "readable.mp4"
    n_frames = 15
    with VideoRecorder(env.model, out, fps=30, width=320, height=240) as rec:
        for _ in range(n_frames):
            for _ in range(10):
                env.step()
            rec.capture(env.data)
    # Round-trip through imageio: confirms it's a real, valid mp4 file
    frames = iio.imread(out, plugin="pyav")
    assert frames.shape[0] >= n_frames - 2  # ffmpeg may drop a tail frame
    assert frames.shape[1] == 240
    assert frames.shape[2] == 320
    assert frames.shape[3] == 3


def test_recorder_context_manager_closes_on_exception(tmp_path: Path):
    """Even when the body of the with-block raises, the mp4 should still be
    finalized so partial recordings are recoverable.
    """
    env = CubliEnv(_make_hw(), SimConfig())
    env.reset()
    out = tmp_path / "partial.mp4"
    with pytest.raises(RuntimeError):
        with VideoRecorder(env.model, out, fps=30, width=320, height=240) as rec:
            for _ in range(5):
                for _ in range(10):
                    env.step()
                rec.capture(env.data)
            raise RuntimeError("simulated crash")
    assert out.exists()
    assert out.stat().st_size > 0
