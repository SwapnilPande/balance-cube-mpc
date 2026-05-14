"""Headless video recording for cubli simulations.

Wraps `mujoco.Renderer` (offscreen renderer) and `imageio`'s ffmpeg writer
to stream frames to an mp4 file. Designed as a context manager so the
output file is always finalized, even on exceptions in the recording loop.

Headless notes:
  - On a server without a display, set `MUJOCO_GL=egl` before importing
    mujoco. The mujoco wheel ships with EGL support on Linux.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio.v2 as iio
import mujoco


class VideoRecorder:
    def __init__(
        self,
        model: mujoco.MjModel,
        output_path: str | Path,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
    ):
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._writer = iio.get_writer(
            str(output_path),
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,  # allow odd width/height
        )
        self._fps = fps
        self._closed = False

    def capture(self, data: Any) -> None:
        """Render the current MuJoCo state and append it as one video frame."""
        self._renderer.update_scene(data)
        frame = self._renderer.render()
        self._writer.append_data(frame)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._writer.close()
        finally:
            self._renderer.close()
            self._closed = True

    def __enter__(self) -> "VideoRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
