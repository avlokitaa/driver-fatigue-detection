"""
Webcam acquisition, preprocessing, and FPS monitoring module.
"""

import time
import cv2
import numpy as np


class WebcamStream:
    """Manages OpenCV VideoCapture with graceful error handling and FPS tracking."""

    def __init__(self, camera_index: int = 0, target_width: int = 640, flip_horizontal: bool = True):
        self.camera_index = camera_index
        self.target_width = target_width
        self.flip_horizontal = flip_horizontal
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Unable to access webcam at index {self.camera_index}. "
                "Verify camera permissions and ensure no other application is using it."
            )

        self._prev_time = time.monotonic()
        self._fps = 0.0

    def read(self) -> tuple[bool, np.ndarray | None, np.ndarray | None, float]:
        """
        Captures a frame, resizes it for CPU efficiency, and converts BGR to RGB.

        Returns:
            (success, bgr_frame, rgb_frame, current_fps)
        """
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None, None, self._fps

        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)

        # Resize maintaining aspect ratio if wider than target_width
        h, w = frame.shape[:2]
        if w > self.target_width:
            scale = self.target_width / float(w)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (self.target_width, new_h), interpolation=cv2.INTER_AREA)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Exponential moving average for smooth FPS readout
        now = time.monotonic()
        dt = now - self._prev_time
        self._prev_time = now
        if dt > 0:
            instant_fps = 1.0 / dt
            self._fps = instant_fps if self._fps == 0.0 else (0.9 * self._fps + 0.1 * instant_fps)

        return True, frame, rgb_frame, self._fps

    def release(self) -> None:
        """Releases webcam hardware resources."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()