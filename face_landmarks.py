"""
MediaPipe Tasks Face Landmarker wrapper for extracting 2D pixel coordinates of eyes and mouth.
Compatible with Python 3.13 and MediaPipe >= 0.10.30.
"""

from dataclasses import dataclass
from pathlib import Path
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np


@dataclass
class FacialLandmarks:
    right_eye: np.ndarray     # Shape (6, 2)
    left_eye: np.ndarray      # Shape (6, 2)
    mouth_top: np.ndarray     # Shape (2,)
    mouth_bottom: np.ndarray  # Shape (2,)
    mouth_left: np.ndarray    # Shape (2,)
    mouth_right: np.ndarray   # Shape (2,)


class FaceLandmarkDetector:
    """Extracts facial landmarks using MediaPipe Tasks FaceLandmarker in VIDEO mode on CPU."""

    def __init__(
        self,
        model_path: Path | str,
        right_eye_indices: tuple[int, ...],
        left_eye_indices: tuple[int, ...],
        mouth_top_idx: int,
        mouth_bottom_idx: int,
        mouth_left_idx: int,
        mouth_right_idx: int,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
    ):
        resolved_model = Path(model_path)
        if not resolved_model.is_file():
            raise RuntimeError(
                f"MediaPipe Face Landmarker model not found at: {resolved_model}\n"
                "Download 'face_landmarker.task' and place it inside the 'models/' directory."
            )

        self.right_eye_indices = right_eye_indices
        self.left_eye_indices = left_eye_indices
        self.mouth_top_idx = mouth_top_idx
        self.mouth_bottom_idx = mouth_bottom_idx
        self.mouth_left_idx = mouth_left_idx
        self.mouth_right_idx = mouth_right_idx
        self._last_timestamp_ms: int = -1

        base_options = mp_python.BaseOptions(
            model_asset_path=str(resolved_model),
            delegate=mp_python.BaseOptions.Delegate.CPU,
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def _next_timestamp_ms(self) -> int:
        """Returns a strictly monotonically increasing timestamp in milliseconds."""
        now_ms = int(time.monotonic() * 1000)
        if now_ms <= self._last_timestamp_ms:
            now_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = now_ms
        return now_ms

    def detect(self, rgb_frame: np.ndarray) -> FacialLandmarks | None:
        """
        Processes an RGB frame and returns pixel coordinates for target landmarks.
        Returns None if no face is detected.
        """
        h, w = rgb_frame.shape[:2]
        contiguous_rgb = np.ascontiguousarray(rgb_frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=contiguous_rgb)

        timestamp_ms = self._next_timestamp_ms()
        results = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not results.face_landmarks:
            return None

        face_landmarks = results.face_landmarks[0]

        def to_pixel(idx: int) -> np.ndarray:
            lm = face_landmarks[idx]
            return np.array([lm.x * w, lm.y * h], dtype=np.float64)

        right_eye = np.array([to_pixel(i) for i in self.right_eye_indices], dtype=np.float64)
        left_eye = np.array([to_pixel(i) for i in self.left_eye_indices], dtype=np.float64)

        return FacialLandmarks(
            right_eye=right_eye,
            left_eye=left_eye,
            mouth_top=to_pixel(self.mouth_top_idx),
            mouth_bottom=to_pixel(self.mouth_bottom_idx),
            mouth_left=to_pixel(self.mouth_left_idx),
            mouth_right=to_pixel(self.mouth_right_idx),
        )

    def close(self) -> None:
        """Frees MediaPipe FaceLandmarker resources."""
        if self._landmarker is not None:
            self._landmarker.close()