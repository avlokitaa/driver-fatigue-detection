"""
Configuration parameters for Real-Time Driver Fatigue Detection.
All thresholds are starting values intended for empirical calibration.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    # Base directory
    BASE_DIR: Path = Path(__file__).resolve().parent

    # Camera settings
    CAMERA_INDEX: int = 0
    FRAME_WIDTH: int = 640
    FLIP_HORIZONTAL: bool = True

    # MediaPipe Face Landmarker (Tasks API) settings
    FACE_LANDMARKER_MODEL_PATH: Path = BASE_DIR / "models" / "face_landmarker.task"
    MAX_NUM_FACES: int = 1
    MIN_DETECTION_CONFIDENCE: float = 0.5
    MIN_TRACKING_CONFIDENCE: float = 0.5
    MIN_PRESENCE_CONFIDENCE: float = 0.5

    # MediaPipe Face Mesh Landmark Indices (indices 0-467 match canonical 468-point topology)
    # Ordered as [p1 (outer), p2 (top-outer), p3 (top-inner), p4 (inner), p5 (bottom-inner), p6 (bottom-outer)]
    RIGHT_EYE_INDICES: tuple[int, ...] = (33, 160, 158, 133, 153, 144)
    LEFT_EYE_INDICES: tuple[int, ...] = (362, 385, 387, 263, 373, 380)

    # Mouth landmarks: inner upper lip (13), inner lower lip (14), left corner (78), right corner (308)
    MOUTH_TOP_INDEX: int = 13
    MOUTH_BOTTOM_INDEX: int = 14
    MOUTH_LEFT_INDEX: int = 78
    MOUTH_RIGHT_INDEX: int = 308

    # Detection thresholds (starting values; calibrate per user/camera angle)
    EAR_THRESHOLD: float = 0.22
    MOR_THRESHOLD: float = 0.55

    # Temporal conditions (in seconds, independent of FPS fluctuations)
    EYE_CLOSED_SECONDS: float = 1.8       # Prolonged closure duration for drowsiness alert
    YAWN_MIN_SECONDS: float = 2.0         # Minimum duration to distinguish yawn from speech
    YAWN_WINDOW_SECONDS: float = 180.0    # Rolling window (3 minutes) for repeated yawns
    YAWN_COUNT_THRESHOLD: int = 3         # Number of yawns within window to trigger alert
    RECOVERY_SECONDS: float = 0.5         # Sustained normal state required to clear alarm

    # Audio settings
    ALARM_SOUND_PATH: Path = BASE_DIR / "assets" / "alarm.wav"
    ALARM_FREQUENCY_HZ: int = 880