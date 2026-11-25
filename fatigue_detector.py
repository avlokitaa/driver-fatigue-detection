"""
Temporal decision engine for driver drowsiness and yawning detection.
Uses wall-clock timestamps to decouple logic from variable webcam FPS.
"""

from collections import deque
from dataclasses import dataclass
import time


@dataclass
class FatigueStatus:
    face_detected: bool
    ear: float
    mor: float
    eye_state: str          # "OPEN" | "CLOSED" | "N/A"
    mouth_state: str        # "NORMAL" | "YAWNING" | "N/A"
    yawns_in_window: int
    overall_status: str     # "NORMAL" | "WARNING" | "DROWSINESS ALERT" | "NO FACE"
    banner_message: str
    alarm_active: bool


class FatigueDetector:
    """Evaluates EAR and MOR over time to detect prolonged eye closure and repeated yawning."""

    def __init__(
        self,
        ear_threshold: float = 0.22,
        mor_threshold: float = 0.55,
        eye_closed_seconds: float = 1.8,
        yawn_min_seconds: float = 2.0,
        yawn_window_seconds: float = 180.0,
        yawn_count_threshold: int = 3,
        recovery_seconds: float = 0.5,
    ):
        self.ear_threshold = ear_threshold
        self.mor_threshold = mor_threshold
        self.eye_closed_seconds = eye_closed_seconds
        self.yawn_min_seconds = yawn_min_seconds
        self.yawn_window_seconds = yawn_window_seconds
        self.yawn_count_threshold = yawn_count_threshold
        self.recovery_seconds = recovery_seconds

        # Internal temporal state
        self._eye_closed_start: float | None = None
        self._yawn_start: float | None = None
        self._yawn_counted_for_current_open: bool = False
        self._yawn_timestamps: deque[float] = deque()

        self._alarm_active: bool = False
        self._normal_since: float | None = None
        self._recent_yawn_banner_until: float = 0.0

    def update(self, ear: float | None, mor: float | None, now: float | None = None) -> FatigueStatus:
        """
        Updates detector state for the current frame.
        If ear or mor is None (missing face), resets active closure/yawn timers
        without counting the missing frame as a fatigue cue.
        """
        current_time = time.monotonic() if now is None else now

        # Prune expired yawns outside the rolling time window
        while self._yawn_timestamps and (current_time - self._yawn_timestamps[0]) > self.yawn_window_seconds:
            self._yawn_timestamps.popleft()

        # Handle missing face gracefully
        if ear is None or mor is None:
            self._eye_closed_start = None
            self._yawn_start = None
            self._yawn_counted_for_current_open = False
            return FatigueStatus(
                face_detected=False,
                ear=0.0,
                mor=0.0,
                eye_state="N/A",
                mouth_state="N/A",
                yawns_in_window=len(self._yawn_timestamps),
                overall_status="NO FACE",
                banner_message="NO FACE DETECTED",
                alarm_active=self._alarm_active,
            )

        # 1. Eye closure evaluation
        prolonged_eye_closure = False
        if ear < self.ear_threshold:
            eye_state = "CLOSED"
            if self._eye_closed_start is None:
                self._eye_closed_start = current_time
            elif (current_time - self._eye_closed_start) >= self.eye_closed_seconds:
                prolonged_eye_closure = True
        else:
            eye_state = "OPEN"
            self._eye_closed_start = None

        # 2. Yawn evaluation (latched per continuous mouth opening)
        if mor > self.mor_threshold:
            mouth_state = "YAWNING"
            if self._yawn_start is None:
                self._yawn_start = current_time
                self._yawn_counted_for_current_open = False
            elif (
                not self._yawn_counted_for_current_open
                and (current_time - self._yawn_start) >= self.yawn_min_seconds
            ):
                self._yawn_timestamps.append(current_time)
                self._yawn_counted_for_current_open = True
                self._recent_yawn_banner_until = current_time + 2.5
        else:
            mouth_state = "NORMAL"
            self._yawn_start = None
            self._yawn_counted_for_current_open = False

        yawns_count = len(self._yawn_timestamps)
        repeated_yawning = yawns_count >= self.yawn_count_threshold

        # 3. Alarm triggering & hysteresis recovery
        raw_alert_condition = prolonged_eye_closure or repeated_yawning

        if raw_alert_condition:
            self._alarm_active = True
            self._normal_since = None
        elif self._alarm_active:
            # Require sustained normal behavior before silencing alarm
            if self._normal_since is None:
                self._normal_since = current_time
            elif (current_time - self._normal_since) >= self.recovery_seconds:
                self._alarm_active = False
                self._normal_since = None
                # Clear yawn history after repeated-yawn alert resolves to avoid immediate re-trigger
                if repeated_yawning:
                    self._yawn_timestamps.clear()
                    yawns_count = 0

        # 4. Determine overall status and banner text
        if self._alarm_active:
            overall_status = "DROWSINESS ALERT"
            if prolonged_eye_closure:
                banner_message = "DROWSINESS ALERT: EYES CLOSED"
            else:
                banner_message = f"DROWSINESS ALERT: {yawns_count} YAWNS DETECTED"
        elif current_time < self._recent_yawn_banner_until or yawns_count > 0:
            overall_status = "WARNING"
            banner_message = "YAWN DETECTED" if current_time < self._recent_yawn_banner_until else ""
        else:
            overall_status = "NORMAL"
            banner_message = ""

        return FatigueStatus(
            face_detected=True,
            ear=ear,
            mor=mor,
            eye_state=eye_state,
            mouth_state=mouth_state,
            yawns_in_window=yawns_count,
            overall_status=overall_status,
            banner_message=banner_message,
            alarm_active=self._alarm_active,
        )