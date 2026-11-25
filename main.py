"""
Main application entry point for Real-Time Driver Fatigue Detection.
Run with: python main.py
Press 'q' or 'ESC' to exit cleanly.
"""

import sys
import cv2
import numpy as np

from alarm import AlarmManager
from camera import WebcamStream
from config import AppConfig
from face_landmarks import FaceLandmarkDetector, FacialLandmarks
from fatigue_detector import FatigueDetector, FatigueStatus
from feature_extraction import compute_average_ear, compute_mor


def draw_overlays(
    frame: np.ndarray,
    landmarks: FacialLandmarks | None,
    status: FatigueStatus,
    fps: float,
    cfg: AppConfig,
) -> None:
    """Draws eye/mouth landmarks and a clean status HUD onto the frame."""
    h, w = frame.shape[:2]

    # Draw eye contours and mouth crosshairs when face is detected
    if landmarks is not None:
        eye_color = (0, 0, 255) if status.eye_state == "CLOSED" else (0, 255, 0)
        mouth_color = (0, 165, 255) if status.mouth_state == "YAWNING" else (0, 255, 0)

        cv2.polylines(frame, [landmarks.right_eye.astype(np.int32)], isClosed=True, color=eye_color, thickness=1)
        cv2.polylines(frame, [landmarks.left_eye.astype(np.int32)], isClosed=True, color=eye_color, thickness=1)

        cv2.line(
            frame,
            tuple(landmarks.mouth_top.astype(np.int32)),
            tuple(landmarks.mouth_bottom.astype(np.int32)),
            mouth_color,
            2,
        )
        cv2.line(
            frame,
            tuple(landmarks.mouth_left.astype(np.int32)),
            tuple(landmarks.mouth_right.astype(np.int32)),
            mouth_color,
            1,
        )

    # Semi-transparent top-left telemetry panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (285, 195), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    status_colors = {
        "NORMAL": (0, 220, 0),
        "WARNING": (0, 200, 255),
        "DROWSINESS ALERT": (0, 0, 255),
        "NO FACE": (180, 180, 180),
    }
    status_color = status_colors.get(status.overall_status, (255, 255, 255))

    cv2.putText(frame, f"STATUS: {status.overall_status}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    cv2.putText(frame, f"EAR: {status.ear:.3f} (th: {cfg.EAR_THRESHOLD:.2f})", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    cv2.putText(frame, f"MOR: {status.mor:.3f} (th: {cfg.MOR_THRESHOLD:.2f})", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    cv2.putText(frame, f"Eyes: {status.eye_state}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    cv2.putText(frame, f"Mouth: {status.mouth_state}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    cv2.putText(
        frame,
        f"Yawns ({int(cfg.YAWN_WINDOW_SECONDS)}s): {status.yawns_in_window}/{cfg.YAWN_COUNT_THRESHOLD}",
        (20, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
    )
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 188), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Bottom warning banner when active
    if status.banner_message:
        banner_color = (0, 0, 220) if status.alarm_active else (0, 140, 255)
        cv2.rectangle(frame, (0, h - 45), (w, h), banner_color, -1)
        cv2.putText(
            frame,
            status.banner_message,
            (20, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )


def main() -> None:
    cfg = AppConfig()
    camera = None
    detector = None
    alarm = None

    try:
        camera = WebcamStream(
            camera_index=cfg.CAMERA_INDEX,
            target_width=cfg.FRAME_WIDTH,
            flip_horizontal=cfg.FLIP_HORIZONTAL,
        )
        detector = FaceLandmarkDetector(
            model_path=cfg.FACE_LANDMARKER_MODEL_PATH,
            right_eye_indices=cfg.RIGHT_EYE_INDICES,
            left_eye_indices=cfg.LEFT_EYE_INDICES,
            mouth_top_idx=cfg.MOUTH_TOP_INDEX,
            mouth_bottom_idx=cfg.MOUTH_BOTTOM_INDEX,
            mouth_left_idx=cfg.MOUTH_LEFT_INDEX,
            mouth_right_idx=cfg.MOUTH_RIGHT_INDEX,
            max_num_faces=cfg.MAX_NUM_FACES,
            min_detection_confidence=cfg.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=cfg.MIN_TRACKING_CONFIDENCE,
            min_presence_confidence=cfg.MIN_PRESENCE_CONFIDENCE,
        )
        fatigue_engine = FatigueDetector(
            ear_threshold=cfg.EAR_THRESHOLD,
            mor_threshold=cfg.MOR_THRESHOLD,
            eye_closed_seconds=cfg.EYE_CLOSED_SECONDS,
            yawn_min_seconds=cfg.YAWN_MIN_SECONDS,
            yawn_window_seconds=cfg.YAWN_WINDOW_SECONDS,
            yawn_count_threshold=cfg.YAWN_COUNT_THRESHOLD,
            recovery_seconds=cfg.RECOVERY_SECONDS,
        )
        alarm = AlarmManager(
            sound_path=cfg.ALARM_SOUND_PATH,
            fallback_freq_hz=cfg.ALARM_FREQUENCY_HZ,
        )

        window_name = "Real-Time Driver Fatigue Detection (Press 'q' or ESC to quit)"
        while True:
            ok, bgr_frame, rgb_frame, fps = camera.read()
            if not ok or bgr_frame is None or rgb_frame is None:
                print("[ERROR] Failed to read frame from webcam. Exiting.")
                break

            landmarks = detector.detect(rgb_frame)
            if landmarks is not None:
                _, _, avg_ear = compute_average_ear(landmarks.right_eye, landmarks.left_eye)
                mor = compute_mor(
                    landmarks.mouth_top,
                    landmarks.mouth_bottom,
                    landmarks.mouth_left,
                    landmarks.mouth_right,
                )
                status = fatigue_engine.update(avg_ear, mor)
            else:
                status = fatigue_engine.update(None, None)

            if status.alarm_active:
                alarm.start()
            else:
                alarm.stop()

            draw_overlays(bgr_frame, landmarks, status, fps, cfg)
            cv2.imshow(window_name, bgr_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                break

    except RuntimeError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if alarm is not None:
            alarm.cleanup()
        if detector is not None:
            detector.close()
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()