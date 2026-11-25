# Real-Time Driver Fatigue Detection

A real-time, CPU-only computer vision prototype that monitors eye closure and yawning using a standard laptop webcam to detect visible cues of driver drowsiness and trigger a non-blocking audio alarm.

> **Disclaimer:** This system is an academic warning aid prototype. It is not a medically validated diagnostic instrument and does not detect all forms of driver impairment (e.g., microsleeps with open eyes, cognitive distraction, or intoxication).

## Features
- **CPU Real-Time Processing:** Uses OpenCV and MediaPipe Face Mesh without requiring a GPU or external sensors.
- **Geometric Facial Ratios:** Computes averaged Eye Aspect Ratio (EAR) and inner-lip Mouth Opening Ratio (MOR).
- **FPS-Independent Temporal Logic:** Uses wall-clock timestamps (`time.monotonic()`) rather than raw frame counts so detection durations stay accurate regardless of webcam frame rate fluctuations.
- **False-Alarm Mitigation:** Ignores normal blinks (100–400 ms), brief speech movements, and frames where no face is detected.
- **Non-Blocking Audio Alerts:** Uses `pygame.mixer` to play a looped warning tone (`assets/alarm.wav`, auto-generated on first run) without freezing video capture.

## Installation

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux / macOS:
   source venv/bin/activate