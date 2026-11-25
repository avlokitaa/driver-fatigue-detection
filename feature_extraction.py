"""
Geometric feature extraction functions for Eye Aspect Ratio (EAR)
and Mouth Opening Ratio (MOR).
"""

import numpy as np


def compute_ear(eye_points: np.ndarray) -> float:
    """
    Computes Eye Aspect Ratio (EAR) from 6 ordered 2D eye landmarks:
        p1: outer corner, p2: upper-outer, p3: upper-inner,
        p4: inner corner, p5: lower-inner, p6: lower-outer.

    Formula:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    if eye_points.shape != (6, 2):
        raise ValueError(f"Expected eye_points of shape (6, 2), got {eye_points.shape}")

    vertical_1 = np.linalg.norm(eye_points[1] - eye_points[5])
    vertical_2 = np.linalg.norm(eye_points[2] - eye_points[4])
    horizontal = np.linalg.norm(eye_points[0] - eye_points[3])

    if horizontal < 1e-6:
        return 0.0

    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def compute_average_ear(right_eye: np.ndarray, left_eye: np.ndarray) -> tuple[float, float, float]:
    """Computes right EAR, left EAR, and their arithmetic mean."""
    right_ear = compute_ear(right_eye)
    left_ear = compute_ear(left_eye)
    avg_ear = (right_ear + left_ear) / 2.0
    return right_ear, left_ear, avg_ear


def compute_mor(
    mouth_top: np.ndarray,
    mouth_bottom: np.ndarray,
    mouth_left: np.ndarray,
    mouth_right: np.ndarray,
) -> float:
    """
    Computes Mouth Opening Ratio (MOR) using inner lip vertical distance
    and mouth corner horizontal distance.

    Formula:
        MOR = ||p_top - p_bottom|| / ||p_left - p_right||
    """
    vertical = np.linalg.norm(mouth_top - mouth_bottom)
    horizontal = np.linalg.norm(mouth_left - mouth_right)

    if horizontal < 1e-6:
        return 0.0

    return float(vertical / horizontal)