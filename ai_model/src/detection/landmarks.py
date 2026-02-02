from typing import Tuple
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark
from enums.HandLandmarkEnum import HandLandmarkEnum


def get_landmarks(landmarks, landmark_id: HandLandmarkEnum):
    """
    Returns a single landmark by ID
    """
    return landmarks[landmark_id]


def _thump_is_extended(landmarks, handedness: str) -> bool:
    """
    Heuristic for thumb "up"/extended based on tip.x vs IP/MCP x.
    handedness: "Left" or "Right" (string)
    Returns True if thumb appears extended away from palm.
    """
    tip = get_landmarks(landmarks, HandLandmarkEnum.THUMB_TIP)
    ip = get_landmarks(landmarks, HandLandmarkEnum.THUMB_IP)
    mcp = get_landmarks(landmarks, HandLandmarkEnum.THUMB_MCP)
    if handedness is None:
        return abs(tip.x - ip.x) > 0.02

    if handedness.lower().startswith("r"):
        return tip.x > ip.x > mcp.x
    else:
        return tip.x < ip.x < mcp.x


def _finger_is_up(landmarks, tip_id: int, pip_id: int, mcp_id: int) -> bool:
    """
    Checks if a finger is up by comparing Y coordinates
    NOTE: Y axis is inverted (top = smaller value)
    """
    tip = landmarks[tip_id]
    pip = landmarks[pip_id]
    mcp = landmarks[mcp_id]
    return tip.y < pip.y < mcp.y


def finger_state(landmarks, handedness_label: str = None) -> dict[str, bool]:
    """
    Returns a dictionary with boolean state whether each finger is 'up' / extended.
    landmarks: sequence-like of 21 normalized landmarks (each has .x, .y, .z).
    handedness_label: "Left" or "Right" or None (used for thumb logic).
    """
    return {
        "thumb": _thump_is_extended(landmarks, handedness_label),
        "index": _finger_is_up(
            landmarks,
            HandLandmark.INDEX_FINGER_TIP,
            HandLandmark.INDEX_FINGER_PIP,
            HandLandmark.INDEX_FINGER_MCP,
        ),
        "middle": _finger_is_up(
            landmarks,
            HandLandmark.MIDDLE_FINGER_TIP,
            HandLandmark.MIDDLE_FINGER_PIP,
            HandLandmark.MIDDLE_FINGER_MCP,
        ),
        "ring": _finger_is_up(
            landmarks,
            HandLandmark.RING_FINGER_TIP,
            HandLandmark.RING_FINGER_PIP,
            HandLandmark.RING_FINGER_MCP,
        ),
        "pinky": _finger_is_up(
            landmarks,
            HandLandmark.PINKY_TIP,
            HandLandmark.PINKY_PIP,
            HandLandmark.PINKY_MCP,
        ),
    }


# ---------- Gestures ----------


def is_fist(landmarks, handedness_label: str = None) -> bool:
    """
    Checks if the hand is a fist: all four fingers (index, middle, ring, pinky) down.
    Thumb is ignored (thumb up = thumbs up, still a fist).
    """
    states = finger_state(landmarks=landmarks, handedness_label=handedness_label)
    return not any((states["index"], states["middle"], states["ring"], states["pinky"]))


def is_open_palm(landmarks, handedness_label: str = None) -> bool:
    """
    Checks if the hand is an open palm
    """
    states = finger_state(landmarks=landmarks, handedness_label=handedness_label)
    return all(states.values())


def is_point(landmarks, handedness_label: str = None) -> bool:
    """
    Returns True if the hand is pointing with the index finger.
    Heuristic: index up, middle/ring/pinky down. Thumb ignored (can be either).
    """
    states = finger_state(landmarks=landmarks, handedness_label=handedness_label)
    return states["index"] and (not states["middle"]) and (not states["ring"])


def detect_gesture(landmarks, handedness_label: str = None) -> Tuple[str, float]:
    """
    Returns (gesture_name, confidence) based on rule heuristics.
    Confidence is 0.0..1.0 (higher = more confident).
    Gesture names: "OPEN_PALM", "FIST", "POINT", "UNKNOWN"
    """
    states = finger_state(landmarks=landmarks, handedness_label=handedness_label)
    four_fingers_down = not any(
        (states["index"], states["middle"], states["ring"], states["pinky"])
    )
    up_count = sum(1 for v in states.values() if v)

    # FIST first: all four fingers closed (thumb up or down = still fist)
    if four_fingers_down:
        return "FIST", 0.80

    # OPEN_PALM: 4 or 5 fingers extended
    if up_count >= 4:
        return "OPEN_PALM", up_count / 5.0

    # POINT: index up, middle/ring/pinky down
    if (
        states["index"]
        and (not states["middle"])
        and (not states["ring"])
        and (not states["pinky"])
    ):
        return "POINT", 0.85

    return "UNKNOWN", max(0.15, up_count / 5.0)
