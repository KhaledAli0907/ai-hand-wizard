from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark
from enums import HandLandmarkEnum


def get_landmarks(landmarks, landmark_id: HandLandmarkEnum):
    """
    Returns a single landmark by ID
    """
    return landmarks[landmark_id]


def is_finger_up(landmarks, tip_id: int, mcp_id: int) -> bool:
    """
    Checks if a finger is up by comparing Y coordinates
    NOTE: Y axis is inverted (top = smaller value)
    """
    tip = landmarks[tip_id]
    mcp = landmarks[mcp_id]
    return tip.y < mcp.y


def finger_state(landmarks) -> dict[str, bool]:
    """
    Returns a dict of finger states (up/down)
    """
    return {
        "index": is_finger_up(
            landmarks, HandLandmark.INDEX_FINGER_TIP, HandLandmark.INDEX_FINGER_MCP
        ),
        "middle": is_finger_up(
            landmarks, HandLandmark.MIDDLE_FINGER_TIP, HandLandmark.MIDDLE_FINGER_MCP
        ),
        "ring": is_finger_up(
            landmarks, HandLandmark.RING_FINGER_TIP, HandLandmark.RING_FINGER_MCP
        ),
        "pinky": is_finger_up(
            landmarks, HandLandmark.PINKY_TIP, HandLandmark.PINKY_MCP
        ),
    }
