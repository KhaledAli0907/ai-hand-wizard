import mediapipe as mp
import numpy as np
import cv2
import config
from detection.landmarks import finger_state


mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles


def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

        # Draw handedness (Left/Right) on the hand.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - config.MARGIN
        cv2.putText(
            annotated_image,
            f"{handedness[0].category_name}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            config.FONT_SIZE,
            config.HANDEDNESS_TEXT_COLOR,
            config.FONT_THICKNESS,
            cv2.LINE_AA,
        )

    # Draw finger state in 2 fixed lines at the top (first hand only).
    if hand_landmarks_list:
        hand_landmarks = hand_landmarks_list[0]
        state = finger_state(hand_landmarks)
        parts = [
            f"{name.capitalize()}: {'up' if is_up else 'down'}"
            for name, is_up in state.items()
        ]
        line1 = "  ".join(parts[:2])  # Index, Middle
        line2 = "  ".join(parts[2:])  # Ring, Pinky

        top_x, line_height = config.MARGIN, 25
        cv2.putText(
            annotated_image,
            line1,
            (top_x, line_height),
            cv2.FONT_HERSHEY_DUPLEX,
            config.FONT_SIZE,
            config.HANDEDNESS_TEXT_COLOR,
            config.FONT_THICKNESS,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated_image,
            line2,
            (top_x, line_height + 25),
            cv2.FONT_HERSHEY_DUPLEX,
            config.FONT_SIZE,
            config.HANDEDNESS_TEXT_COLOR,
            config.FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated_image
