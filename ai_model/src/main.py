import cv2
from detection.hand_detector import HandDetector
from visualization.draw import draw_landmarks_on_image
from detection.landmarks import finger_state


def main():
    print("Starting the program...")

    detector = HandDetector(model_path="assets/hand_landmarker.task")

    # STEP 3: Load the input image.
    # Access the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")
    cv2.namedWindow("Hand Tracking", cv2.WINDOW_AUTOSIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        detection_result = detector.detect_for_video(frame)
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                state = finger_state(hand_landmarks)
                print(state)

            frame = draw_landmarks_on_image(frame, detection_result)

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
