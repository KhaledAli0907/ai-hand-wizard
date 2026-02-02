

from pyexpat import model
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core import image
import numpy as np

class HandDetector:
    def __init__(self, model_path: str, num_hands: int = 2) -> None:
        base_options = python.BaseOptions(model_asset_path = model_path)
        options = vision.HandLandmarkerOptions(
            base_options = base_options,
            num_hands = num_hands,
            running_mode = vision.RunningMode.VIDEO
        )
        
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.frame_count = 0
        
    def detect_for_video(self, frame_bgr: np.ndarray) -> vision.HandLandmarkerResult:
        """
        Takes a BGR frame from OpenCV and returns MediaPipe detection result
        """
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        mp_image = image.Image(
            image.ImageFormat.SRGB,
            np.ascontiguousarray(rgb_frame),
        )
        
        timestamp_ms = int(self.frame_count * 1000 / 30)
        self.frame_count += 1
        
        result = self.detector.detect_for_video(mp_image, timestamp_ms)
        return result