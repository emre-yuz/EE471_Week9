import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import sys
import numpy as np
from POSE_LANDMARKER import draw_landmarks_on_image

def classify_hands(image_path):
    """
    Classifies hand position relative to shoulders using pose landmarks.
    
    Returns:
        "both" - both hands above shoulders
        "left" - only left hand above shoulder
        "right" - only right hand above shoulder
        "neither" or "None" - no hands above shoulders
    """
    # STEP 2: Create a PoseLandmarker object
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False) # Mask not needed for classification
    
    detector = vision.PoseLandmarker.create_from_options(options)
    
    # STEP 3: Load the input image
    image = mp.Image.create_from_file(image_path)

    # STEP 4: Detect pose landmarks
    detection_result = detector.detect(image)

    # STEP 5: Classification Logic
    if not detection_result.pose_landmarks: 
        return "None"

    # MediaPipe Tasks returns a list of landmarks for each person detected
    # We take the first person (index 0)
    landmarks = detection_result.pose_landmarks[0]
    
    # Landmark indices:
    # 11: left shoulder, 15: left wrist
    # 12: right shoulder, 16: right wrist
    # In MediaPipe, the Y-axis is inverted (0 is top, 1 is bottom)
    # Therefore, "Up" means a SMALLER Y-value than the shoulder
    left_wrist_up = landmarks[15].y < landmarks[11].y
    right_wrist_up = landmarks[16].y < landmarks[12].y

    if left_wrist_up and right_wrist_up:
        return "both"
    elif left_wrist_up:
        return "left"
    elif right_wrist_up:
        return "right"
    else:
        return "neither"

if __name__ == "__main__":
    image_path = "pose-1.jpg"
    
    result = classify_hands(image_path)    
    print("The pose is : "  + result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
