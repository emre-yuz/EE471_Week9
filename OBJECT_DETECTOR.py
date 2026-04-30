import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

IMAGE_FILE = 'face-1.png'

import cv2

img = cv2.imread(IMAGE_FILE)
if img is None:
    raise FileNotFoundError(f"Could not read image file: {IMAGE_FILE}")

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

