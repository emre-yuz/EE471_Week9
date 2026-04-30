import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
IMAGE_FILE = 'face-2.png'
MODEL_FILE = 'efficientnet_lite0_imgcls.tflite'

def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))

  cv2.imshow('Image', img)


# Load the image file.
image = cv2.imread(IMAGE_FILE)
if image is None:
  raise FileNotFoundError(f"Could not read image file: {IMAGE_FILE}")

resize_and_show(image)

# Create an ImageClassifier object.
base_options = python.BaseOptions(model_asset_path=MODEL_FILE)
options = vision.ImageClassifierOptions(base_options=base_options, max_results=4)
classifier = vision.ImageClassifier.create_from_options(options)

# Load the input image for MediaPipe.
mp_image = mp.Image.create_from_file(IMAGE_FILE)

# Classify the input image.
classification_result = classifier.classify(mp_image)

# Print the top results.
if classification_result.classifications:
  for idx, category in enumerate(classification_result.classifications[0].categories):
    print(f"{idx + 1}. {category.category_name} ({category.score:.2f})")
else:
  print('No classification results.')
