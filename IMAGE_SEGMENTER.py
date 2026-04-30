import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import math

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
IMAGE_FILE = 'face-3.png'
MODEL_FILE = 'selfie_segmenter.tflite'
MASK_COLOR = (255, 255, 255)  # White for foreground
BG_COLOR = (0, 0, 0)  # Black for background


def resize_and_show(image, window_name='Image'):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))

  cv2.imshow(window_name, img)


# Create a image segmenter instance with the image mode:
options = vision.ImageSegmenterOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_FILE),
    running_mode=vision.RunningMode.IMAGE,
    output_category_mask=True)

with vision.ImageSegmenter.create_from_options(options) as segmenter:

    # Load and display the original image
    original_image = cv2.imread(IMAGE_FILE)
    if original_image is None:
        raise FileNotFoundError(f"Could not read image file: {IMAGE_FILE}")
    print(f'Original image: {IMAGE_FILE}')
    resize_and_show(original_image, 'Original Image')

    # Create the MediaPipe image file that will be segmented
    image = mp.Image.create_from_file(IMAGE_FILE)

    # Retrieve the masks for the segmented image
    segmentation_result = segmenter.segment(image)
    category_mask = segmentation_result.category_mask

    # Generate solid color images for showing the output segmentation mask.
    image_data = image.numpy_view()
    if image_data.shape[2] == 4:
        image_data = cv2.cvtColor(image_data, cv2.COLOR_RGBA2RGB)
    fg_image = np.zeros(image_data.shape, dtype=np.uint8)
    fg_image[:] = MASK_COLOR
    bg_image = np.zeros(image_data.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR

    condition = category_mask.numpy_view().squeeze(-1) > 0.2

    output_image = np.where(condition[..., None], fg_image, bg_image)

    print(f'Segmentation mask of {IMAGE_FILE}:')
    resize_and_show(output_image, 'Segmented Mask')

    # Wait for key press to close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    