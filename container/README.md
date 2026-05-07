# Pose Estimation Container

This project contains a containerized pose estimation application using MediaPipe.

## Prerequisites

- Docker installed on your system
- Docker Compose (optional, for easier management)

## Building the Container

### Using Docker directly:

```bash
cd container
docker build -f Dockerfile -t pose-estimation ..
```

### Using Docker Compose:

```bash
cd container
docker-compose build
```

## Running the Application

### Using Docker directly:

```bash
# Run with default image (pose-1.jpg)
docker run --rm pose-estimation

# Run with a specific image
docker run --rm -v $(pwd)/..:/app/data pose-estimation python POSE_ESTIMATION.py /app/data/your-image.jpg
```

### Using Docker Compose:

```bash
# From the container directory
cd container
docker-compose up

# Run and remove container after execution
docker-compose up --rm
```

## File Structure

- `POSE_ESTIMATION.py` - Main pose estimation script (in parent directory)
- `POSE_LANDMARKER.py` - Pose landmarking utilities (in parent directory)
- `run_pose_clean.py` - Clean runner that suppresses MediaPipe warnings (in parent directory)
- `*.task` - MediaPipe model files (in parent directory)
- `*.tflite` - TensorFlow Lite model files (in parent directory)
- `*.jpg/*.png` - Sample input images (in parent directory)

## Customization

### Changing the Input Image

The default script processes `pose-1.jpg`. To use a different image:

1. Place your image in the project directory
2. Modify the `image_path` variable in `POSE_ESTIMATION.py`
3. Or pass the image path as an argument when running the container

### Using Different Models

The application uses `pose_landmarker_full.task` by default. You can modify the model path in the scripts to use different MediaPipe models.

## Troubleshooting

- If you encounter OpenCV GUI issues in containers, the application may need to be run in headless mode
- Ensure all model files (*.task, *.tflite) are present in the container
- For GPU acceleration, you may need to use a different base image with CUDA support