#!/usr/bin/env python3
"""
Clean runner for pose estimation - suppresses MediaPipe warnings
"""
import subprocess
import sys
import os

def run_clean_pose_estimation():
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, 'POSE_ESTIMATION.py')

    # Run the pose estimation script and capture output
    result = subprocess.run(
        [sys.executable, script_path] + sys.argv[1:],
        capture_output=True,
        text=True,
        cwd=script_dir
    )

    # Filter out warning lines and print only the result
    lines = result.stdout.strip().split('\n')
    for line in lines:
        if not (line.startswith('INFO:') or line.startswith('W0000') or 'inference_feedback_manager' in line or 'landmark_projection_calculator' in line):
            print(line)

    # Print any actual errors (not warnings)
    if result.stderr and not any(warning in result.stderr for warning in ['INFO:', 'W0000', 'inference_feedback_manager', 'landmark_projection_calculator']):
        print(result.stderr, file=sys.stderr)

if __name__ == "__main__":
    run_clean_pose_estimation()