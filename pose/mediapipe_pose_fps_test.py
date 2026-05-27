# Camera capture + OpenCV + MediaPipe Pose + landmark drawing 전체 FPS 측정
# git checkout -b feat/pose-fps

'''
Camera Capture
→ OpenCV
→ RGB 변환
→ MediaPipe inference
→ landmark drawing
→ 화면 출력

Frame Skip = 1 / complexity = 0, 1 / resolution = 640x480
'''

# pip install mediapipe opencv-python

import time
import cv2
import mediapipe as mp
from picamera2 import Picamera2

WIDTH = 640
HEIGHT = 480

MODEL_COMPLEXITY = 1

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=MODEL_COMPLEXITY,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (WIDTH, HEIGHT)}
)

picam2.configure(config)
picam2.start()

frame_count = 0
start_time = time.time()

print("Starting MediaPipe Pose FPS test...")

try:
    while True:
        frame = picam2.capture_array()

        results = pose.process(frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        frame_count += 1
        elapsed = time.time() - start_time

        if elapsed >= 1.0:
            fps = frame_count / elapsed
            print(f"MediaPipe Pose FPS: {fps:.2f}")

            frame_count = 0
            start_time = time.time()

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    picam2.stop()
    pose.close()
