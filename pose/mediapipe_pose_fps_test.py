# Camera capture + OpenCV + MediaPipe Pose + landmark drawing 전체 FPS 측정
# git checkout -b feat/pose-fps

'''
Camera Capture
→ OpenCV
→ RGB 변환
→ MediaPipe inference
→ landmark drawing
→ 화면 출력
'''

# pip install mediapipe opencv-python

import time
import cv2
import mediapipe as mp
from picamera2 import Picamera2

WIDTH = 640
HEIGHT = 480

MODEL_COMPLEXITY = 0

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
fps = 0.0

while True:
    frame = picam2.capture_array()

    # MediaPipe는 RGB 입력 사용
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb_frame)

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

    cv2.putText(
        frame,
        f"Pose FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("MediaPipe Pose FPS Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
pose.close()
cv2.destroyAllWindows()