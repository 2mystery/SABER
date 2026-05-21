# Camera capture + OpenCV + MediaPipe Pose + landmark drawing 전체 FPS 측정
# Frame Skip = 2 버전

'''
Pipeline
Camera Capture
→ OpenCV
→ MediaPipe inference
→ landmark drawing
→ 화면 출력

Frame Skip = 2
→ 2프레임마다 MediaPipe inference 수행
'''

import time
import cv2
import mediapipe as mp
from picamera2 import Picamera2

WIDTH = 1280
HEIGHT = 720

MODEL_COMPLEXITY = 0
FRAME_SKIP = 2

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

inference_count = 0

results = None

print("Starting MediaPipe Pose FPS test...")
print(f"Resolution: {WIDTH}x{HEIGHT}")
print(f"Model Complexity: {MODEL_COMPLEXITY}")
print(f"Frame Skip: {FRAME_SKIP}")

try:
    while True:
        frame = picam2.capture_array()

        # Frame skipping
        if inference_count % FRAME_SKIP == 0:
            results = pose.process(frame)

        inference_count += 1

        # Draw landmarks
        if results and results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # 화면 출력
        cv2.imshow("MediaPipe Pose FPS Test", frame)

        # FPS 계산
        frame_count += 1
        elapsed = time.time() - start_time

        if elapsed >= 1.0:
            fps = frame_count / elapsed

            print(f"MediaPipe Pose FPS: {fps:.2f}")

            frame_count = 0
            start_time = time.time()

        # ESC 키 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    picam2.stop()
    pose.close()
    cv2.destroyAllWindows()
