# Wide Cam이 AI 없이 몇 FPS 나오는지 측정
# git checkout -b feat/camera-fps

# pip install opencv-python
# sudo apt install -y python3-picamera2
# 먼저 640x480 측정하고, 그다음 1280x720으로 바꿔서 측정

import time
import cv2
from picamera2 import Picamera2

WIDTH = 640
HEIGHT = 480

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (WIDTH, HEIGHT)}
)
picam2.configure(config)
picam2.start()

frame_count = 0
start_time = time.time()

while True:
    frame = picam2.capture_array()

    frame_count += 1
    elapsed = time.time() - start_time

    if elapsed >= 1.0:
        fps = frame_count / elapsed
        print(f"Camera FPS: {fps:.2f}")
        frame_count = 0
        start_time = time.time()

    cv2.imshow("Camera FPS Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()