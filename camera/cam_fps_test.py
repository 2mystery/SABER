# Camera capture + OpenCV + MediaPipe Pose + landmark drawing 전체 FPS 측정
# git checkout -b feat/pose-fps

import time
import cv2
import mediapipe as mp
from picamera2 import Picamera2


WIDTH = 640
HEIGHT = 480

MODEL_COMPLEXITY = 0

FRAME_SKIP = 1

TEST_DURATION = 30


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

# =====================================
# FPS Variables
# =====================================

total_frame_count = 0
fps_frame_count = 0

fps = 0.0

fps_values = []

fps_timer = time.time()
test_start_time = time.time()

last_results = None

print("=====================================")
print("Pose FPS Benchmark Started")
print(f"Resolution      : {WIDTH}x{HEIGHT}")
print(f"Model Complexity: {MODEL_COMPLEXITY}")
print(f"Frame Skip      : {FRAME_SKIP}")
print(f"Test Duration   : {TEST_DURATION}s")
print("=====================================")



while True:

    frame = picam2.capture_array()

    total_frame_count += 1
    fps_frame_count += 1

    # Picamera2 already outputs RGB888
    # 꼭 RGB2BGR 해야 하는지?
    rgb_frame = frame

    # =================================
    # Frame Skipping
    # =================================

    if total_frame_count % FRAME_SKIP == 0:
        last_results = pose.process(rgb_frame)

    # =================================
    # Landmark Overlay
    # =================================

    if last_results and last_results.pose_landmarks:

        mp_drawing.draw_landmarks(
            frame,
            last_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # =================================
    # FPS Calculation
    # =================================

    elapsed = time.time() - fps_timer

    if elapsed >= 1.0:

        fps = fps_frame_count / elapsed

        fps_values.append(fps)

        print(
            f"[FPS] {fps:.2f} | "
            f"Resolution={WIDTH}x{HEIGHT} | "
            f"Complexity={MODEL_COMPLEXITY} | "
            f"Skip={FRAME_SKIP}"
        )

        fps_frame_count = 0
        fps_timer = time.time()

    # =================================
    # Overlay Text
    # =================================

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"{WIDTH}x{HEIGHT} | complexity={MODEL_COMPLEXITY} | skip={FRAME_SKIP}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("MediaPipe Pose FPS Test", frame)

    # =================================
    # Test Duration End
    # =================================

    total_test_elapsed = time.time() - test_start_time

    if total_test_elapsed >= TEST_DURATION:
        print("\n=====================================")
        print("Benchmark Finished")

        avg_fps = sum(fps_values) / len(fps_values)
        min_fps = min(fps_values)

        print(f"Average FPS : {avg_fps:.2f}")
        print(f"Minimum FPS : {min_fps:.2f}")

        print("=====================================")
        break

    # =================================
    # Quit
    # =================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



picam2.stop()
pose.close()
cv2.destroyAllWindows()