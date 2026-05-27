import time
import cv2
import mediapipe as mp
from picamera2 import Picamera2

# Baseline settings
WIDTH = 640
HEIGHT = 480
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

# Use XRGB8888 for stable OpenCV display on Raspberry Pi
config = picam2.create_preview_configuration(
    main={"format": "XRGB8888", "size": (WIDTH, HEIGHT)}
)

picam2.configure(config)
picam2.start()

# Let auto exposure / auto white balance stabilize
time.sleep(2)

frame_count = 0
processed_count = 0
start_time = time.time()

latest_results = None
latest_pose_info = None


def get_landmark_info(landmarks, landmark_id):
    lm = landmarks[landmark_id]
    return {
        "x": lm.x,
        "y": lm.y,
        "visibility": lm.visibility
    }


def print_pose_info(pose_info):
    print("\n[One Person Pose Info]")
    for name, info in pose_info.items():
        print(
            f"{name}: "
            f"x={info['x']:.3f}, "
            f"y={info['y']:.3f}, "
            f"visibility={info['visibility']:.3f}"
        )


try:
    while True:
        frame = picam2.capture_array()
        frame_count += 1

        # XRGB8888 usually comes as 4-channel image.
        # OpenCV display uses BGR, so keep first 3 channels.
        frame_bgr = frame[:, :, :3].copy()

        if frame_count % FRAME_SKIP == 0:
            # MediaPipe requires RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            latest_results = pose.process(frame_rgb)
            processed_count += 1

            latest_pose_info = None

            if latest_results.pose_landmarks:
                landmarks = latest_results.pose_landmarks.landmark

                nose = get_landmark_info(
                    landmarks,
                    mp_pose.PoseLandmark.NOSE
                )
                left_shoulder = get_landmark_info(
                    landmarks,
                    mp_pose.PoseLandmark.LEFT_SHOULDER
                )
                right_shoulder = get_landmark_info(
                    landmarks,
                    mp_pose.PoseLandmark.RIGHT_SHOULDER
                )

                latest_pose_info = {
                    "nose": nose,
                    "left_shoulder": left_shoulder,
                    "right_shoulder": right_shoulder
                }

                print_pose_info(latest_pose_info)

        if latest_results and latest_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame_bgr,
                latest_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        elapsed_time = time.time() - start_time
        camera_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        pose_fps = processed_count / elapsed_time if elapsed_time > 0 else 0

        cv2.putText(
            frame_bgr,
            f"Camera FPS: {camera_fps:.2f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame_bgr,
            f"Pose FPS: {pose_fps:.2f}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame_bgr,
            f"Resolution: {WIDTH}x{HEIGHT}, Complexity: {MODEL_COMPLEXITY}, Skip: {FRAME_SKIP}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        if latest_pose_info:
            nose = latest_pose_info["nose"]
            cv2.putText(
                frame_bgr,
                f"Nose: x={nose['x']:.2f}, y={nose['y']:.2f}, v={nose['visibility']:.2f}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2
            )
        else:
            cv2.putText(
                frame_bgr,
                "No person detected",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

        cv2.imshow("SABER - One Person Pose Test", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    picam2.stop()
    pose.close()
    cv2.destroyAllWindows()

    total_time = time.time() - start_time
    print("\n===== Final Result =====")
    print(f"Resolution: {WIDTH}x{HEIGHT}")
    print(f"Model Complexity: {MODEL_COMPLEXITY}")
    print(f"Frame Skip: {FRAME_SKIP}")
    print(f"Total Frames: {frame_count}")
    print(f"Processed Pose Frames: {processed_count}")

    if total_time > 0:
        print(f"Camera FPS: {frame_count / total_time:.2f}")
        print(f"Pose FPS: {processed_count / total_time:.2f}")