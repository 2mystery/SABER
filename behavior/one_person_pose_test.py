import time
import cv2
import mediapipe as mp
from picamera2 import Picamera2

from behavior.head_turn_detector import HeadTurnDetector
from behavior.leaving_seat_detector import LeavingSeatDetector
from behavior.leaving_seat_detector2 import LeavingSeatDetector2
from behavior.downward_pose_detector import DownwardPoseDetector


# =========================
# Settings
# =========================

WIDTH = 640
HEIGHT = 480
MODEL_COMPLEXITY = 0
FRAME_SKIP = 2

# "v1": 기존 LeavingSeatDetector
# "v2": LeavingSeatDetector2
LEAVING_SEAT_VERSION = "v1"


# =========================
# Detector initialization
# =========================

head_turn_detector = HeadTurnDetector()
downward_pose_detector = DownwardPoseDetector()

if LEAVING_SEAT_VERSION == "v1":
    leaving_seat_detector = LeavingSeatDetector()
elif LEAVING_SEAT_VERSION == "v2":
    leaving_seat_detector = LeavingSeatDetector2()
else:
    raise ValueError("Invalid LEAVING_SEAT_VERSION. Use 'v1' or 'v2'.")


# =========================
# MediaPipe initialization
# =========================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=MODEL_COMPLEXITY,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


# =========================
# Camera initialization
# =========================

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={
        "format": "RGB888",
        "size": (WIDTH, HEIGHT),
    }
)
picam2.configure(config)
picam2.start()


# =========================
# Helper functions
# =========================

def draw_status_panel(frame, head_result, seat_result, downward_result):
    x = 20
    y = 30
    line_gap = 30

    status_lines = [
        f"HEAD     : {head_result['state']}",
        f"SEAT     : {seat_result['state']}",
        f"DOWNWARD : {downward_result['state']}",
    ]

    for line in status_lines:
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )
        y += line_gap

    alert_lines = []

    if head_result["detected"]:
        alert_lines.append(f"[HEAD] {head_result['message']}")

    if seat_result["detected"]:
        alert_lines.append(f"[SEAT] {seat_result['message']}")

    if downward_result["detected"]:
        alert_lines.append(f"[DOWNWARD] {downward_result['message']}")

    for alert in alert_lines:
        cv2.putText(
            frame,
            alert,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        y += line_gap


def print_result(title, result):
    print(f"\n[{title}]")
    print(f"detected: {result['detected']}")
    print(f"state   : {result['state']}")
    print(f"message : {result['message']}")


def print_all_results(head_result, seat_result, downward_result):
    print("\n================ Detection Result ================")
    print_result("HEAD TURN", head_result)
    print_result("LEAVING SEAT", seat_result)
    print_result("DOWNWARD POSE", downward_result)


# =========================
# Main loop
# =========================

frame_count = 0
processed_count = 0
start_time = time.time()
last_print_time = time.time()

latest_results = None

print("One-person pose test started.")
print(f"LeavingSeatDetector version: {LEAVING_SEAT_VERSION}")
print("Press 'q' on the camera window or Ctrl+C in terminal to stop.")

try:
    while True:
        frame_rgb = picam2.capture_array()
        frame_count += 1

        # OpenCV display uses BGR
        display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Frame skip 적용
        if frame_count % FRAME_SKIP == 0:
            latest_results = pose.process(frame_rgb)
            processed_count += 1

        pose_detected = False

        nose_x = None
        nose_y = None
        left_shoulder_y = None
        right_shoulder_y = None

        head_result = {
            "detected": False,
            "message": "No nose landmark detected",
            "state": "unknown",
        }

        seat_result = leaving_seat_detector.update(
            pose_detected=False,
            nose_y=None,
            left_shoulder_y=None,
            right_shoulder_y=None,
        )

        downward_result = downward_pose_detector.update(
            nose_y=None,
            left_shoulder_y=None,
            right_shoulder_y=None,
        )

        if latest_results and latest_results.pose_landmarks:
            pose_detected = True
            landmarks = latest_results.pose_landmarks.landmark

            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            nose_x = nose.x
            nose_y = nose.y
            left_shoulder_y = left_shoulder.y
            right_shoulder_y = right_shoulder.y

            head_result = head_turn_detector.update(nose_x)

            seat_result = leaving_seat_detector.update(
                pose_detected=pose_detected,
                nose_y=nose_y,
                left_shoulder_y=left_shoulder_y,
                right_shoulder_y=right_shoulder_y,
            )

            downward_result = downward_pose_detector.update(
                nose_y=nose_y,
                left_shoulder_y=left_shoulder_y,
                right_shoulder_y=right_shoulder_y,
            )

            mp_drawing.draw_landmarks(
                display_frame,
                latest_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
            )

            landmark_text = (
                f"nose.y={nose_y:.3f} "
                f"L_sh.y={left_shoulder_y:.3f} "
                f"R_sh.y={right_shoulder_y:.3f}"
            )

            cv2.putText(
                display_frame,
                landmark_text,
                (20, HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

        # VNC/OpenCV 화면 좌측 상단에 상태 표시
        draw_status_panel(
            display_frame,
            head_result,
            seat_result,
            downward_result,
        )

        # FPS 표시
        elapsed = time.time() - start_time
        camera_fps = frame_count / elapsed if elapsed > 0 else 0
        pose_fps = processed_count / elapsed if elapsed > 0 else 0

        cv2.putText(
            display_frame,
            f"Camera FPS: {camera_fps:.2f} | Pose FPS: {pose_fps:.2f}",
            (20, HEIGHT - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        # 터미널 출력은 1초에 한 번만
        now = time.time()
        if now - last_print_time >= 1.0:
            print_all_results(head_result, seat_result, downward_result)
            last_print_time = now

        cv2.imshow("One Person Pose Test", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Stopped by user.")
            break

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    pose.close()
    picam2.stop()
    cv2.destroyAllWindows()

    total_time = time.time() - start_time
    print("\n====== Final Result ======")
    print(f"Resolution: {WIDTH}x{HEIGHT}")
    print(f"Model Complexity: {MODEL_COMPLEXITY}")
    print(f"Frame Skip: {FRAME_SKIP}")
    print(f"Total Frames: {frame_count}")
    print(f"Processed Pose Frames: {processed_count}")
    print(f"Camera FPS: {frame_count / total_time:.2f}")
    print(f"Pose FPS: {processed_count / total_time:.2f}")