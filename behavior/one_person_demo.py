import time
import cv2
import mediapipe as mp
from picamera2 import Picamera2

from behavior.head_turn_detector import HeadTurnDetector
from behavior.leaving_seat_detector import LeavingSeatDetector
from behavior.leaving_seat_detector2 import LeavingSeatDetector2
from behavior.downward_pose_detector import DownwardPoseDetector

from ui.saber_dashboard import render_one_person_dashboard


# =========================
# Camera / Pose settings
# =========================
WIDTH = 640
HEIGHT = 480
MODEL_COMPLEXITY = 0
FRAME_SKIP = 2

DEBUG_MODE = False


# =========================
# Detector version setting
# =========================
# "v1": 기존 LeavingSeatDetector
# "v2": LeavingSeatDetector2
LEAVING_SEAT_VERSION = "v1"


# =========================
# Threshold override settings
# =========================
# 여기 값만 바꾸면서 실험하면 됨
# detector 파일 내부를 직접 수정하지 않아도 됨
HEAD_TURN_CONFIG = {
    "offset_threshold": 0.06,
    "min_turn_changes": 3,
    "history_size": 30,
}

LEAVING_SEAT_V1_CONFIG = {
    "no_pose_frame_threshold": 6,
    "upper_body_y_threshold": 0.15,
}

LEAVING_SEAT_V2_CONFIG = {
    "partial_nose_y_threshold": 0.05,
    "partial_shoulder_center_y_threshold": 0.30,
    "partial_seconds": 2.0,
    "no_person_seconds": 2.0,
    "complete_leaving_seconds": 5.0,
}

DOWNWARD_POSE_CONFIG = {
    "calibration_seconds": 3.0,
    "weak_downward_delta": 0.04,
    "downward_delta": 0.08,
    "prolonged_seconds": 2.5,
}


# =========================
# Detector initialization
# =========================
head_turn_detector = HeadTurnDetector(**HEAD_TURN_CONFIG)
downward_pose_detector = DownwardPoseDetector(**DOWNWARD_POSE_CONFIG)

if LEAVING_SEAT_VERSION == "v1":
    leaving_seat_detector = LeavingSeatDetector(**LEAVING_SEAT_V1_CONFIG)
elif LEAVING_SEAT_VERSION == "v2":
    leaving_seat_detector = LeavingSeatDetector2(**LEAVING_SEAT_V2_CONFIG)
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

# 초록빛 문제 해결용:
# BGR888 / RGB888 대신 XRGB8888 사용
config = picam2.create_preview_configuration(
    main={
        "format": "XRGB8888",
        "size": (WIDTH, HEIGHT),
    }
)

picam2.configure(config)
picam2.start()

# Auto exposure / white balance 안정화 시간
time.sleep(2)


# =========================
# Helper functions
# =========================

def get_unknown_head_result():
    return {
        "detected": False,
        "message": "No nose landmark detected",
        "state": "unknown",
    }


def get_unknown_downward_result():
    return {
        "detected": False,
        "message": "Required landmark not detected",
        "state": "unknown",
    }


def get_waiting_seat_result():
    return {
        "detected": False,
        "message": "Waiting for pose update",
        "state": "waiting",
    }


def print_result(title, result):
    print(f"\n[{title}]")
    print(f"detected: {result.get('detected', False)}")
    print(f"state   : {result.get('state', 'unknown')}")
    print(f"message : {result.get('message', '')}")

    extra_keys = [
        "head_offset",
        "no_pose_count",
        "pose_detected",
        "nose_y",
        "shoulder_center_y",
        "nose_delta",
        "duration",
    ]

    for key in extra_keys:
        if key in result:
            print(f"{key:<15}: {result[key]}")


def print_all_results(head_result, seat_result, downward_result, pose_fps):
    print("\n================ Detection Result ================")
    print(f"Pose FPS: {pose_fps:.2f}")
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

# skipped frame에서도 마지막 detector 결과를 계속 표시하기 위한 변수
latest_head_result = get_unknown_head_result()
latest_seat_result = get_waiting_seat_result()
latest_downward_result = get_unknown_downward_result()

# debug mode에서만 표시할 landmark 값
latest_landmark_values = None

print("One-person SABER dashboard started.")
print(f"Resolution: {WIDTH}x{HEIGHT}")
print("Camera Format: XRGB8888")
print(f"Model Complexity: {MODEL_COMPLEXITY}")
print(f"Frame Skip: {FRAME_SKIP}")
print(f"LeavingSeatDetector version: {LEAVING_SEAT_VERSION}")
print(f"Debug Mode: {DEBUG_MODE}")
print("Press 'q' on the dashboard window or Ctrl+C in terminal to stop.")

try:
    while True:
        # =========================
        # Frame capture
        # =========================
        frame = picam2.capture_array()
        frame_count += 1

        # XRGB8888은 보통 4채널로 들어옴.
        # OpenCV display는 BGR 3채널 기준이므로 앞 3채널만 사용.
        display_frame = frame[:, :, :3].copy()

        # MediaPipe Pose는 RGB 입력을 사용하므로 inference용만 RGB 변환.
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

        processed_this_frame = False

        # =========================
        # Pose inference with frame skip
        # =========================
        if frame_count % FRAME_SKIP == 0:
            latest_results = pose.process(frame_rgb)
            processed_count += 1
            processed_this_frame = True

        # =========================
        # Detector update
        # =========================
        # 중요:
        # detector update는 pose.process가 실제로 수행된 프레임에서만 한다.
        # skipped frame에서는 latest_*_result를 그대로 화면에 표시한다.
        if processed_this_frame:
            if latest_results and latest_results.pose_landmarks:
                landmarks = latest_results.pose_landmarks.landmark

                nose = landmarks[mp_pose.PoseLandmark.NOSE]
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

                nose_x = nose.x
                nose_y = nose.y

                left_shoulder_x = left_shoulder.x
                right_shoulder_x = right_shoulder.x

                left_shoulder_y = left_shoulder.y
                right_shoulder_y = right_shoulder.y

                latest_head_result = head_turn_detector.update(
                    nose_x=nose_x,
                    left_shoulder_x=left_shoulder_x,
                    right_shoulder_x=right_shoulder_x,
                )

                latest_seat_result = leaving_seat_detector.update(
                    pose_detected=True,
                    nose_y=nose_y,
                    left_shoulder_y=left_shoulder_y,
                    right_shoulder_y=right_shoulder_y,
                )

                latest_downward_result = downward_pose_detector.update(
                    nose_y=nose_y,
                    left_shoulder_y=left_shoulder_y,
                    right_shoulder_y=right_shoulder_y,
                )

                latest_landmark_values = {
                    "nose_x": nose_x,
                    "nose_y": nose_y,
                    "left_shoulder_y": left_shoulder_y,
                    "right_shoulder_y": right_shoulder_y,
                }

            else:
                latest_head_result = get_unknown_head_result()

                latest_seat_result = leaving_seat_detector.update(
                    pose_detected=False,
                    nose_y=None,
                    left_shoulder_y=None,
                    right_shoulder_y=None,
                )

                latest_downward_result = get_unknown_downward_result()
                latest_landmark_values = None

        # =========================
        # Landmark drawing
        # =========================
        # 화면에는 최신 pose landmark를 표시한다.
        # 단, latest_results가 no-pose이면 그리지 않는다.
        if latest_results and latest_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                display_frame,
                latest_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
            )

        # =========================
        # FPS calculation
        # =========================
        elapsed = time.time() - start_time
        pose_fps = processed_count / elapsed if elapsed > 0 else 0.0

        # =========================
        # Dashboard rendering
        # =========================
        dashboard_frame = render_one_person_dashboard(
            camera_frame=display_frame,
            head_result=latest_head_result,
            seat_result=latest_seat_result,
            downward_result=latest_downward_result,
            pose_fps=pose_fps,
            processed_this_frame=processed_this_frame,
            debug_mode=DEBUG_MODE,
            landmark_values=latest_landmark_values,
        )

        # =========================
        # Terminal logging
        # =========================
        now = time.time()
        if now - last_print_time >= 1.0:
            print_all_results(
                latest_head_result,
                latest_seat_result,
                latest_downward_result,
                pose_fps=pose_fps,
            )
            last_print_time = now

        # =========================
        # Display
        # =========================
        cv2.imshow("SABER Dashboard", dashboard_frame)

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
    print("Camera Format: XRGB8888")
    print(f"Model Complexity: {MODEL_COMPLEXITY}")
    print(f"Frame Skip: {FRAME_SKIP}")
    print(f"Total Frames: {frame_count}")
    print(f"Processed Pose Frames: {processed_count}")

    if total_time > 0:
        print(f"Pose FPS: {processed_count / total_time:.2f}")