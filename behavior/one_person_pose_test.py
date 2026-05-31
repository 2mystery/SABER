from behavior.head_turn_detector import HeadTurnDetector
from behavior.leaving_seat_detector import LeavingSeatDetector
from behavior.downward_pose_detector import DownwardPoseDetector

head_turn_detector = HeadTurnDetector()
leaving_seat_detector = LeavingSeatDetector()
downward_pose_detector = DownwardPoseDetector()
...

if latest_results.pose_landmarks:
    pose_detected = True

    nose_x = nose["x"]
    nose_y = nose["y"]
    left_shoulder_y = left_shoulder["y"]
    right_shoulder_y = right_shoulder["y"]

else:
    pose_detected = False
    nose_x = None
    nose_y = None
    left_shoulder_y = None
    right_shoulder_y = None


head_result = head_turn_detector.update(nose_x)

seat_result = leaving_seat_detector.update(
    pose_detected=pose_detected,
    nose_y=nose_y,
    left_shoulder_y=left_shoulder_y,
    right_shoulder_y=right_shoulder_y
)

downward_result = downward_pose_detector.update(
    nose_y=nose_y,
    left_shoulder_y=left_shoulder_y,
    right_shoulder_y=right_shoulder_y
)

# 출력
print("\n================ Detection Result ================")

print("[HEAD TURN]")
print(f"detected: {head_result['detected']}")
print(f"state   : {head_result['state']}")
print(f"message : {head_result['message']}")

print("\n[LEAVING SEAT]")
print(f"detected: {seat_result['detected']}")
print(f"state   : {seat_result['state']}")
print(f"message : {seat_result['message']}")

print("\n[DOWNWARD POSE]")
print(f"detected: {downward_result['detected']}")
print(f"state   : {downward_result['state']}")
print(f"message : {downward_result['message']}")