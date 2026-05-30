from behavior.head_turn_detector import HeadTurnDetector
from behavior.leaving_seat_detector import LeavingSeatDetector

head_turn_detector = HeadTurnDetector()
leaving_seat_detector = LeavingSeatDetector()
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

# 출력
print(head_result)
print(seat_result)