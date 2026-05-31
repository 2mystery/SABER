import time


class LeavingSeatDetector2:
    def __init__(
        self,
        partial_nose_y_threshold=0.05,
        partial_shoulder_center_y_threshold=0.30,
        partial_seconds=2.0,
        no_person_seconds=2.0,
        complete_leaving_seconds=5.0
    ):
        self.partial_nose_y_threshold = partial_nose_y_threshold
        self.partial_shoulder_center_y_threshold = partial_shoulder_center_y_threshold

        self.partial_seconds = partial_seconds
        self.no_person_seconds = no_person_seconds
        self.complete_leaving_seconds = complete_leaving_seconds

        self.partial_start_time = None
        self.no_pose_start_time = None

    def update(
        self,
        pose_detected,
        nose_y=None,
        left_shoulder_y=None,
        right_shoulder_y=None
    ):
        # 1. 사람이 완전히 사라진 경우
        # pose 자체가 잡히지 않으면 complete leaving 후보로 판단
        if not pose_detected:
            self.partial_start_time = None
            duration = self._update_timer("no_pose_start_time")

            if duration >= self.complete_leaving_seconds:
                return {
                    "detected": True,
                    "message": "Student left seat detected",
                    "state": "complete_leaving"
                }

            if duration >= self.no_person_seconds:
                return {
                    "detected": True,
                    "message": "No person detected",
                    "state": "no_person"
                }

            return {
                "detected": False,
                "message": "Temporary pose loss",
                "state": "temporary_pose_loss"
            }

        # pose가 다시 잡히면 complete leaving timer reset
        self.no_pose_start_time = None

        # 2. pose는 잡혔지만 필요한 landmark 값이 없는 경우
        if nose_y is None or left_shoulder_y is None or right_shoulder_y is None:
            self.partial_start_time = None
            return {
                "detected": False,
                "message": "Required landmark not detected",
                "state": "unknown"
            }

        shoulder_center_y = (left_shoulder_y + right_shoulder_y) / 2

        # 3. 화면에서 일부 벗어난 경우
        # Case 6 기준:
        # nose.y = 0.01
        # shoulder_center_y = (0.20 + 0.24) / 2 = 0.22
        partial_condition = (
            nose_y < self.partial_nose_y_threshold
            or shoulder_center_y < self.partial_shoulder_center_y_threshold
        )

        if partial_condition:
            duration = self._update_timer("partial_start_time")

            if duration >= self.partial_seconds:
                return {
                    "detected": True,
                    "message": "Partial leaving detected",
                    "state": "partial_leaving"
                }

            return {
                "detected": False,
                "message": "Partial leaving candidate",
                "state": "partial_leaving_candidate"
            }

        # 4. 정상 상태
        self.partial_start_time = None

        return {
            "detected": False,
            "message": "Normal posture",
            "state": "normal"
        }

    def _update_timer(self, timer_name):
        now = time.time()

        if getattr(self, timer_name) is None:
            setattr(self, timer_name, now)

        return now - getattr(self, timer_name)