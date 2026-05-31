import time


class DownwardPoseDetector:
    def __init__(
        self,
        baseline_nose_y=0.33,
        baseline_shoulder_center_y=0.535,
        weak_downward_delta=0.12,
        downward_delta=0.17,
        prolonged_downward_delta=0.22,
        shoulder_shift_delta=0.15,
        downward_seconds=3.0,
        prolonged_seconds=5.0,
        body_shift_seconds=2.0
    ):
        self.baseline_nose_y = baseline_nose_y
        self.baseline_shoulder_center_y = baseline_shoulder_center_y

        self.weak_downward_delta = weak_downward_delta
        self.downward_delta = downward_delta
        self.prolonged_downward_delta = prolonged_downward_delta
        self.shoulder_shift_delta = shoulder_shift_delta

        self.downward_seconds = downward_seconds
        self.prolonged_seconds = prolonged_seconds
        self.body_shift_seconds = body_shift_seconds

        self.downward_start_time = None
        self.prolonged_start_time = None
        self.body_shift_start_time = None

    def update(self, nose_y=None, left_shoulder_y=None, right_shoulder_y=None):
        if nose_y is None or left_shoulder_y is None or right_shoulder_y is None:
            self._reset_timers()
            return {
                "detected": False,
                "message": "Required landmark not detected",
                "state": "unknown"
            }

        shoulder_center_y = (left_shoulder_y + right_shoulder_y) / 2

        nose_delta = nose_y - self.baseline_nose_y
        shoulder_delta = shoulder_center_y - self.baseline_shoulder_center_y

        # 1. Body position shift
        if nose_delta >= self.downward_delta and shoulder_delta >= self.shoulder_shift_delta:
            duration = self._update_timer("body_shift_start_time")
            self.downward_start_time = None
            self.prolonged_start_time = None

            if duration >= self.body_shift_seconds:
                return {
                    "detected": True,
                    "message": "Body position shift detected",
                    "state": "body_shift"
                }

            return {
                "detected": False,
                "message": "Body position shift candidate",
                "state": "body_shift_candidate"
            }

        # 2. Prolonged downward posture
        if nose_delta >= self.prolonged_downward_delta and shoulder_delta < self.shoulder_shift_delta:
            duration = self._update_timer("prolonged_start_time")
            self.downward_start_time = None
            self.body_shift_start_time = None

            if duration >= self.prolonged_seconds:
                return {
                    "detected": True,
                    "message": "Prolonged downward posture detected",
                    "state": "prolonged_downward"
                }

            return {
                "detected": False,
                "message": "Prolonged downward posture candidate",
                "state": "prolonged_downward_candidate"
            }

        # 3. Downward posture
        if nose_delta >= self.downward_delta and shoulder_delta < self.shoulder_shift_delta:
            duration = self._update_timer("downward_start_time")
            self.prolonged_start_time = None
            self.body_shift_start_time = None

            if duration >= self.downward_seconds:
                return {
                    "detected": True,
                    "message": "Downward posture detected",
                    "state": "downward"
                }

            return {
                "detected": False,
                "message": "Downward posture candidate",
                "state": "downward_candidate"
            }

        # 4. Weak downward movement
        if nose_delta >= self.weak_downward_delta:
            self._reset_timers()
            return {
                "detected": False,
                "message": "Weak downward movement",
                "state": "weak_downward"
            }

        # 5. Normal
        self._reset_timers()
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

    def _reset_timers(self):
        self.downward_start_time = None
        self.prolonged_start_time = None
        self.body_shift_start_time = None