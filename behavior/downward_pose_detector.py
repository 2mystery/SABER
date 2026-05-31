import time


class DownwardPoseDetector:
    def __init__(
        self,
        calibration_seconds=3.0,
        weak_downward_delta=0.08,
        downward_delta=0.12,
        prolonged_downward_delta=0.18,
        shoulder_shift_delta=0.25,
        downward_seconds=1.5,
        prolonged_seconds=3.0,
        body_shift_seconds=2.0
    ):
        # Calibration settings
        self.calibration_seconds = calibration_seconds

        self.baseline_nose_y = None
        self.baseline_shoulder_center_y = None

        self.calibration_start_time = None
        self.calibration_nose_values = []
        self.calibration_shoulder_values = []
        self.is_calibrated = False

        # Threshold settings
        self.weak_downward_delta = weak_downward_delta
        self.downward_delta = downward_delta
        self.prolonged_downward_delta = prolonged_downward_delta
        self.shoulder_shift_delta = shoulder_shift_delta

        # Duration settings
        self.downward_seconds = downward_seconds
        self.prolonged_seconds = prolonged_seconds
        self.body_shift_seconds = body_shift_seconds

        # Timer states
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

        # 1. Dynamic baseline calibration
        if not self.is_calibrated:
            return self._calibrate(nose_y, shoulder_center_y)

        nose_delta = nose_y - self.baseline_nose_y
        shoulder_delta = shoulder_center_y - self.baseline_shoulder_center_y

        # 2. Prolonged downward posture
        # 코가 많이 내려간 상태가 일정 시간 이상 유지되면 prolonged_downward
        if nose_delta >= self.prolonged_downward_delta:
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
                "message": "Observing prolonged downward posture",
                "state": "observing"
            }

        # 3. Downward posture
        # 코가 기준보다 내려간 상태가 일정 시간 이상 유지되면 downward
        if nose_delta >= self.downward_delta:
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
                "message": "Observing downward posture",
                "state": "observing"
            }

        # 4. Body position shift
        # 어깨 중심이 크게 내려간 상태가 일정 시간 이상 유지되면 body_shift
        if shoulder_delta >= self.shoulder_shift_delta:
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
                "message": "Observing body position shift",
                "state": "observing"
            }

        # 5. Weak downward movement
        # 살짝 고개가 내려간 정도라면 alert로 보지 않음
        if nose_delta >= self.weak_downward_delta:
            self._reset_timers()
            return {
                "detected": False,
                "message": "Weak downward movement",
                "state": "weak_downward"
            }

        # 6. Normal posture
        self._reset_timers()
        return {
            "detected": False,
            "message": "Normal posture",
            "state": "normal"
        }

    def _calibrate(self, nose_y, shoulder_center_y):
        now = time.time()

        if self.calibration_start_time is None:
            self.calibration_start_time = now

        self.calibration_nose_values.append(nose_y)
        self.calibration_shoulder_values.append(shoulder_center_y)

        elapsed = now - self.calibration_start_time

        if elapsed < self.calibration_seconds:
            return {
                "detected": False,
                "message": "Calibrating normal posture",
                "state": "calibrating"
            }

        self.baseline_nose_y = (
            sum(self.calibration_nose_values)
            / len(self.calibration_nose_values)
        )

        self.baseline_shoulder_center_y = (
            sum(self.calibration_shoulder_values)
            / len(self.calibration_shoulder_values)
        )

        self.is_calibrated = True

        return {
            "detected": False,
            "message": (
                f"Calibration complete: "
                f"nose={self.baseline_nose_y:.3f}, "
                f"shoulder={self.baseline_shoulder_center_y:.3f}"
            ),
            "state": "calibrated"
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