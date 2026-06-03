import time


class DownwardPoseDetector:
    def __init__(
        self,
        calibration_seconds=3.0,
        weak_downward_delta=0.04,
        downward_delta=0.08,
        prolonged_seconds=2.5,
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

        # Duration settings
        self.prolonged_seconds = prolonged_seconds

        # Timer state
        self.downward_start_time = None

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

        # 2. Downward / Prolonged downward
        # downward_delta 이상 숙인 상태가 오래 유지되면 prolonged_downward
        if nose_delta >= self.downward_delta:
            duration = self._update_timer("downward_start_time")

            if duration >= self.prolonged_seconds:
                return {
                    "detected": True,
                    "message": "Prolonged downward posture detected",
                    "state": "prolonged_downward",
                    "nose_delta": nose_delta,
                    "duration": duration,
                }

            return {
                "detected": False,
                "message": "Downward posture",
                "state": "downward",
                "nose_delta": nose_delta,
                "duration": duration,
            }

        # downward 기준 아래로 올라오면 timer reset
        self._reset_timers()

        # 3. Weak downward
        if nose_delta >= self.weak_downward_delta:
            return {
                "detected": False,
                "message": "Weak downward posture",
                "state": "weak_downward",
                "nose_delta": nose_delta,
                "duration": 0,
            }

        # 4. Normal
        return {
            "detected": False,
            "message": "Normal posture",
            "state": "normal",
            "nose_delta": nose_delta,
            "duration": 0,
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