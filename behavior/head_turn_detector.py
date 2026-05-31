from collections import deque


class HeadTurnDetector:
    def __init__(
        self,
        offset_threshold=0.06,
        min_turn_changes=3,
        history_size=30
    ):
        self.offset_threshold = offset_threshold
        self.min_turn_changes = min_turn_changes
        self.history = deque(maxlen=history_size)

    def update(self, nose_x, left_shoulder_x=None, right_shoulder_x=None):
        if nose_x is None or left_shoulder_x is None or right_shoulder_x is None:
            return {
                "detected": False,
                "message": "Required landmarks not detected",
                "state": "unknown"
            }

        shoulder_center_x = (left_shoulder_x + right_shoulder_x) / 2
        head_offset = nose_x - shoulder_center_x

        if head_offset > self.offset_threshold:
            state = "left"
        elif head_offset < -self.offset_threshold:
            state = "right"
        else:
            state = "center"

        self.history.append(state)

        turn_changes = self._count_left_right_changes()

        if turn_changes >= self.min_turn_changes:
            return {
                "detected": True,
                "message": "Repeated head turning detected",
                "state": state,
                "head_offset": head_offset
            }

        return {
            "detected": False,
            "message": "Normal head movement",
            "state": state,
            "head_offset": head_offset
        }

    def _count_left_right_changes(self):
        filtered = [s for s in self.history if s in ["left", "right"]]

        if len(filtered) < 2:
            return 0

        changes = 0
        for i in range(1, len(filtered)):
            if filtered[i] != filtered[i - 1]:
                changes += 1

        return changes