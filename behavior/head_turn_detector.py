from collections import deque


class HeadTurnDetector:
    def __init__(
        self,
        left_threshold=0.50,
        right_threshold=0.40,
        min_turn_changes=3,
        history_size=30
    ):
        self.left_threshold = left_threshold
        self.right_threshold = right_threshold
        self.min_turn_changes = min_turn_changes
        self.history = deque(maxlen=history_size)

    def update(self, nose_x):
        if nose_x is None:
            return {
                "detected": False,
                "message": "No nose landmark detected",
                "state": "unknown"
            }

        if nose_x > self.left_threshold:
            state = "left"
        elif nose_x < self.right_threshold:
            state = "right"
        else:
            state = "center"

        self.history.append(state)

        turn_changes = self._count_left_right_changes()

        if turn_changes >= self.min_turn_changes:
            return {
                "detected": True,
                "message": "Repeated head turning detected",
                "state": state
            }

        return {
            "detected": False,
            "message": "Normal head movement",
            "state": state
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