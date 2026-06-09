from collections import deque


class HeadTurnDetector:
    def __init__(
        self,
        offset_threshold=0.04,
        critical_turn_count=2,
        history_size=30
    ):
        """
        Head turning detector.

        기준:
        - center -> left/right: 1 turn event
        - left -> right or right -> left: 1 additional turn event
        - same direction 유지: count 증가 X

        Severity:
        - 0 turn  -> low
        - 1 turn  -> high
        - 2+ turns -> critical
        """
        self.offset_threshold = offset_threshold
        self.critical_turn_count = critical_turn_count

        self.state_history = deque(maxlen=history_size)
        self.turn_event_history = deque(maxlen=history_size)

        self.previous_state = "center"

    def update(self, nose_x, left_shoulder_x=None, right_shoulder_x=None):
        if nose_x is None or left_shoulder_x is None or right_shoulder_x is None:
            return {
                "detected": False,
                "severity": "unknown",
                "message": "Required landmarks not detected",
                "state": "unknown",
                "head_offset": None,
                "turn_count": 0,
            }

        shoulder_center_x = (left_shoulder_x + right_shoulder_x) / 2
        head_offset = nose_x - shoulder_center_x

        if head_offset > self.offset_threshold:
            state = "left"
        elif head_offset < -self.offset_threshold:
            state = "right"
        else:
            state = "center"

        self.state_history.append(state)

        is_turn_event = self._is_new_turn_event(state)
        self.turn_event_history.append(1 if is_turn_event else 0)

        self.previous_state = state

        recent_turn_count = sum(self.turn_event_history)

        if recent_turn_count >= self.critical_turn_count:
            return {
                "detected": True,
                "severity": "critical",
                "message": "Repeated head turning detected",
                "state": state,
                "head_offset": head_offset,
                "turn_count": recent_turn_count,
            }

        if recent_turn_count >= 1:
            return {
                "detected": True,
                "severity": "high",
                "message": "Head turning detected",
                "state": state,
                "head_offset": head_offset,
                "turn_count": recent_turn_count,
            }

        return {
            "detected": False,
            "severity": "low",
            "message": "Normal head position",
            "state": state,
            "head_offset": head_offset,
            "turn_count": recent_turn_count,
        }

    def _is_new_turn_event(self, state):
        """
        같은 방향을 유지하는 frame은 turn event로 세지 않는다.

        예:
        center -> left  : event
        left -> left    : no event
        left -> center  : no event
        center -> right : event
        left -> right   : event
        right -> left   : event
        """

        if state not in ["left", "right"]:
            return False

        if self.previous_state == "center":
            return True

        if self.previous_state in ["left", "right"] and self.previous_state != state:
            return True

        return False