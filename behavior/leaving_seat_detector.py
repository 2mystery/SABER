class LeavingSeatDetector:
    def __init__(
        self,
        no_pose_frame_threshold=10,
        upper_body_y_threshold=0.15
    ):
        self.no_pose_frame_threshold = no_pose_frame_threshold
        self.upper_body_y_threshold = upper_body_y_threshold
        self.no_pose_count = 0

    def update(self, pose_detected, nose_y=None, left_shoulder_y=None, right_shoulder_y=None):
        if not pose_detected:
            self.no_pose_count += 1
        else:
            self.no_pose_count = 0

        if self.no_pose_count >= self.no_pose_frame_threshold:
            return {
                "detected": True,
                "message": "Student absent from monitored area",
                "state": "absent"
            }

        if pose_detected and nose_y is not None and left_shoulder_y is not None and right_shoulder_y is not None:
            shoulder_center_y = (left_shoulder_y + right_shoulder_y) / 2

            if nose_y < self.upper_body_y_threshold or shoulder_center_y < self.upper_body_y_threshold:
                return {
                    "detected": True,
                    "message": "Student partially left the monitored area",
                    "state": "partial_leave"
                }

        return {
            "detected": False,
            "message": "Student present",
            "state": "present"
        }