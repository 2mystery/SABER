import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from picamera2 import Picamera2


from behavior.head_turn_detector import HeadTurnDetector
from behavior.downward_pose_detector import DownwardPoseDetector
from behavior.leaving_seat_detector import LeavingSeatDetector


# =========================
# Config
# =========================

WIDTH = 640
HEIGHT = 480

MAX_PEOPLE = 3
MODEL_PATH = "models/pose_landmarker_lite.task"

VISIBILITY_THRESHOLD = 0.5

# track matching
MAX_MATCH_DISTANCE = 0.18
MAX_MISSED_FRAMES = 20

# detector configs
HEAD_TURN_CONFIG = {
    "offset_threshold": 0.04,
    "min_turn_changes": 3,
    "history_size": 30,
}

LEAVING_SEAT_CONFIG = {
    "no_pose_frame_threshold": 6,
    "upper_body_y_threshold": 0.15,
}

DOWNWARD_POSE_CONFIG = {
    "calibration_seconds": 3.0,
    "weak_downward_delta": 0.04,
    "downward_delta": 0.08,
    "prolonged_seconds": 2.5,
}


# =========================
# Landmark index
# =========================

NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


@dataclass
class PoseData:
    """
    한 명의 pose에서 detector에 필요한 landmark만 추출한 구조.
    x, y는 MediaPipe normalized coordinate.
    즉, frame width/height 기준 0.0 ~ 1.0 범위.
    """
    pose_index: int

    nose_x: Optional[float]
    nose_y: Optional[float]

    left_shoulder_x: Optional[float]
    left_shoulder_y: Optional[float]

    right_shoulder_x: Optional[float]
    right_shoulder_y: Optional[float]

    center_x: float
    center_y: float

    bbox: Tuple[int, int, int, int]


@dataclass
class PersonTrack:
    """
    person_id별 상태 저장 객체.
    detector들은 내부 history/timer/calibration state를 갖기 때문에
    반드시 사람마다 따로 instance를 유지해야 함.
    """
    person_id: int
    center_x: float
    center_y: float

    head_detector: HeadTurnDetector = field(
        default_factory=lambda: HeadTurnDetector(**HEAD_TURN_CONFIG)
    )
    downward_detector: DownwardPoseDetector = field(
        default_factory=lambda: DownwardPoseDetector(**DOWNWARD_POSE_CONFIG)
    )
    leaving_detector: LeavingSeatDetector = field(
        default_factory=lambda: LeavingSeatDetector(**LEAVING_SEAT_CONFIG)
    )

    missed_frames: int = 0
    last_seen_frame: int = 0

    last_head_result: dict = field(default_factory=dict)
    last_downward_result: dict = field(default_factory=dict)
    last_leaving_result: dict = field(default_factory=dict)


class MultiPersonBehaviorMonitor:
    def __init__(self):
        self.tracks: Dict[int, PersonTrack] = {}
        self.next_person_id = 1
        self.frame_index = 0

    def update(self, poses: List[PoseData]) -> Dict[int, dict]:
        """
        현재 frame에서 감지된 여러 pose를 기존 person track과 매칭하고,
        person_id별 detector를 업데이트한다.
        """
        self.frame_index += 1

        matched_track_ids = set()
        matched_pose_indices = set()

        # 1. 기존 track과 현재 pose를 nearest-neighbor 방식으로 매칭
        matches = self._match_tracks_to_poses(poses)

        for track_id, pose_index in matches:
            track = self.tracks[track_id]
            pose = poses[pose_index]

            matched_track_ids.add(track_id)
            matched_pose_indices.add(pose_index)

            track.center_x = pose.center_x
            track.center_y = pose.center_y
            track.missed_frames = 0
            track.last_seen_frame = self.frame_index

            self._update_detectors_for_track(track, pose)

        # 2. 매칭 안 된 pose는 새로운 person track으로 등록
        for i, pose in enumerate(poses):
            if i in matched_pose_indices:
                continue

            new_track = PersonTrack(
                person_id=self.next_person_id,
                center_x=pose.center_x,
                center_y=pose.center_y,
                last_seen_frame=self.frame_index,
            )

            self.tracks[self.next_person_id] = new_track
            self._update_detectors_for_track(new_track, pose)

            matched_track_ids.add(self.next_person_id)
            self.next_person_id += 1

        # 3. 이번 frame에서 안 보인 기존 track은 leaving detector에 pose_detected=False 전달
        for track_id, track in list(self.tracks.items()):
            if track_id in matched_track_ids:
                continue

            track.missed_frames += 1
            track.last_leaving_result = track.leaving_detector.update(
                pose_detected=False
            )

            # head/downward는 landmark가 없으므로 unknown 처리
            track.last_head_result = {
                "detected": False,
                "message": "Pose missing",
                "state": "unknown",
            }
            track.last_downward_result = {
                "detected": False,
                "message": "Pose missing",
                "state": "unknown",
            }

            # 너무 오래 사라진 사람은 track 제거
            if track.missed_frames > MAX_MISSED_FRAMES:
                del self.tracks[track_id]

        # 4. 결과 반환
        return self._build_behavior_states()

    def _match_tracks_to_poses(self, poses: List[PoseData]) -> List[Tuple[int, int]]:
        """
        단순 nearest-neighbor tracking.
        현재 pose center와 기존 track center의 normalized distance가 가장 가까운 것끼리 매칭.
        """
        candidates = []

        for track_id, track in self.tracks.items():
            for pose_index, pose in enumerate(poses):
                dist = self._distance(
                    track.center_x,
                    track.center_y,
                    pose.center_x,
                    pose.center_y,
                )

                if dist <= MAX_MATCH_DISTANCE:
                    candidates.append((dist, track_id, pose_index))

        candidates.sort(key=lambda x: x[0])

        matches = []
        used_tracks = set()
        used_poses = set()

        for dist, track_id, pose_index in candidates:
            if track_id in used_tracks or pose_index in used_poses:
                continue

            matches.append((track_id, pose_index))
            used_tracks.add(track_id)
            used_poses.add(pose_index)

        return matches

    def _update_detectors_for_track(self, track: PersonTrack, pose: PoseData):
        pose_detected = (
            pose.nose_x is not None
            and pose.nose_y is not None
            and pose.left_shoulder_x is not None
            and pose.left_shoulder_y is not None
            and pose.right_shoulder_x is not None
            and pose.right_shoulder_y is not None
        )

        track.last_head_result = track.head_detector.update(
            nose_x=pose.nose_x,
            left_shoulder_x=pose.left_shoulder_x,
            right_shoulder_x=pose.right_shoulder_x,
        )

        track.last_downward_result = track.downward_detector.update(
            nose_y=pose.nose_y,
            left_shoulder_y=pose.left_shoulder_y,
            right_shoulder_y=pose.right_shoulder_y,
        )

        track.last_leaving_result = track.leaving_detector.update(
            pose_detected=pose_detected,
            nose_y=pose.nose_y,
            left_shoulder_y=pose.left_shoulder_y,
            right_shoulder_y=pose.right_shoulder_y,
        )

    def _build_behavior_states(self) -> Dict[int, dict]:
        states = {}

        for track_id, track in self.tracks.items():
            states[track_id] = {
                "person_id": track_id,
                "center": {
                    "x": track.center_x,
                    "y": track.center_y,
                },
                "missed_frames": track.missed_frames,
                "head_turn": track.last_head_result,
                "downward_pose": track.last_downward_result,
                "leaving_seat": track.last_leaving_result,
            }

        return states

    @staticmethod
    def _distance(x1, y1, x2, y2) -> float:
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# =========================
# MediaPipe helpers
# =========================

def get_visible_landmark(landmarks, index: int, visibility_threshold=VISIBILITY_THRESHOLD):
    """
    landmark visibility가 낮으면 None으로 처리.
    """
    if index >= len(landmarks):
        return None

    lm = landmarks[index]

    if hasattr(lm, "visibility") and lm.visibility < visibility_threshold:
        return None

    return lm


def extract_pose_data(
    landmarks,
    pose_index: int,
    frame_width: int,
    frame_height: int,
) -> Optional[PoseData]:
    """
    MediaPipe PoseLandmarker가 반환한 한 명의 landmarks에서
    detector에 필요한 nose/shoulder 좌표와 tracking center/bbox를 추출.
    """
    nose = get_visible_landmark(landmarks, NOSE)
    left_shoulder = get_visible_landmark(landmarks, LEFT_SHOULDER)
    right_shoulder = get_visible_landmark(landmarks, RIGHT_SHOULDER)
    left_hip = get_visible_landmark(landmarks, LEFT_HIP)
    right_hip = get_visible_landmark(landmarks, RIGHT_HIP)

    # tracking center 계산용 후보 landmark
    center_candidates = [
        lm for lm in [nose, left_shoulder, right_shoulder, left_hip, right_hip]
        if lm is not None
    ]

    if len(center_candidates) == 0:
        return None

    center_x = sum(lm.x for lm in center_candidates) / len(center_candidates)
    center_y = sum(lm.y for lm in center_candidates) / len(center_candidates)

    visible_points = [
        lm for lm in landmarks
        if hasattr(lm, "visibility") and lm.visibility >= VISIBILITY_THRESHOLD
    ]

    if len(visible_points) == 0:
        return None

    xs = [lm.x for lm in visible_points]
    ys = [lm.y for lm in visible_points]

    x_min = int(max(0, min(xs) * frame_width))
    y_min = int(max(0, min(ys) * frame_height))
    x_max = int(min(frame_width - 1, max(xs) * frame_width))
    y_max = int(min(frame_height - 1, max(ys) * frame_height))

    return PoseData(
        pose_index=pose_index,

        nose_x=nose.x if nose is not None else None,
        nose_y=nose.y if nose is not None else None,

        left_shoulder_x=left_shoulder.x if left_shoulder is not None else None,
        left_shoulder_y=left_shoulder.y if left_shoulder is not None else None,

        right_shoulder_x=right_shoulder.x if right_shoulder is not None else None,
        right_shoulder_y=right_shoulder.y if right_shoulder is not None else None,

        center_x=center_x,
        center_y=center_y,

        bbox=(x_min, y_min, x_max, y_max),
    )


def create_pose_landmarker():
    """
    MediaPipe Tasks PoseLandmarker 생성.
    num_poses를 MAX_PEOPLE로 설정해야 multi-person pose 결과를 받을 수 있음.
    """
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=MAX_PEOPLE,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    return PoseLandmarker.create_from_options(options)


# =========================
# Drawing helpers
# =========================

def draw_pose_info(frame, pose: PoseData, person_id: int, behavior: dict):
    x_min, y_min, x_max, y_max = pose.bbox

    head_state = behavior["head_turn"].get("state", "unknown")
    down_state = behavior["downward_pose"].get("state", "unknown")
    seat_state = behavior["leaving_seat"].get("state", "unknown")

    is_alert = (
        behavior["head_turn"].get("detected", False)
        or behavior["downward_pose"].get("detected", False)
        or behavior["leaving_seat"].get("detected", False)
    )

    color = (0, 0, 255) if is_alert else (0, 255, 0)

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

    label = f"ID {person_id}"
    cv2.putText(
        frame,
        label,
        (x_min, max(20, y_min - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )

    state_text = f"H:{head_state} D:{down_state} S:{seat_state}"
    cv2.putText(
        frame,
        state_text,
        (x_min, min(frame.shape[0] - 10, y_max + 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
    )


def draw_track_without_pose(frame, track: PersonTrack):
    """
    pose가 잠깐 사라진 track 표시.
    화면 좌표는 normalized center 기준으로 표시.
    """
    cx = int(track.center_x * frame.shape[1])
    cy = int(track.center_y * frame.shape[0])

    text = f"ID {track.person_id}: missing {track.missed_frames}"
    cv2.putText(
        frame,
        text,
        (max(0, cx - 60), max(20, cy)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )


def print_behavior_states(behavior_states: Dict[int, dict], fps: float):
    parts = [f"FPS:{fps:.2f}"]

    for person_id, state in behavior_states.items():
        head = state["head_turn"].get("state", "unknown")
        down = state["downward_pose"].get("state", "unknown")
        seat = state["leaving_seat"].get("state", "unknown")
        missed = state["missed_frames"]

        parts.append(
            f"ID{person_id}[H:{head}, D:{down}, S:{seat}, missed:{missed}]"
        )

    print(" | ".join(parts))


# =========================
# Main
# =========================

def main():
    picam2 = Picamera2()

    config = picam2.create_preview_configuration(
        main={
            "format": "XRGB8888",
            "size": (WIDTH, HEIGHT),
        }
    )

    picam2.configure(config)
    picam2.start()

    monitor = MultiPersonBehaviorMonitor()

    prev_time = time.time()
    fps = 0.0
    frame_counter = 0

    with create_pose_landmarker() as landmarker:
        try:
            while True:
                raw_frame = picam2.capture_array()

                display_frame = raw_frame[:, :, :3].copy()

                # MediaPipe Tasks는 SRGB image를 받음
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = int(time.time() * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                poses: List[PoseData] = []

                if result.pose_landmarks:
                    for pose_index, landmarks in enumerate(result.pose_landmarks):
                        pose_data = extract_pose_data(
                            landmarks=landmarks,
                            pose_index=pose_index,
                            frame_width=WIDTH,
                            frame_height=HEIGHT,
                        )

                        if pose_data is not None:
                            poses.append(pose_data)

                behavior_states = monitor.update(poses)

                # 현재 frame에서 감지된 pose와 매칭된 track 찾아서 draw
                # draw를 위해 다시 가장 가까운 track id를 찾음
                for pose in poses:
                    nearest_track_id = None
                    nearest_dist = float("inf")

                    for track_id, track in monitor.tracks.items():
                        dist = math.sqrt(
                            (track.center_x - pose.center_x) ** 2
                            + (track.center_y - pose.center_y) ** 2
                        )

                        if dist < nearest_dist:
                            nearest_dist = dist
                            nearest_track_id = track_id

                    if nearest_track_id is not None and nearest_track_id in behavior_states:
                        draw_pose_info(
                            frame=display_frame,
                            pose=pose,
                            person_id=nearest_track_id,
                            behavior=behavior_states[nearest_track_id],
                        )

                for track_id, track in monitor.tracks.items():
                    if track.missed_frames > 0:
                        draw_track_without_pose(display_frame, track)

                # FPS 계산
                frame_counter += 1
                now = time.time()
                elapsed = now - prev_time

                if elapsed >= 1.0:
                    fps = frame_counter / elapsed
                    frame_counter = 0
                    prev_time = now
                    print_behavior_states(behavior_states, fps)

                cv2.putText(
                    display_frame,
                    f"FPS: {fps:.2f} | People: {len(poses)} | Tracks: {len(monitor.tracks)}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("SABER Multi-Person Behavior Detection", display_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            picam2.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()