import cv2
import numpy as np


# =========================
# Layout settings
# =========================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

PANEL_WIDTH = 320
DASHBOARD_WIDTH = CAMERA_WIDTH + PANEL_WIDTH
DASHBOARD_HEIGHT = CAMERA_HEIGHT


# =========================
# Color settings
# OpenCV uses BGR, not RGB
# =========================

COLOR_BG = (24, 26, 32)
COLOR_PANEL = (34, 37, 45)
COLOR_CARD = (46, 50, 60)
COLOR_CARD_DARK = (38, 41, 50)

COLOR_TEXT = (235, 235, 235)
COLOR_TEXT_MUTED = (170, 175, 185)
COLOR_WHITE = (255, 255, 255)

COLOR_GREEN = (80, 200, 120)
COLOR_YELLOW = (0, 220, 255)
COLOR_ORANGE = (0, 140, 255)
COLOR_RED = (60, 60, 255)
COLOR_BLUE = (255, 150, 70)
COLOR_GRAY = (130, 130, 130)


# =========================
# Public renderer
# =========================

def render_one_person_dashboard(
    camera_frame,
    head_result,
    seat_result,
    downward_result,
    pose_fps=0.0,
    processed_this_frame=False,
    debug_mode=False,
    landmark_values=None,
):
    """
    One-person SABER dashboard renderer.

    camera_frame:
        OpenCV BGR frame, expected size 640x480.

    head_result, seat_result, downward_result:
        detector.update() result dictionaries.

    pose_fps:
        Pose inference FPS only. Camera FPS is intentionally hidden.

    processed_this_frame:
        Whether pose inference was actually processed in this frame.

    debug_mode:
        If True, raw landmark values are shown.

    landmark_values:
        Optional dict:
        {
            "nose_x": ...,
            "nose_y": ...,
            "left_shoulder_y": ...,
            "right_shoulder_y": ...
        }
    """

    frame = _prepare_camera_frame(camera_frame)

    suspicion = compute_suspicion_level(
        head_result=head_result,
        seat_result=seat_result,
        downward_result=downward_result,
    )

    reason = build_reason_text(
        head_result=head_result,
        seat_result=seat_result,
        downward_result=downward_result,
        suspicion_level=suspicion,
    )

    dashboard = np.full(
        (DASHBOARD_HEIGHT, DASHBOARD_WIDTH, 3),
        COLOR_BG,
        dtype=np.uint8,
    )

    # Left: camera frame
    dashboard[0:CAMERA_HEIGHT, 0:CAMERA_WIDTH] = frame

    # Camera overlay
    _draw_camera_badge(
        dashboard,
        suspicion_level=suspicion,
        x=18,
        y=18,
    )

    if debug_mode:
        _draw_debug_overlay(
            dashboard,
            landmark_values=landmark_values,
            head_result=head_result,
            processed_this_frame=processed_this_frame,
        )

    # Right: dashboard panel
    panel_x = CAMERA_WIDTH
    _draw_side_panel_background(dashboard, panel_x)

    _draw_header(
        dashboard,
        x=panel_x + 20,
        y=28,
        pose_fps=pose_fps,
        processed_this_frame=processed_this_frame,
    )

    _draw_suspicion_card(
        dashboard,
        x=panel_x + 20,
        y=118,
        w=PANEL_WIDTH - 40,
        h=86,
        suspicion_level=suspicion,
    )

    _draw_evidence_section(
        dashboard,
        x=panel_x + 20,
        y=225,
        w=PANEL_WIDTH - 40,
        head_result=head_result,
        seat_result=seat_result,
        downward_result=downward_result,
    )

    _draw_reason_section(
        dashboard,
        x=panel_x + 20,
        y=388,
        w=PANEL_WIDTH - 40,
        reason=reason,
        suspicion_level=suspicion,
    )

    if suspicion in ["HIGH", "CRITICAL"]:
        _draw_alert_banner(
            dashboard,
            suspicion_level=suspicion,
            reason=reason,
        )

    return dashboard


# =========================
# Suspicion / Evidence logic
# =========================

def compute_suspicion_level(head_result, seat_result, downward_result):
    head_state = head_result.get("state", "unknown")
    seat_state = seat_result.get("state", "unknown")
    downward_state = downward_result.get("state", "unknown")

    head_detected = head_result.get("detected", False)

    # Highest severity first
    if seat_state == "absent":
        return "CRITICAL"

    if downward_state == "prolonged_downward":
        return "CRITICAL"

    if seat_state == "partial_leave":
        return "HIGH"

    if downward_state == "downward":
        return "HIGH"

    if head_detected:
        return "HIGH"

    if downward_state == "weak_downward":
        return "MEDIUM"

    if head_state in ["left", "right"]:
        return "MEDIUM"

    if head_state in ["unknown", "waiting"] or seat_state in ["unknown", "waiting"] or downward_state in ["unknown", "calibrating", "calibrated"]:
        return "UNKNOWN"

    return "LOW"


def build_reason_text(head_result, seat_result, downward_result, suspicion_level):
    head_state = head_result.get("state", "unknown")
    seat_state = seat_result.get("state", "unknown")
    downward_state = downward_result.get("state", "unknown")

    if seat_state == "absent":
        return "Student absent from monitored area"

    if downward_state == "prolonged_downward":
        duration = downward_result.get("duration", None)

        if duration is not None:
            return f"Downward posture maintained for {duration:.1f}s"

        return "Prolonged downward posture observed"

    if seat_state == "partial_leave":
        return "Student partially left monitored area"

    if downward_state == "downward":
        return "Downward posture observed"

    if head_result.get("detected", False):
        return "Repeated head turning observed"

    if downward_state == "weak_downward":
        return "Weak downward movement observed"

    if head_state in ["left", "right"]:
        return f"Head turned {head_state}"

    if suspicion_level == "UNKNOWN":
        return "Waiting for reliable pose evidence"

    return "No suspicious behavior observed"


def get_level_color(level):
    if level == "LOW":
        return COLOR_GREEN

    if level == "MEDIUM":
        return COLOR_YELLOW

    if level == "HIGH":
        return COLOR_ORANGE

    if level == "CRITICAL":
        return COLOR_RED

    return COLOR_GRAY


def get_state_color(state):
    if state in ["normal", "present", "center"]:
        return COLOR_GREEN

    if state in ["weak_downward", "left", "right"]:
        return COLOR_YELLOW

    if state in ["downward", "partial_leave"]:
        return COLOR_ORANGE

    if state in ["prolonged_downward", "absent"]:
        return COLOR_RED

    if state in ["calibrating", "calibrated"]:
        return COLOR_BLUE

    return COLOR_GRAY


# =========================
# Drawing helpers
# =========================

def _prepare_camera_frame(camera_frame):
    if camera_frame is None:
        return np.full((CAMERA_HEIGHT, CAMERA_WIDTH, 3), (0, 0, 0), dtype=np.uint8)

    frame = camera_frame.copy()

    if frame.shape[1] != CAMERA_WIDTH or frame.shape[0] != CAMERA_HEIGHT:
        frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))

    return frame


def _draw_side_panel_background(img, panel_x):
    cv2.rectangle(
        img,
        (panel_x, 0),
        (DASHBOARD_WIDTH, DASHBOARD_HEIGHT),
        COLOR_PANEL,
        -1,
    )

    cv2.line(
        img,
        (panel_x, 0),
        (panel_x, DASHBOARD_HEIGHT),
        (65, 70, 82),
        1,
    )


def _draw_header(img, x, y, pose_fps, processed_this_frame):
    _put_text(
        img,
        "SABER",
        x,
        y,
        scale=0.9,
        color=COLOR_WHITE,
        thickness=2,
    )

    _put_text(
        img,
        "Explainable Suspicious",
        x,
        y + 32,
        scale=0.48,
        color=COLOR_TEXT_MUTED,
        thickness=1,
    )

    _put_text(
        img,
        "Behavior Monitoring",
        x,
        y + 54,
        scale=0.48,
        color=COLOR_TEXT_MUTED,
        thickness=1,
    )

    update_state = "UPDATED" if processed_this_frame else "SKIPPED"
    update_color = COLOR_GREEN if processed_this_frame else COLOR_GRAY

    _draw_badge(
        img,
        text=f"Pose FPS {pose_fps:.1f}",
        x=x,
        y=y + 74,
        color=COLOR_BLUE,
    )

    _draw_badge(
        img,
        text=update_state,
        x=x + 130,
        y=y + 74,
        color=update_color,
    )


def _draw_suspicion_card(img, x, y, w, h, suspicion_level):
    level_color = get_level_color(suspicion_level)

    _draw_card(img, x, y, w, h, border_color=level_color)

    _put_text(
        img,
        "Suspicion Level",
        x + 16,
        y + 26,
        scale=0.5,
        color=COLOR_TEXT_MUTED,
        thickness=1,
    )

    _put_text(
        img,
        suspicion_level,
        x + 16,
        y + 65,
        scale=1.0,
        color=level_color,
        thickness=2,
    )


def _draw_evidence_section(img, x, y, w, head_result, seat_result, downward_result):
    _put_text(
        img,
        "Behavior Evidence",
        x,
        y,
        scale=0.55,
        color=COLOR_WHITE,
        thickness=2,
    )

    card_y = y + 18
    card_h = 42
    gap = 10

    _draw_evidence_card(
        img,
        x=x,
        y=card_y,
        w=w,
        h=card_h,
        title="Head Movement",
        state=head_result.get("state", "unknown"),
    )

    _draw_evidence_card(
        img,
        x=x,
        y=card_y + card_h + gap,
        w=w,
        h=card_h,
        title="Downward Posture",
        state=downward_result.get("state", "unknown"),
    )

    _draw_evidence_card(
        img,
        x=x,
        y=card_y + (card_h + gap) * 2,
        w=w,
        h=card_h,
        title="Seat Presence",
        state=seat_result.get("state", "unknown"),
    )


def _draw_evidence_card(img, x, y, w, h, title, state):
    state_color = get_state_color(state)

    _draw_card(img, x, y, w, h, border_color=state_color)

    _put_text(
        img,
        title,
        x + 12,
        y + 18,
        scale=0.42,
        color=COLOR_TEXT_MUTED,
        thickness=1,
    )

    _put_text(
        img,
        state,
        x + 12,
        y + 36,
        scale=0.48,
        color=state_color,
        thickness=2,
    )


def _draw_reason_section(img, x, y, w, reason, suspicion_level):
    level_color = get_level_color(suspicion_level)

    _put_text(
        img,
        "Reason",
        x,
        y,
        scale=0.55,
        color=COLOR_WHITE,
        thickness=2,
    )

    _draw_card(
        img,
        x=x,
        y=y + 14,
        w=w,
        h=64,
        border_color=level_color,
        fill_color=COLOR_CARD_DARK,
    )

    wrapped_lines = _wrap_text(reason, max_chars=27)

    line_y = y + 40
    for line in wrapped_lines[:2]:
        _put_text(
            img,
            line,
            x + 12,
            line_y,
            scale=0.43,
            color=COLOR_TEXT,
            thickness=1,
        )
        line_y += 20


def _draw_alert_banner(img, suspicion_level, reason):
    color = get_level_color(suspicion_level)

    x = 0
    y = 0
    w = CAMERA_WIDTH
    h = 44

    overlay = img.copy()

    cv2.rectangle(
        overlay,
        (x, y),
        (x + w, y + h),
        color,
        -1,
    )

    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    _put_text(
        img,
        "[ALERT] Suspicious behavior observed",
        x + 18,
        y + 27,
        scale=0.65,
        color=COLOR_WHITE,
        thickness=2,
    )


def _draw_camera_badge(img, suspicion_level, x, y):
    color = get_level_color(suspicion_level)

    _draw_badge(
        img,
        text=f"Student 1 | {suspicion_level}",
        x=x,
        y=y,
        color=color,
    )

    # camera area border
    cv2.rectangle(
        img,
        (0, 0),
        (CAMERA_WIDTH - 1, CAMERA_HEIGHT - 1),
        color,
        2,
    )


def _draw_debug_overlay(img, landmark_values, head_result, processed_this_frame):
    if landmark_values is None:
        return

    nose_x = landmark_values.get("nose_x", 0.0)
    nose_y = landmark_values.get("nose_y", 0.0)
    left_shoulder_y = landmark_values.get("left_shoulder_y", 0.0)
    right_shoulder_y = landmark_values.get("right_shoulder_y", 0.0)
    head_offset = head_result.get("head_offset", 0.0)

    update_text = "Pose Update: YES" if processed_this_frame else "Pose Update: SKIP"

    debug_lines = [
        update_text,
        f"nose.x={nose_x:.3f} nose.y={nose_y:.3f}",
        f"offset={head_offset:.3f}",
        f"L_sh.y={left_shoulder_y:.3f} R_sh.y={right_shoulder_y:.3f}",
    ]

    x = 18
    y = CAMERA_HEIGHT - 88

    cv2.rectangle(
        img,
        (x - 8, y - 22),
        (x + 360, y + 74),
        (0, 0, 0),
        -1,
    )

    for line in debug_lines:
        _put_text(
            img,
            line,
            x,
            y,
            scale=0.45,
            color=COLOR_WHITE,
            thickness=1,
        )
        y += 22


def _draw_card(img, x, y, w, h, border_color=COLOR_GRAY, fill_color=COLOR_CARD):
    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        fill_color,
        -1,
    )

    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        border_color,
        1,
    )


def _draw_badge(img, text, x, y, color):
    text_width = max(90, len(text) * 9)
    badge_h = 26

    cv2.rectangle(
        img,
        (x, y),
        (x + text_width, y + badge_h),
        color,
        -1,
    )

    _put_text(
        img,
        text,
        x + 8,
        y + 18,
        scale=0.45,
        color=COLOR_WHITE,
        thickness=1,
    )


def _put_text(img, text, x, y, scale=0.5, color=COLOR_TEXT, thickness=1):
    cv2.putText(
        img,
        str(text),
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _wrap_text(text, max_chars=28):
    words = str(text).split()
    lines = []
    current = ""

    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current = f"{current} {word}".strip()
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines