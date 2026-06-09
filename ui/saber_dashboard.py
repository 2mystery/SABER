import time
import cv2
import numpy as np


# =========================
# Layout settings
# =========================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

PANEL_WIDTH = 340

DASHBOARD_WIDTH = CAMERA_WIDTH + PANEL_WIDTH
DASHBOARD_HEIGHT = CAMERA_HEIGHT

CRITICAL_ALERT_SECONDS = 1.0


# =========================
# Color settings (BGR)
# =========================

COLOR_BG = (24, 26, 32)
COLOR_PANEL = (34, 37, 45)
COLOR_CARD = (46, 50, 60)
COLOR_CARD_DARK = (40, 43, 52)

COLOR_TEXT = (235, 235, 235)
COLOR_TEXT_MUTED = (175, 180, 190)
COLOR_WHITE = (255, 255, 255)

COLOR_GREEN = (80, 200, 120)
COLOR_ORANGE = (0, 140, 255)
COLOR_RED = (60, 60, 255)
COLOR_BLUE = (255, 150, 70)
COLOR_GRAY = (130, 130, 130)


# =========================
# Alert timer state
# =========================

_critical_start_time = None


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
    frame = _prepare_camera_frame(camera_frame)

    suspicion_level = compute_suspicion_level(
        head_result=head_result,
        seat_result=seat_result,
        downward_result=downward_result,
    )

    reason = build_reason_text(
        head_result=head_result,
        seat_result=seat_result,
        downward_result=downward_result,
        suspicion_level=suspicion_level,
    )

    alert_active, critical_duration = update_critical_alert_state(
        suspicion_level=suspicion_level,
    )

    dashboard = np.full(
        (DASHBOARD_HEIGHT, DASHBOARD_WIDTH, 3),
        COLOR_BG,
        dtype=np.uint8,
    )

    dashboard[0:CAMERA_HEIGHT, 0:CAMERA_WIDTH] = frame

    _draw_camera_badge(
        dashboard,
        suspicion_level=suspicion_level,
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

    panel_x = CAMERA_WIDTH
    _draw_side_panel_background(dashboard, panel_x)

    _draw_header(
        dashboard,
        x=panel_x + 20,
        y=34,
        pose_fps=pose_fps,
        processed_this_frame=processed_this_frame,
    )

    _draw_suspicion_card(
        dashboard,
        x=panel_x + 20,
        y=110,
        w=PANEL_WIDTH - 40,
        h=78,
        suspicion_level=suspicion_level,
        critical_duration=critical_duration,
    )

    _draw_evidence_section(
        dashboard,
        x=panel_x + 20,
        y=210,
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
        suspicion_level=suspicion_level,
    )

    if alert_active:
        _draw_alert_banner(dashboard)

    return dashboard


# =========================
# Suspicion / Evidence logic
# =========================

def compute_suspicion_level(head_result, seat_result, downward_result):
    seat_state = seat_result.get("state", "unknown")
    downward_state = downward_result.get("state", "unknown")
    head_severity = head_result.get("severity", "low")

    if seat_state == "absent":
        return "CRITICAL"

    if downward_state == "prolonged_downward":
        return "CRITICAL"

    if head_severity == "critical":
        return "CRITICAL"

    if seat_state == "partial_leave":
        return "HIGH"

    if downward_state in ["weak_downward", "downward"]:
        return "HIGH"

    if head_severity == "high":
        return "HIGH"

    if (
        head_result.get("state", "unknown") in ["unknown", "waiting"]
        or seat_state in ["unknown", "waiting"]
        or downward_state in ["unknown", "calibrating", "calibrated"]
    ):
        return "UNKNOWN"

    return "LOW"


def build_reason_text(head_result, seat_result, downward_result, suspicion_level):
    head_state = head_result.get("state", "unknown")
    seat_state = seat_result.get("state", "unknown")
    downward_state = downward_result.get("state", "unknown")
    head_severity = head_result.get("severity", "low")
    turn_count = head_result.get("turn_count", 0)

    if seat_state == "absent":
        return "Student absent from monitored area"

    if downward_state == "prolonged_downward":
        duration = downward_result.get("duration", None)
        if duration is not None:
            return f"Downward posture maintained for {duration:.1f}s"
        return "Prolonged downward posture observed"

    if head_severity == "critical":
        return f"Head turned {turn_count} times"

    if seat_state == "partial_leave":
        return "Student partially left monitored area"

    if downward_state == "downward":
        return "Downward posture observed"

    if downward_state == "weak_downward":
        return "Weak downward movement observed"

    if head_severity == "high":
        return "Head turning observed"

    if head_state in ["left", "right"]:
        return f"Head turned {head_state}"

    if suspicion_level == "UNKNOWN":
        return "Waiting for reliable pose evidence"

    return "No suspicious behavior observed"


def update_critical_alert_state(suspicion_level):
    global _critical_start_time

    now = time.time()

    if suspicion_level == "CRITICAL":
        if _critical_start_time is None:
            _critical_start_time = now

        duration = now - _critical_start_time
        return duration >= CRITICAL_ALERT_SECONDS, duration

    _critical_start_time = None
    return False, 0.0


def get_level_color(level):
    if level == "LOW":
        return COLOR_GREEN
    if level == "HIGH":
        return COLOR_ORANGE
    if level == "CRITICAL":
        return COLOR_RED
    return COLOR_GRAY


def get_state_color(state):
    if state in ["normal", "present", "center"]:
        return COLOR_GREEN

    if state in ["weak_downward", "left", "right", "downward", "partial_leave"]:
        return COLOR_ORANGE

    if state in ["prolonged_downward", "absent"]:
        return COLOR_RED

    if state in ["calibrating", "calibrated"]:
        return COLOR_BLUE

    return COLOR_GRAY


def get_head_color(head_result):
    severity = head_result.get("severity", "low")

    if severity == "critical":
        return COLOR_RED

    if severity == "high":
        return COLOR_ORANGE

    return get_state_color(head_result.get("state", "unknown"))


# =========================
# Drawing helpers
# =========================

def _prepare_camera_frame(camera_frame):
    if camera_frame is None:
        return np.full(
            (CAMERA_HEIGHT, CAMERA_WIDTH, 3),
            (0, 0, 0),
            dtype=np.uint8,
        )

    frame = camera_frame.copy()

    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if frame.shape[2] == 4:
        frame = frame[:, :, :3].copy()

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
        scale=0.95,
        color=COLOR_WHITE,
        thickness=2,
    )

    update_state = "UPDATED" if processed_this_frame else "SKIPPED"
    update_color = COLOR_GREEN if processed_this_frame else COLOR_GRAY

    _draw_badge(
        img,
        text=f"Pose FPS {pose_fps:.1f}",
        x=x,
        y=y + 28,
        color=COLOR_BLUE,
    )

    _draw_badge(
        img,
        text=update_state,
        x=x + 135,
        y=y + 28,
        color=update_color,
    )


def _draw_suspicion_card(img, x, y, w, h, suspicion_level, critical_duration):
    level_color = get_level_color(suspicion_level)

    _draw_card(img, x, y, w, h, border_color=level_color)

    _put_text(
        img,
        "Suspicion Level",
        x + 16,
        y + 24,
        scale=0.52,
        color=COLOR_TEXT_MUTED,
        thickness=1,
    )

    _put_text(
        img,
        suspicion_level,
        x + 16,
        y + 60,
        scale=1.0,
        color=level_color,
        thickness=2,
    )

    if suspicion_level == "CRITICAL":
        _put_text(
            img,
            f"{critical_duration:.1f}s",
            x + w - 64,
            y + 60,
            scale=0.48,
            color=COLOR_TEXT_MUTED,
            thickness=1,
        )


def _draw_evidence_section(img, x, y, w, head_result, seat_result, downward_result):
    _put_text(
        img,
        "Behavior Evidence",
        x,
        y,
        scale=0.58,
        color=COLOR_WHITE,
        thickness=2,
    )

    card_y = y + 16
    card_h = 40
    gap = 8

    _draw_evidence_card(
        img,
        x=x,
        y=card_y,
        w=w,
        h=card_h,
        title="Head Movement",
        state=head_result.get("state", "unknown"),
        border_color=get_head_color(head_result),
        extra=f"turns: {head_result.get('turn_count', 0)}",
    )

    _draw_evidence_card(
        img,
        x=x,
        y=card_y + (card_h + gap),
        w=w,
        h=card_h,
        title="Downward Posture",
        state=downward_result.get("state", "unknown"),
        border_color=get_state_color(downward_result.get("state", "unknown")),
    )

    _draw_evidence_card(
        img,
        x=x,
        y=card_y + (card_h + gap) * 2,
        w=w,
        h=card_h,
        title="Seat Presence",
        state=seat_result.get("state", "unknown"),
        border_color=get_state_color(seat_result.get("state", "unknown")),
    )


def _draw_evidence_card(img, x, y, w, h, title, state, border_color, extra=None):
    _draw_card(img, x, y, w, h, border_color=border_color)

    _put_text(
        img,
        title,
        x + 12,
        y + 16,
        scale=0.40,
        color=COLOR_TEXT_MUTED,
        thickness=1,
    )

    _put_text(
        img,
        state,
        x + 12,
        y + 34,
        scale=0.48,
        color=border_color,
        thickness=2,
    )

    if extra:
        _put_text(
            img,
            extra,
            x + w - 84,
            y + 34,
            scale=0.38,
            color=COLOR_TEXT_MUTED,
            thickness=1,
        )


def _draw_reason_section(img, x, y, w, reason, suspicion_level):
    level_color = get_level_color(suspicion_level)

    _put_text(
        img,
        "Reason",
        x,
        y,
        scale=0.58,
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

    wrapped_lines = _wrap_text(reason, max_chars=26)

    line_y = y + 42
    for line in wrapped_lines[:2]:
        _put_text(
            img,
            line,
            x + 12,
            line_y,
            scale=0.52,
            color=COLOR_TEXT,
            thickness=1,
        )
        line_y += 22


def _draw_alert_banner(img):
    x = 0
    y = 0
    w = CAMERA_WIDTH
    h = 42

    overlay = img.copy()

    cv2.rectangle(
        overlay,
        (x, y),
        (x + w, y + h),
        COLOR_RED,
        -1,
    )

    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)

    _put_text(
        img,
        "[ALERT] Critical suspicious behavior",
        x + 18,
        y + 27,
        scale=0.62,
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
    text_width = max(100, len(text) * 9)
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
        next_text = f"{current} {word}".strip()

        if len(next_text) <= max_chars:
            current = next_text
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines