# One-Person Threshold Tuning Result

## Test Environment

| Item             | Value                             |
| ---------------- | --------------------------------- |
| Device           | Raspberry Pi 4                    |
| Camera           | Raspberry Pi Camera Module 3 Wide |
| Input Resolution | 640 × 480                         |
| Pose Model       | MediaPipe Pose                    |
| Model Complexity | 0                                 |
| Frame Skip       | 2                                 |
| Camera FPS       | Approximately 12 FPS              |
| Pose FPS         | Approximately 6 FPS               |
| Test Mode        | One-person pose monitoring        |

## Final Threshold Configuration

```python
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
```

---

# 1. Head Turning Detection

## Purpose

Head Turning detection identifies repeated left-right head movements that may indicate suspicious behavior during an exam.

## Detection Logic

The detector uses the relative position of the nose with respect to the shoulder center.

```python
head_offset = nose_x - shoulder_center_x
```

Using the shoulder center instead of the absolute `nose.x` value makes the detection more robust to the student's position in the frame.

## Final Parameters

| Parameter          | Final Value | Description                                                                             |
| ------------------ | ----------: | --------------------------------------------------------------------------------------- |
| `offset_threshold` |      `0.04` | Minimum horizontal head offset required to classify the head direction as left or right |
| `min_turn_changes` |         `3` | Minimum number of left-right direction changes required to detect repeated head turning |
| `history_size`     |        `30` | Number of recent pose updates used for temporal analysis                                |

## Final Decision

The final `offset_threshold` was set to `0.04` because it provided stable detection of repeated head turning while maintaining acceptable sensitivity for one-person monitoring.

---

# 2. Leaving Seat Detection

## Purpose

Leaving Seat detection identifies when the student is no longer present in the monitored area.

## Detection Logic

The detector checks whether MediaPipe Pose landmarks are missing for consecutive pose update frames.

If no pose is detected continuously for a certain number of frames, the system determines that the student has left the monitored area.

## Final Parameters

| Parameter                 | Final Value | Description                                                                               |
| ------------------------- | ----------: | ----------------------------------------------------------------------------------------- |
| `no_pose_frame_threshold` |         `6` | Number of consecutive no-pose frames required to trigger leaving seat detection           |
| `upper_body_y_threshold`  |      `0.15` | Upper-body landmark position threshold used for additional seat/upper-body state checking |

## Final Decision

The final `no_pose_frame_threshold` was set to `6`.
Since the pose processing speed is approximately 6 FPS, this means that the system can detect a complete leaving-seat event after roughly 1 second of continuous pose absence.

This value was selected because it provides fast detection while still reducing false alarms from short temporary pose losses.

---

# 3. Downward Posture Detection

## Purpose

Downward Posture detection identifies when the student keeps looking downward for a prolonged period.

This behavior may indicate suspicious attention shift during a monitored exam.

## Detection Logic

The detector uses an initial calibration period to estimate the student's normal posture.
After calibration, it compares the current nose position with the calibrated baseline.

If the nose moves downward beyond the configured threshold and remains in that state for a certain duration, the system triggers a downward posture warning.

## Final Parameters

| Parameter             | Final Value | Description                                                |
| --------------------- | ----------: | ---------------------------------------------------------- |
| `calibration_seconds` |       `3.0` | Initial time used to estimate the normal sitting posture   |
| `weak_downward_delta` |      `0.04` | Small downward movement threshold                          |
| `downward_delta`      |      `0.08` | Main downward posture threshold                            |
| `prolonged_seconds`   |       `2.5` | Required duration for prolonged downward posture detection |

## Final Decision

The final `downward_delta` was set to `0.08`, and `prolonged_seconds` was set to `2.5`.

This configuration prevents brief natural downward movements from being immediately classified as suspicious, while still detecting prolonged downward posture.

---

# Summary

| Behavior         |          Final Main Threshold | Temporal Condition                                          | Expected Detection                 |
| ---------------- | ----------------------------: | ----------------------------------------------------------- | ---------------------------------- |
| Head Turning     |     `offset_threshold = 0.04` | At least 3 left-right changes within recent 30 pose updates | Repeated head turning              |
| Leaving Seat     | `no_pose_frame_threshold = 6` | No pose detected for about 1 second                         | Student absent from monitored area |
| Downward Posture |       `downward_delta = 0.08` | Downward posture maintained for 2.5 seconds                 | Prolonged downward posture         |

## Final Notes

These thresholds were finalized for the current one-person monitoring setup.
They are suitable for the current baseline environment:

* Raspberry Pi 4
* Raspberry Pi Camera Module 3 Wide
* 640 × 480 resolution
* MediaPipe Pose `model_complexity = 0`
* `frame_skip = 2`

For future multi-person monitoring, these thresholds may need to be adjusted again because camera distance, student scale, landmark visibility, and occlusion patterns can change.
