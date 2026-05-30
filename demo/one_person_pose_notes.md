# One-Person Pose Detection Test Results

## Objective

The objective of this experiment was to verify the stability of MediaPipe Pose on Raspberry Pi 4 and investigate landmark behavior under different student actions before implementing suspicious behavior detection logic.

---

## Experimental Setup

### Hardware

* Raspberry Pi 4
* Raspberry Pi Camera Module 3 Wide

### Software

* Python
* OpenCV
* MediaPipe Pose

### Configuration

| Parameter        | Value     |
| ---------------- | --------- |
| Resolution       | 640 × 480 |
| Model Complexity | 0         |
| Frame Skip       | 2         |
| Pose FPS         | ~6 FPS    |
| Camera FPS       | ~12 FPS   |

---

## Tested Scenarios

### 1. Normal Sitting Posture

| Metric          | Value |
| --------------- | ----- |
| Nose x          | 0.45  |
| Nose y          | 0.33  |
| Nose visibility | 0.997 |

Observation:

* Stable landmark detection.
* High visibility values (>0.99).
* Used as baseline posture.

---

### 2. Head Turned Left

| Metric          | Value |
| --------------- | ----- |
| Nose x          | 0.53  |
| Nose y          | 0.36  |
| Nose visibility | 0.98  |

Observation:

* Nose x coordinate increased noticeably.
* Pose landmarks remained stable.

---

### 3. Head Turned Right

| Metric          | Value |
| --------------- | ----- |
| Nose x          | 0.39  |
| Nose y          | 0.39  |
| Nose visibility | 0.99  |

Observation:

* Nose x coordinate decreased noticeably.
* Clear distinction from normal posture.

---

### 4. Prolonged Downward Posture

| Metric          | Value |
| --------------- | ----- |
| Nose x          | 0.42  |
| Nose y          | 0.58  |
| Nose visibility | 0.99  |

Observation:

* Nose y coordinate increased significantly.
* Strong indicator for downward posture detection.

---

### 5. Partial Leaving of Camera View

| Metric          | Value |
| --------------- | ----- |
| Nose x          | 0.40  |
| Nose y          | 0.01  |
| Nose visibility | 0.99  |

Observation:

* Visibility remained high despite partial departure from view.
* Visibility alone may not be sufficient for leaving-seat detection.

---

### 6. Complete Leaving of Seat

Observation:

* Pose landmarks disappeared.
* "No person detected" message displayed.
* Suitable for absence monitoring.

---

### 7. Fast Head Movement

Observation:

* Landmark tracking remained stable.
* Temporary fluctuations were minimal.

---

### 8. Low-Light Condition

Observation:

* Pose detection occasionally failed during initialization.
* Tracking stabilized after approximately 1–2 seconds.
* Landmarks were recovered successfully.

---

## Key Findings

### Head Turning

Normal:

* Nose x ≈ 0.45

Left Turn:

* Nose x ≈ 0.53

Right Turn:

* Nose x ≈ 0.39

Result:

* Nose x changes consistently with head orientation.
* Head turning detection can be implemented using nose x movement.

---

### Downward Posture

Normal:

* Nose y ≈ 0.33

Downward:

* Nose y ≈ 0.58

Result:

* Significant increase observed.
* Downward posture is the easiest suspicious behavior to detect using simple threshold logic.

---

### Leaving Seat

Result:

* Landmark visibility alone is insufficient.
* Landmark disappearance and pose detection failure should be used together.

---

## Conclusion

The experiment confirmed that MediaPipe Pose provides stable landmark detection on Raspberry Pi 4 under the selected baseline configuration.

The observed landmark variations indicate that the following suspicious behaviors can be implemented using lightweight temporal rule-based logic:

1. Head Turning
2. Downward Posture
3. Leaving Seat

These results will be used to design temporal suspicious behavior detectors in the next development stage.
