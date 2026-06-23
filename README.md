# SABER(Student Attention and BEhavior monitoR) : Real-Time Student Attention and Behavior Monitoring for Smart Proctoring

**SABER** is a lightweight edge AI system for real-time student attention and suspicious behavior monitoring.
It runs on **Raspberry Pi 4** with **Raspberry Pi Camera Module 3 Wide**, uses **MediaPipe Pose** to extract pose landmarks, and detects suspicious behavior patterns using **rule-based temporal logic**.

SABER does **not** directly classify cheating.
Instead, it provides explainable suspicious behavior evidence to assist human proctors.

---

## Project Overview

SABER monitors three suspicious behavior patterns

1. **Repeated Head Turning**
   Detects repeated left-right head movements using nose and shoulder landmarks.

2. **Downward Posture**
   Detects downward posture using nose y-position changes after initial calibration.

3. **Leaving Seat**
   Detects when the student disappears from the monitored area by counting consecutive missing-pose frames.

The system provides warning messages through a dashboard UI, including Pose FPS, suspicion level, behavior evidence, reason messages, and alert banners.

---

## System Pipeline

```text
Camera
→ Frame Capture
→ MediaPipe Pose
→ Pose Landmarks
→ Temporal Behavior Detectors
→ SABER Dashboard Warning
```

The main design principle is to avoid single-frame decisions.
Instead, SABER uses temporal logic to reduce false positives caused by short movements or temporary pose estimation failures.

---

## Hardware

| Component                         | Description                       |
| --------------------------------- | --------------------------------- |
| Raspberry Pi 4                    | Main edge computing device        |
| Raspberry Pi Camera Module 3 Wide | Camera input module               |
| SD Card                           | Raspberry Pi OS and project files |
| Monitor / VNC Viewer              | Execution screen monitoring       |
| Putty / SSH                       | Remote terminal access            |

---

## Software

| Software        | Role                               |
| --------------- | ---------------------------------- |
| Python          | Main programming language          |
| OpenCV          | Frame processing and visualization |
| MediaPipe Pose  | Pose landmark estimation           |
| Picamera2       | Raspberry Pi camera capture        |
| Raspberry Pi OS | Runtime environment                |

---

## Repository Structure

```text
SABER/
├── behavior/
│   ├── head_turn_detector.py
│   ├── downward_pose_detector.py
│   ├── leaving_seat_detector.py
│   ├── leaving_seat_detector2.py
│   ├── one_person_demo.py
│   ├── one_person_pose_test.py
│   └── multi_person_pose_test.py
│
├── camera/
│   └── camera_fps_test.py
│
├── demo/
│   └── fps_experiment_results.md
│   └── one_person_pose_results.md
│   └── one_person_threshold_tuning.md
|
├── pose/
│   ├── mediapipe_pose_fps_test.py
│   ├── mediapipe_pose_fps_test1.py
│   └── one_person_pose_test.py
│
├── ui/
│   └── saber_dashboard.py
│
├── utils/
│   └── fps.py
│
├── demo/
│   ├── fps_experiment_results.md
│   ├── one_person_pose_results.md
│   └── one_person_threshold_tuning.md
│
└── README.md
```

---

## Final Runtime Configuration

The final runtime configuration was selected based on FPS experiments on Raspberry Pi 4.

| Setting                    | Value                 |
| -------------------------- | --------------------- |
| Resolution                 | 640 × 480             |
| MediaPipe model_complexity | 0                     |
| Frame skip                 | 2                     |
| Target setting             | One-person monitoring |

Initial FPS tests showed that camera capture was not the main bottleneck.
Camera-only FPS was about 33 FPS, but FPS decreased significantly when MediaPipe Pose inference was included.
Therefore, Pose FPS was used as the main real-time performance metric.

---

## Detector Configuration

### Head Turning Detector

```python
HEAD_TURN_CONFIG = {
    "offset_threshold": 0.04,
    "min_turn_changes": 3,
    "history_size": 30,
}
```

The detector calculates the relative head offset:

```python
shoulder_center_x = (left_shoulder_x + right_shoulder_x) / 2
head_offset = nose_x - shoulder_center_x
```

If repeated left-right state changes occur within the recent history window, the system detects repeated head turning.

---

### Downward Posture Detector

```python
DOWNWARD_POSE_CONFIG = {
    "calibration_seconds": 3.0,
    "weak_downward_delta": 0.04,
    "downward_delta": 0.08,
    "prolonged_seconds": 2.5,
}
```

The detector first calibrates the normal posture baseline for 3 seconds.
After calibration, it detects downward posture using the change in nose y-position.

---

### Leaving Seat Detector

```python
LEAVING_SEAT_CONFIG = {
    "no_pose_frame_threshold": 6,
    "upper_body_y_threshold": 0.15,
}
```

The detector counts consecutive frames where pose landmarks are not detected.
If the no-pose count reaches the threshold, the system detects that the student is absent from the monitored area.

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/SABER.git
cd SABER
```

### 2. Install dependencies

```bash
pip install opencv-python mediapipe numpy
```

For Raspberry Pi camera input, Picamera2 should be installed and configured on Raspberry Pi OS.

```bash
sudo apt install -y python3-picamera2
```

### 3. Run the one-person demo

Recommended:

```bash
python -m behavior.one_person_demo
```

If the module command does not work in your environment, run:

```bash
python behavior/one_person_demo.py
```

---

## Evaluation Results

### Pose FPS

| Scenario           | Avg Pose FPS |
| ------------------ | -----------: |
| Normal sitting     |         4.60 |
| Head turning       |         4.83 |
| Downward posture   |         4.90 |
| Leaving seat       |         4.86 |
| One-person average |         4.80 |
| Two-person test    |          < 3 |

### Detection Latency

| Behavior         | Avg Latency |
| ---------------- | ----------: |
| Head turning     |      0.54 s |
| Downward posture |      0.03 s |
| Leaving seat     |      0.94 s |

Note: Downward posture latency was measured until the first downward state, including weak downward, not until the final prolonged warning.

### Resource Usage

| Scenario              | CPU Usage | RAM Usage | Temperature |
| --------------------- | --------: | --------: | ----------: |
| One-person monitoring |     51.5% |   731 MiB |      65.7°C |

---

## Dashboard Output

The SABER dashboard displays:

* Real-time camera frame
* MediaPipe Pose landmarks
* Pose FPS
* Pose update status
* Suspicion level
* Behavior evidence
* Warning reason
* Alert banner

Each detector returns `detected`, `message`, and `state`, and the dashboard converts these outputs into explainable warning messages.

---

## Failure Cases

SABER has several limitations because it relies on 2D pose landmarks and rule-based temporal logic.

### 1. Writing on Paper

Writing can be falsely detected as downward posture because the nose moves downward in a similar way.

### 2. Occlusion

Partial or full occlusion can make pose landmarks unstable or missing.

### 3. Initial Non-frontal Posture

If the user is not facing forward during the initial calibration, the downward posture baseline can be incorrectly estimated.

### 4. Multi-person FPS Limitation

Multi-person monitoring was implemented experimentally, but Pose FPS dropped below 3 on Raspberry Pi 4 with two people.
Therefore, final evaluation was conducted in the one-person setting.

---

## Main Contributions

* Implemented a real-time edge AI monitoring prototype on Raspberry Pi 4
* Designed three rule-based temporal behavior detectors
* Used pose landmarks to provide explainable suspicious behavior evidence
* Selected runtime configuration through FPS experiments
* Tuned detector thresholds through repeated behavior trials
* Evaluated Pose FPS, detection latency, and resource usage
* Analyzed practical failure cases and future improvement directions

---

## Future Work

* Multi-person monitoring optimization
* Occlusion robustness improvement
* Adaptive calibration and manual recalibration
* Distinguishing writing posture from suspicious downward posture
* Hand tracking or object detection
* Lightweight person tracking
* Detector confidence score
* Raspberry Pi 5 or AI accelerator deployment
* Testing in more diverse classroom environments

---

## Team
Course Info: IOT systems, HUFS
### shinbi Lee, Gyuri Yoon 🤗🤩🤓👻👾

---
