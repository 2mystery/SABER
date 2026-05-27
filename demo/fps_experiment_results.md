# SABER - FPS Experiment Results

## FPS Results

| Test | Resolution | Person | Model | Frame Skip | FPS |
|---|---|---|---|---|---|
| Camera only | 640x480 | - | - | - | 33 |
| Camera only | 1280x720 | - | - | - | 33 |
| MediaPipe Pose | 640x480 | 1 | complexity 0 | 1 | 9.8 |
| MediaPipe Pose | 640x480 | 1 | complexity 0 | 2 | 13.5 |
| MediaPipe Pose | 1280x720 | 1 | complexity 0 | 2 | 10 |
| MediaPipe Pose | 640x480 | 1 | complexity 1 | 1 | 7 |

---

## Analysis

- Camera-only FPS remained around 33 FPS regardless of resolution.
- `model_complexity=1` significantly reduced FPS on Raspberry Pi 4.
- `frame_skip=2` improved real-time performance.
- 1280x720 resolution reduced FPS without major benefits... 
- 640x480 provided the best balance between performance and pose detection.

---

## Final Baseline Configuration

```text
Resolution: 640x480
MediaPipe Pose model_complexity: 0
Frame Skip: 2
Target FPS: approximately 13 FPS
Person: single-person monitoring
```

---

## Conclusion

The following setting was selected as the final baseline configuration for the SABER project:

- 640x480 resolution
- MediaPipe Pose complexity 0
- frame_skip=2

This configuration achieved the best trade-off between:
- real-time performance
- lightweight edge AI processing
- stable pose landmark detection