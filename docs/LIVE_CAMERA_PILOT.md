# Local camera pilot - 2026-09-17

The standalone service was switched from the recorded-video example to a Git-ignored config.local.json. It includes the C920 webcam, the previously configured ESP32 gripper camera, and the recorded deer example for comparison. The model uses CUDA and cached weights only.

## Executed checks

| Check | Observed outcome |
|---|---|
| Device selection | Windows Media Foundation enumeration identified the physical C920 at index 0; the capture card was index 1. |
| Webcam capture | Completed a 10-second capture window with 9 processed frames at 640x480. Median inference 177.48 ms; first inference p95 694.47 ms. |
| HTTP responsiveness | Health endpoint p95 12.57 ms during the first trial. |
| Missing network camera | The gripper status request timed out; its stream could not be opened. |
| Failure isolation | With both sources selected, the webcam processed 9 frames at 178 ms median while the gripper reported unavailable and zero frames. |
| Live cancellation | Cancellation acknowledged in 1.85 ms after 2 processed frames. This is acknowledgement time, not a measurement of physical camera release latency. |
| Reacquisition | The next webcam search completed with 5 frames, median inference 170.83 ms, total 5.48 seconds. |

The initial search took 40.45 seconds including cold process/model startup and a 10-second capture window. Warm inference timing should not be interpreted as startup time.

The target phrase was mug. No matches were reported. The user had not confirmed placement of a test object, so this verifies the acquisition/inference path, not object-detection accuracy. No camera images or video were written to disk. Performance-only summaries are in the ignored .work directory.

The gripper's timeout does not establish whether it is unpowered, disconnected, or at a different address. No router configuration was changed.

## Continue the supervised trial

1. Open http://127.0.0.1:8765 and select only C920 desk webcam.
2. Place a mug in view, enter mug, enable Timed search, choose 10 seconds, and start.
3. Check that the box is on the actual mug. Repeat with the mug moved, partly occluded, and removed.
4. Power the ESP32 gripper camera and verify its address against config.local.json.
5. Test the gripper alone, then both cameras together.
6. With the owner present, test a real unplug/replug and network interruption.

To restart on this PC, from the repository directory:

```powershell
.venv/Scripts/python.exe -m grounded_stream serve --config config.local.json --port 8765
```

Do not run another USB preview in a browser or desktop application while this service owns the camera. Camera indices can change after reconnecting devices.

For device discovery, the diagnostic environment used cv2-enumerate-cameras 1.3.3. Its [upstream documentation](https://github.com/lukehugh/cv2_enumerate_cameras/blob/main/README.md) describes backend-specific enumeration. It is an optional diagnostic helper, not a runtime dependency.

Live gripper operation, object-grounded accuracy, actual unplug/reconnect, and multi-hour behavior remain unverified. Homebase integration remains a later step.
