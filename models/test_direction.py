import cv2
import depthai as dai
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

# Load YOLO model
model = YOLO("yolo11m.pt")

# Kind of working -> tracking object direction

# DepthAI pipeline setup
pipeline = dai.Pipeline()

# Configure RGB camera
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(1280, 720)
cam_rgb.setInterleaved(False)
cam_rgb.setFps(30)

# Configure stereo depth (same as before)
left = pipeline.create(dai.node.MonoCamera)
right = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(False)  # Limits depth range to ~0.3-10m (better close-range accuracy).	Disabling extends range but reduces precision.
stereo.setSubpixel(False)  # Disables subpixel refinement (faster but less accurate depth edges).	Enabling improves depth resolution at a performance cost.
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # Aligns depth map to RGB camera perspective.

left.out.link(stereo.left)
right.out.link(stereo.right)

# Spatial Location Calculator
spatialCalc = pipeline.create(dai.node.SpatialLocationCalculator)
stereo.depth.link(spatialCalc.inputDepth)

spatial_cfg_in = pipeline.create(dai.node.XLinkIn)
spatial_cfg_in.setStreamName("spatial_cfg")
spatial_cfg_in.out.link(spatialCalc.inputConfig)

# Outputs
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_spatial = pipeline.create(dai.node.XLinkOut)
xout_spatial.setStreamName("spatial")
spatialCalc.out.link(xout_spatial.input)

# Track history storage
track_history = defaultdict(lambda: [])  # {track_id: [(x_center, y_center), ...]}

with dai.Device(pipeline) as device:
    rgb_queue = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    spatial_queue = device.getOutputQueue("spatial", maxSize=4, blocking=False)
    spatial_cfg_queue = device.getInputQueue("spatial_cfg")

    while True:
        rgb_frame = rgb_queue.get().getCvFrame()
        results = model.track(rgb_frame, persist=True, iou=0.35, conf=0.25, tracker="bytetrack.yaml")
        detections = results[0].boxes

        # Get tracking IDs (if available)
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().tolist()
        else:
            track_ids = []

        # Prepare ROIs for spatial calculator
        cfg = dai.SpatialLocationCalculatorConfig()
        roi_datas = []

        for i, box in enumerate(detections):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label = model.names[class_id]

            # Calculate center of bounding box
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            # Store track history (if tracking ID exists)
            if i < len(track_ids):
                track_id = track_ids[i]
                track_history[track_id].append((x_center, y_center))
                if len(track_history[track_id]) > 30:  # Keep last 30 points
                    track_history[track_id].pop(0)

            # Define ROI for depth calculation
            topLeft = dai.Point2f(x1/1280, y1/720)
            bottomRight = dai.Point2f(x2/1280, y2/720)
            roi_data = dai.SpatialLocationCalculatorConfigData()
            roi_data.roi = dai.Rect(topLeft, bottomRight)
            roi_datas.append(roi_data)

            # Draw bounding box
            cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add ROIs to config
        if roi_datas:
            cfg.setROIs(roi_datas)
            spatial_cfg_queue.send(cfg)

            try:
                spatial_data = spatial_queue.get().getSpatialLocations()
                for i, data in enumerate(spatial_data):
                    if i >= len(detections):
                        continue

                    z = data.spatialCoordinates.z
                    distance_m = z / 1000
                    box = detections[i]
                    x1, y1, _, _ = map(int, box.xyxy[0])
                    label = model.names[int(box.cls[0])]

                    # Display distance
                    cv2.putText(rgb_frame, f"{label}: {distance_m:.2f}m", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            except Exception as e:
                print(f"Depth read failed: {e}")

        # Draw direction trails (for tracked objects)
        for track_id, track in track_history.items():
            if len(track) > 1:  # Need at least 2 points to estimate direction
                # Convert points to numpy array
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                # Draw the trail
                cv2.polylines(rgb_frame, [points], isClosed=False, color=(0, 255, 255), thickness=2)

                # Calculate direction vector (latest - previous)
                dx = track[-1][0] - track[-2][0]
                dy = track[-1][1] - track[-2][1]
                direction = np.arctan2(dy, dx)  # Angle in radians

                # Draw arrow for direction
                start_point = (int(track[-1][0]), int(track[-1][1]))
                end_point = (int(track[-1][0] + 50 * np.cos(direction)),
                             int(track[-1][1] + 50 * np.sin(direction)))
                cv2.arrowedLine(rgb_frame, start_point, end_point, (255, 0, 0), 2)

        cv2.imshow("RGB + Tracking + Direction", rgb_frame)
        if cv2.waitKey(1) == ord("q"):
            break

cv2.destroyAllWindows()
