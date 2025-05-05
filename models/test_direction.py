import cv2
import depthai as dai
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import json
import time

# Kind of working -> tracking object direction

# Load YOLO model
model = YOLO("yolo11m.pt")

# DepthAI pipeline setup
pipeline = dai.Pipeline()

# Configure RGB camera
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(1280, 720)
cam_rgb.setInterleaved(False)

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

# Track history storage for tracking objects movment history
# ex; {
#     1: [(320, 240), (322, 241), (325, 243)],  # Tracked object 1's movement
#     2: [(100, 150), (105, 152), (110, 155)],  # Tracked object 2's movement
# }
track_history = defaultdict(lambda: [])  # {track_id: [(x_center, y_center), ...]}

# Add this near your other initializations
all_frames_data = []
frame_data = {
    "timestamp": None,
    "objects": []
}

# Inside your main loop (before processing detections)
frame_data["timestamp"] = time.time()
frame_data["objects"] = []  # Clear previous frame data

with dai.Device(pipeline) as device:
    rgb_queue = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    spatial_queue = device.getOutputQueue("spatial", maxSize=4, blocking=False)
    spatial_cfg_queue = device.getInputQueue("spatial_cfg")

    while True:
        rgb_frame = rgb_queue.get().getCvFrame()
        results = model.track(rgb_frame, persist=True, iou=0.35, conf=0.25, tracker="bytetrack.yaml", verbose=False)
        detections = results[0].boxes

        # Get tracking IDs (if available)
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().tolist()
        else:
            track_ids = []

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

            topLeft = dai.Point2f(x1/1280, y1/720)
            bottomRight = dai.Point2f(x2/1280, y2/720)

            roi_data = dai.SpatialLocationCalculatorConfigData()
            roi_data.roi = dai.Rect(topLeft, bottomRight)
            roi_datas.append(roi_data)

            cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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

                    cv2.putText(rgb_frame, f"{label}: {distance_m:.2f}m", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Create object data dictionary 
                    obj_data = {
                        "name": label,
                        "ID": int(track_ids[i]) if i < len(track_ids) else -1,
                        "Conf": round(float(box.conf[0]), 2),
                        "Distance": round(distance_m, 2),
                        "Position": {
                            "x": float((x1 + x2) / 2),
                            "y": float((y1 + y2) / 2),
                            "z": float(z)
                        },
                        "Prediction": None
                    }

                # Draw direction trails (for tracked objects)
                for track_id, track in track_history.items():
                    if len(track) > 1:  # Need at least 2 points to estimate direction
                        points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))  # Convert points to numpy array

                        # cv2.polylines(rgb_frame, [points], isClosed=False, color=(0, 255, 255), thickness=2)  # Draw the trail

                        # check if object has moved , (maybe to reduce errors in slight movement)
                        if len(track) >= 3:
                            # calc total movment over last 3 pos
                            movement = np.sqrt((track[-1][0] - track[-3][0])**2 + (track[-1][1] - track[-3][1])**2)

                            if movement > 50:  # if movment above 10 pixels
                                # Calculate direction vector (latest - previous) Draw Arrow direction
                                dx = track[-1][0] - track[-2][0]
                                dy = track[-1][1] - track[-2][1]
                                direction = np.arctan2(dy, dx)  # Angle in radians
                                # Draw arrow for direction
                                start_point = (int(track[-1][0]), int(track[-1][1]))
                                end_point = (int(track[-1][0] + 25 * np.cos(direction)),
                                             int(track[-1][1] + 25 * np.sin(direction)))
                                cv2.arrowedLine(rgb_frame, start_point, end_point, (255, 0, 0), 3)

                                # PREDICTION #
                                # Calculate average of last 3 positions
                                avg_x = (track[-1][0] + track[-2][0] + track[-3][0]) / 3
                                avg_y = (track[-1][1] + track[-2][1] + track[-3][1]) / 3
                                # Calculate movement vector from average to current position
                                dx = track[-1][0] - avg_x
                                dy = track[-1][1] - avg_y
                                # Predict next position by extending this vector
                                pred_x = track[-1][0] + dx
                                pred_y = track[-1][1] + dy

                                obj_data["Prediction"] = {
                                    "x": float(pred_x),
                                    "y": float(pred_y)
                                }

                                # Draw prediction
                                cv2.circle(rgb_frame, (int(pred_x), int(pred_y)), 8, (255, 0, 0), -1)

                frame_data["objects"].append(obj_data)

            except Exception as e:
                print(f"{e}")

            # Add this frame's data to the collection
            all_frames_data.append(frame_data)

            # After processing all detections
            # Print the JSON for this frame
            print(json.dumps(frame_data, indent=2))

            # Save to a file (appending each frame)
            # with open("tracking_data.json", "a") as f:
            #     f.write(json.dumps(frame_data) + "\n")  # Newline separated JSON

        cv2.imshow("RGB + Tracking + Direction", rgb_frame)
        if cv2.waitKey(1) == ord("q"):
            break

# After the main loop ends, save all data to a single JSON file
with open("structured_tracking_data.json", "w") as f:
    json.dump(all_frames_data, f, indent=2)

cv2.destroyAllWindows()
