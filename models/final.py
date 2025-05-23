import cv2
import depthai as dai
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import json
import time

# Load YOLO model
model = YOLO("yolo11s.pt").cuda()  # GPU
# model = YOLO("yolo11m.pt")  # non GPU

# DepthAI pipeline
pipeline = dai.Pipeline()
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)

left = pipeline.create(dai.node.MonoCamera)
right = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.ROBOTICS)
left.out.link(stereo.left)
right.out.link(stereo.right)

spatialCalc = pipeline.create(dai.node.SpatialLocationCalculator)
stereo.depth.link(spatialCalc.inputDepth)
spatial_cfg_in = pipeline.create(dai.node.XLinkIn)
spatial_cfg_in.setStreamName("spatial_cfg")
spatial_cfg_in.out.link(spatialCalc.inputConfig)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_spatial = pipeline.create(dai.node.XLinkOut)
xout_spatial.setStreamName("spatial")
spatialCalc.out.link(xout_spatial.input)

track_history = defaultdict(lambda: [])
all_frames_data = []
fps_counter, fps = 0, 0
start_time = time.time()


def movement(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def predict_next(track, steps=5):
    if len(track) < 3:
        return track[-1]
    avg = np.mean(track[-3:], axis=0)
    delta = np.array(track[-1]) - avg
    return track[-1] + delta * steps


with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS) as device:
    rgb_queue = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    spatial_queue = device.getOutputQueue("spatial", maxSize=4, blocking=False)
    spatial_cfg_queue = device.getInputQueue("spatial_cfg")

    while True:
        fps_counter += 1
        if (time.time() - start_time) > 1.0:
            fps = fps_counter
            fps_counter = 0
            start_time = time.time()

        rgb_frame = rgb_queue.get().getCvFrame()
        results = model.track(rgb_frame, persist=True, iou=0.4, conf=0.45, tracker="bytetrack.yaml", verbose=False)
        detections = results[0].boxes

        cv2.putText(rgb_frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        track_ids = results[0].boxes.id.int().tolist() if results[0].boxes.id is not None else []
        cfg = dai.SpatialLocationCalculatorConfig()
        roi_datas = []

        frame_data = {
            "timestamp": time.time(),
            "objects": []
        }

        for i, box in enumerate(detections):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            topLeft = dai.Point2f(x1 / 640, y1 / 480)
            bottomRight = dai.Point2f(x2 / 640, y2 / 480)
            roi_data = dai.SpatialLocationCalculatorConfigData()
            roi_data.roi = dai.Rect(topLeft, bottomRight)
            roi_datas.append(roi_data)
            cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        if roi_datas:
            cfg.setROIs(roi_datas)
            spatial_cfg_queue.send(cfg)

            try:
                spatial_data = spatial_queue.get().getSpatialLocations()
                for i, data in enumerate(spatial_data):
                    if i >= len(detections):
                        continue

                    box = detections[i]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
                    z = data.spatialCoordinates.z
                    distance_m = z / 1000

                    class_id = int(box.cls[0])
                    label = model.names[class_id]
                    track_id = track_ids[i] if i < len(track_ids) else -1

                    track_history[track_id].append((x_center, y_center, z))
                    if len(track_history[track_id]) > 20:
                        track_history[track_id].pop(0)

                    obj_data = {
                        "name": label,
                        "ID": track_id,
                        "Conf": round(float(box.conf[0]), 2),
                        "Distance": round(distance_m, 2),
                        "Position": {
                            "x": float(x_center),
                            "y": float(y_center),
                            "z": float(z)
                        },
                        "Prediction": None
                    }

                    text = f"{label} ID:{track_id} {distance_m:.2f}m"
                    cv2.putText(rgb_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    track = track_history[track_id]
                    xy_track = [(x, y) for x, y, _ in track]

                    if len(xy_track) >= 3 and movement(xy_track[-3], xy_track[-1]) > 10:
                        dir_vec = np.array(xy_track[-1]) - np.array(xy_track[-2])
                        angle = np.arctan2(dir_vec[1], dir_vec[0])
                        arrow_end = np.array(xy_track[-1]) + 25 * np.array([np.cos(angle), np.sin(angle)])
                        cv2.arrowedLine(rgb_frame, tuple(map(int, xy_track[-1])), tuple(arrow_end.astype(int)), (255, 0, 0), 2)

                        pred = predict_next(track)
                        obj_data["Prediction"] = {
                            "x": float(pred[0]),
                            "y": float(pred[1]),
                            "z": float(pred[2])
                        }
                        cv2.circle(rgb_frame, tuple(map(int, pred[:2])), 6, (255, 0, 0), -1)

                    frame_data["objects"].append(obj_data)

            except Exception as e:
                print(f"{e}")

            all_frames_data.append(frame_data)
            if fps_counter % 10 == 0:
                print(json.dumps(frame_data, indent=2))

        cv2.imshow("RGB + Tracking + Direction", rgb_frame)
        if cv2.waitKey(1) == ord("q"):
            break

with open("tracking_data.json", "w") as f:
    json.dump(all_frames_data, f, indent=2)

cv2.destroyAllWindows()
