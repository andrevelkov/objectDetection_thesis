import time
import cv2
import depthai as dai
from ultralytics import YOLO

# Load YOLO model for object detection
# YOLO models -> yolo11n = nano (smallest) , yolo11n.pt yolo11s.pt yolo11m.pt yolo11l.pt yolo11x.pt
model = YOLO("yolo11n.pt").cuda()

# DepthAI pipeline setup
pipeline = dai.Pipeline()

# Configure RGB camera
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
# cam_rgb.setFps(30)  # Set camera FPS

# Stereo depth setup (left/right mono cameras + depth processing)
left = pipeline.create(dai.node.MonoCamera)
right = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

# set left and right mono camera res and assign port
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

# Configure stereo depth settings (noise reduction, alignment to RGB)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)  # Reduces noise
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(True)  # Limits depth range to ~0.3-8m (better close-range acc). Disabling extends range but reduces precision.
stereo.setSubpixel(True)  # Disables subpixel refinement (faster but less accurate depth edges).Enabling improves depth resolution at a performance cost.
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # Aligns depth map to RGB camera perspective.

# TODO: test long  vs short 
# Two stereo configurations, close vs longer range
# close_range_config = {
#   "subpixel": False,           # Faster, better for near
#   "extendedDisparity": True,   # Helps close objects (<1m)
#   "confidenceThreshold": 100,  # Lower threshold for near
# }

# long_range_config = {
#   "subpixel": True,            # Slower, better for far
#   "extendedDisparity": False,  # Disable for >3m
#   "confidenceThreshold": 200,  # Higher threshold to reduce noise
# }

# connect mono cams output to stereo input
left.out.link(stereo.left)
right.out.link(stereo.right)

# Spatial Location Calculator, Spatial calculator for depth measurements in ROIs (region of interest)
spatialCalc = pipeline.create(dai.node.SpatialLocationCalculator)
stereo.depth.link(spatialCalc.inputDepth)

# Configure spatial calculator input
# Creates an input link to send ROI (Region of Interest) data to the depth calculator
spatial_cfg_in = pipeline.create(dai.node.XLinkIn)
spatial_cfg_in.setStreamName("spatial_cfg")
spatial_cfg_in.out.link(spatialCalc.inputConfig)

# Create output streams for RGB and spatial data
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_spatial = pipeline.create(dai.node.XLinkOut)
xout_spatial.setStreamName("spatial")
spatialCalc.out.link(xout_spatial.input)

# Initialize FPS variables
fps_counter = 0
fps = 0
start_time = time.time()

# Connect to device, main processing loop
with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS) as device:
    # Get cams and depth data queues
    rgb_queue = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    spatial_queue = device.getOutputQueue("spatial", maxSize=4, blocking=False)
    spatial_cfg_queue = device.getInputQueue("spatial_cfg")

    while True:
        fps_counter += 1
        if (time.time() - start_time) > 1.0:  # Update every second
            fps = fps_counter
            fps_counter = 0
            start_time = time.time()

        # Get rgb frame and run inference
        rgb_frame = rgb_queue.get().getCvFrame()
        # Add NMS, Non-MAximum Suppression, to YOLO inference (reduces duplicate boxes)
        # results = model(rgb_frame, iou=0.45, conf=0.35)  # Adjust thresholds, intersection over union and confidence
        results = model.track(rgb_frame, persist=True, iou=0.35, conf=0.25, tracker="bytetrack.yaml", verbose=True)  # with tracking
        detections = results[0].boxes

        cv2.putText(rgb_frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(rgb_frame, f"Objects: {len(detections) if 'detections' in locals() else 0}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Prepare ROIs for spatial calculator
        cfg = dai.SpatialLocationCalculatorConfig()
        roi_datas = []

        # for each detection, get box coordinates, calculate ROI for depth, draw box and display distance
        for box in detections:
            # extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # get class id and label
            class_id = int(box.cls[0])
            label = model.names[int(box.cls[0])]

            # Convert pixel coordinates to normalized coordinates (0-1)
            # for DepthAI spatial calculator (matches 1280x720 preview size)
            topLeft = dai.Point2f(x1/1280, y1/720)
            bottomRight = dai.Point2f(x2/1280, y2/720)

            # Create config and add ROI
            roi_data = dai.SpatialLocationCalculatorConfigData()
            roi_data.roi = dai.Rect(topLeft, bottomRight)
            roi_datas.append(roi_data)

            # Draw bounding box (for visualization)
            cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # If any detections, calculate their depth
        if roi_datas:
            # send to spatial calculator
            cfg.setROIs(roi_datas)
            spatial_cfg_queue.send(cfg)

            try:
                # get depth data
                spatial_data = spatial_queue.get().getSpatialLocations()

                # process each depth
                for i, data in enumerate(spatial_data):
                    # Get Z coordinate (depth) in mm and convert to meters
                    z = data.spatialCoordinates.z
                    distance_m = z / 1000  # Convert mm to meters

                    # Get the corresponding detection info
                    box = detections[i]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = model.names[int(box.cls[0])]

                    obj_id = int(box.id[0]) if box.id is not None else -1
                    conf = box.conf[0].item()

                    # Custom print format using YOLO's detected values
                    # print(f"  {label} (ID: {obj_id}, Conf: {conf:.2f}): {distance_m:.2f}m")

                    json_obj = {
                        "name": label,
                        "ID": obj_id,
                        "Conf": round(conf, 2),
                        "Distance": round(distance_m, 2),
                    }

                    print(json_obj)

                    # display label and distance above box
                    cv2.putText(rgb_frame, f"{label}: {distance_m:.2f}m", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,)

            except Exception as e:
                print(f"Depth read failed: {e}")
                # pass  # Skip if no spatial data received

        cv2.imshow("RGB + Spatial Depth", rgb_frame)
        if cv2.waitKey(1) == ord("q"):
            break

cv2.destroyAllWindows()
