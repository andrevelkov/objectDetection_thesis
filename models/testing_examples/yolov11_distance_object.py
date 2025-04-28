from ultralytics import YOLO
import cv2
import depthai as dai

# model = YOLO('yolo11m.pt')
# results = model.track(source="0", save=True, show=True)
# print(results)

# Load YOLO model
model = YOLO("yolo11s.pt")  # or yolov5, yolov8, etc.

# DepthAI pipeline setup
pipeline = dai.Pipeline()

# Configure RGB camera (match depth resolution)
cam_rgb = pipeline.create(dai.node.ColorCamera)
# cam_rgb.setPreviewSize(1080, 920)
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)

# Configure left + right (stereo) cameras for depth
left = pipeline.create(dai.node.MonoCamera)
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

right = pipeline.create(dai.node.MonoCamera)
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

# Stereo depth setup
stereo = pipeline.create(dai.node.StereoDepth)
# Higher values = more strict filtering
# low values more depth points but may include noisy/incorrect ones
# Controls quality vs quantity of depth data, 200 strict but good for accuracy.
stereo.initialConfig.setConfidenceThreshold(100)
left.out.link(stereo.left)
right.out.link(stereo.right)

# Depth Pro Enhancements
# stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)  # Smoother depth
# stereo.setLeftRightCheck(True)  # Better occlusion handling
# stereo.setExtendedDisparity(True)  # Closer minimum distance
# stereo.setSubpixel(True)  # Finer depth precision
# stereo.setDepthAlign(dai.CameraBoardSocket.RGB)  # Align depth to RGB

# Outputs
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# Connect to device
with dai.Device(pipeline) as device:
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    depth_queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    while True:
        rgb_frame = rgb_queue.get().getCvFrame()
        depth_frame = depth_queue.get().getFrame()

        # Run YOLO detection
        results = model(rgb_frame)
        detections = results[0].boxes

        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label = model.names[class_id]

            # Get depth at the center of the bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            depth_mm = depth_frame[center_y, center_x]  # Depth in mm

            # Draw bounding box + distance
            cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                rgb_frame,
                f"{label}: {depth_mm/1000:.2f}m",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        cv2.imshow("RGB + Depth", rgb_frame)
        if cv2.waitKey(1) == ord("q"):
            break
