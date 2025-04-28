import cv2
import depthai as dai
from ultralytics import YOLO

# Initialize YOLO
model = YOLO("yolov8n.pt")  # Replace with yolov11n.pt when available

# DepthAI Pipeline
pipeline = dai.Pipeline()

# RGB Camera
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)

# Stereo Depth
left = pipeline.create(dai.node.MonoCamera)
right = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
left.out.link(stereo.left)
right.out.link(stereo.right)

# Outputs
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# Device Connection
with dai.Device(pipeline) as device:
    rgb_queue = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    depth_queue = device.getOutputQueue("depth", maxSize=4, blocking=False)

    while True:
        rgb_frame = rgb_queue.get().getCvFrame()
        depth_frame = depth_queue.get().getFrame()

        # YOLO Detection
        results = model(rgb_frame)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            # Distance Calculation
            depth_mm = depth_frame[center_y, center_x]
            distance_m = depth_mm / 1000  # Convert to meters

            # Draw Results
            cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(rgb_frame, f"{distance_m:.2f}m", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Object Detection + Depth", rgb_frame)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
