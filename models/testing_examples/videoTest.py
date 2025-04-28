from ultralytics import YOLO

# Load a pretrained Yolo v8n model
model = YOLO('yolo11n.pt')

# Run inference on the source
# Set source to '0' for live cam
results = model.track(source='testVideo2.mp4',
                      show=True,
                      save=True,
                      tracker='bytetrack.yaml')
