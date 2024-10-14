import cv2
import numpy as np
from ultralytics import YOLO
from math import sqrt
from tqdm import tqdm
import sys
import os

class CircleAnnotator:
    def __init__(self, color=(255, 255, 255), thickness=2):
        self.color = color
        self.thickness = thickness

    def annotate(self, frame, detections, historical_overlay):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        red_color = (0, 0, 255)  # Red color in BGR
        alpha = 0.4  # 40% transparency

        for detection in detections:
            x1, y1, x2, y2 = map(int, detection[:4])
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            radius = int(sqrt((x1 - center[0]) ** 2 + (y1 - center[1]) ** 2))

            circle_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.circle(circle_mask, center, radius, 1, thickness=-1)
            mask += circle_mask

        overlap_mask = (mask > 1).astype(np.uint8)
        overlay = np.zeros_like(frame)
        overlay[overlap_mask == 1] = red_color

        cv2.addWeighted(overlay, 0.1, historical_overlay, 0.9, 0, historical_overlay)
        cv2.addWeighted(historical_overlay, alpha, frame, 1 - alpha, 0, frame)

        return frame

# Check if a video file is provided as an argument
if len(sys.argv) < 2:
    print("Please provide a video file as an argument.")
    sys.exit(1)

# Get the input video file name from the first argument
video_path = sys.argv[1]

# Generate the output file name
base_name = os.path.splitext(os.path.basename(video_path))[0]
output_path = f"{base_name}_reds.mp4"

# Load YOLOv8 model
model = YOLO('yolov8s-visdrone-enot.pt')

# Open video file
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Initialize CircleAnnotator and historical overlay
circle_annotator = CircleAnnotator()
historical_overlay = np.zeros((height, width, 3), dtype=np.uint8)

# Define the labels we're interested in
target_labels = ['pedestrian', 'people', 'person']

# Process video frames
for _ in tqdm(range(total_frames)):
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame, imgsz=1280)
    
    # Extract detections for 'pedestrian' and 'people' only
    detections = []
    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls)]
            if label.lower() in target_labels:
                detections.append(box.xyxy[0].cpu().numpy())

    # Annotate frame
    annotated_frame = circle_annotator.annotate(
        frame=frame.copy(),
        detections=detections,
        historical_overlay=historical_overlay
    )

    # Write frame to output video
    out.write(annotated_frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved as: {output_path}")