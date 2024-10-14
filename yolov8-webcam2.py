import cv2
import numpy as np
from ultralytics import YOLO

RECT_WIDTH = 10  # Increased from 1 to 10
ALPHA = 0.5
RADIUS_FACTOR = 1.2
GAMMA = 0.7
MODEL_PATH = 'yolov8n-pose.pt'
CONFIDENCE_THRESHOLD = 0.5
PIXEL_SKIP = 30  # New constant to define pixel sampling rate

def extract_pixels(image, start_point, end_point):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    vector = np.array(end_point) - np.array(start_point)
    length = int(np.linalg.norm(vector))
    angle = np.arctan2(vector[1], vector[0])
    center = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
    rect = (center, (length, RECT_WIDTH), np.degrees(angle))
    cv2.fillPoly(mask, [cv2.boxPoints(rect).astype(int)], 255)
    
    # Create a list of points along the line
    t = np.linspace(0, 1, length)
    points = np.column_stack((
        start_point[0] + t * vector[0],
        start_point[1] + t * vector[1]
    )).astype(int)
    
    # Sample pixels along the line
    pixels = image[points[:, 1], points[:, 0]][::PIXEL_SKIP]
    
    return pixels, length

def create_circular_gradient(shape, center, radius, pixels):
    if len(pixels) == 0:
        return np.zeros(shape, dtype=np.float32)
    y, x = np.ogrid[:shape[0], :shape[1]]
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask = dist <= radius
    normalized_dist = (dist[mask] / radius * (len(pixels) - 1)).astype(int)
    gradient = np.zeros(shape, dtype=np.float32)
    gradient[mask] = pixels[normalized_dist]
    return gradient

def process_arm(frame, keypoints, confidences, shoulder_idx, elbow_idx, wrist_idx):
    if (confidences[shoulder_idx] < CONFIDENCE_THRESHOLD or
        confidences[elbow_idx] < CONFIDENCE_THRESHOLD or
        confidences[wrist_idx] < CONFIDENCE_THRESHOLD):
        return None

    shoulder = tuple(map(int, keypoints[shoulder_idx]))
    elbow = tuple(map(int, keypoints[elbow_idx]))
    wrist = tuple(map(int, keypoints[wrist_idx]))

    pixels_se, len_se = extract_pixels(frame, shoulder, elbow)
    pixels_ew, len_ew = extract_pixels(frame, elbow, wrist)
    
    if len(pixels_se) == 0 or len(pixels_ew) == 0:
        return None

    combined_pixels = np.concatenate((pixels_se, pixels_ew))
    radius = int((len_se + len_ew) * RADIUS_FACTOR)
    return create_circular_gradient(frame.shape, shoulder, radius, combined_pixels)

def process_frame(frame, model):
    results = model(frame)

    kinesphere_overlay = np.zeros_like(frame, dtype=np.float32)

    for result in results:
        if result.keypoints is None or len(result.keypoints) == 0:
            continue
        
        keypoints = result.keypoints.xy[0]
        confidences = result.keypoints.conf[0]

        if keypoints.shape[0] < 17:
            continue

        left_gradient = process_arm(frame, keypoints, confidences, 5, 7, 9)
        right_gradient = process_arm(frame, keypoints, confidences, 6, 8, 10)

        if left_gradient is not None:
            kinesphere_overlay = cv2.add(kinesphere_overlay, left_gradient * 0.5)
        if right_gradient is not None:
            kinesphere_overlay = cv2.add(kinesphere_overlay, right_gradient * 0.5)

    if np.max(kinesphere_overlay) > 0:
        kinesphere_only = cv2.normalize(kinesphere_overlay, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        kinesphere_only = cv2.pow(kinesphere_only / 255.0, GAMMA) * 255.0
        kinesphere_only = kinesphere_only.astype(np.uint8)

        # Convert to HSV and increase saturation
        kinesphere_hsv = cv2.cvtColor(kinesphere_only, cv2.COLOR_BGR2HSV)
        kinesphere_hsv[:,:,1] = np.clip(kinesphere_hsv[:,:,1] * 1.2, 0, 255).astype(np.uint8)
        kinesphere_only = cv2.cvtColor(kinesphere_hsv, cv2.COLOR_HSV2BGR)

        alpha_channel = np.where(kinesphere_only[..., 0] > 0, 255, 0).astype(np.uint8)
        alpha_mask = alpha_channel.astype(bool)
        blended_frame = cv2.addWeighted(frame, 1 - ALPHA, kinesphere_only, ALPHA, 0)
        final_frame = frame.copy()
        final_frame[alpha_mask] = blended_frame[alpha_mask]
    else:
        final_frame = frame

    return final_frame

def process_webcam():
    camera_index = 1
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    model = YOLO(MODEL_PATH)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processing frame {frame_count}")

        try:
            processed_frame = process_frame(frame, model)
            cv2.imshow('Kinesphere Visualization', processed_frame)
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or Escape key
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_webcam()