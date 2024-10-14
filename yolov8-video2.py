import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os

RECT_WIDTH = 10
ALPHA = 0.5
RADIUS_FACTOR = 1.25
GAMMA = 0.7
MODEL_PATH = 'yolov8n-pose.pt'
CONFIDENCE_THRESHOLD = 0.5
PIXEL_SKIP = 30
YOLO_IMAGE_SIZE = 1280

def extract_pixels(image, start_point, end_point):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    vector = np.array(end_point) - np.array(start_point)
    length = int(np.linalg.norm(vector))
    angle = np.arctan2(vector[1], vector[0])
    center = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
    rect = (center, (length, RECT_WIDTH), np.degrees(angle))
    cv2.fillPoly(mask, [cv2.boxPoints(rect).astype(int)], 255)
    
    t = np.linspace(0, 1, length)
    points = np.column_stack((
        start_point[0] + t * vector[0],
        start_point[1] + t * vector[1]
    )).astype(int)
    
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
    results = model(frame, imgsz=YOLO_IMAGE_SIZE)

    kinesphere_overlay = np.zeros_like(frame, dtype=np.float32)

    for result in results:
        if result.keypoints is None or len(result.keypoints) == 0:
            continue
        
        for person_keypoints, person_confidences in zip(result.keypoints.xy, result.keypoints.conf):
            if person_keypoints.shape[0] < 17:
                continue

            left_gradient = process_arm(frame, person_keypoints, person_confidences, 5, 7, 9)
            right_gradient = process_arm(frame, person_keypoints, person_confidences, 6, 8, 10)

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

def process_video(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = f"{base_name}_kinespheres.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    bitrate = 20000000

    model = YOLO(MODEL_PATH)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processing frame {frame_count}/{total_frames}")

        try:
            processed_frame = process_frame(frame, model)
            out.write(processed_frame)
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            out.write(frame)

    cap.release()
    out.release()

    temp_output = f"{base_name}_temp.mp4"
    os.rename(output_path, temp_output)
    os.system(f"ffmpeg -i {temp_output} -b:v {bitrate} -maxrate {bitrate} -bufsize {bitrate//2} {output_path}")
    os.remove(temp_output)

    print(f"Processing complete. Output saved as {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video file with kinesphere visualization.")
    parser.add_argument("input_video", help="Path to the input video file")
    args = parser.parse_args()

    process_video(args.input_video)