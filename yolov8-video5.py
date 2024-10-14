import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os
import shutil
from collections import defaultdict


RECT_WIDTH = 10
ALPHA = 0.7
RADIUS_FACTOR = 1.25
GAMMA = 0.7
POSE_MODEL_PATH = 'yolov8n-pose.pt'
SEG_MODEL_PATH = 'yolov8s-seg.pt'
CONFIDENCE_THRESHOLD = 0.55
PIXEL_SKIP = 30
YOLO_IMAGE_SIZE = 1280
BLUR_RADIUS = 37
ENABLE_ROSE_OVERLAY = True  # Set this to False to disable rose overlay
ROSE_COLOR = np.array([193, 182, 255])  # BGR format
COLOR_SMOOTHING_FRAMES = 5


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

def process_arm(frame, keypoints, confidences, shoulder_idx, elbow_idx, wrist_idx, color_history):
    if (confidences[shoulder_idx] < CONFIDENCE_THRESHOLD or
        confidences[elbow_idx] < CONFIDENCE_THRESHOLD or
        confidences[wrist_idx] < CONFIDENCE_THRESHOLD):
        return None, color_history

    shoulder = tuple(map(int, keypoints[shoulder_idx]))
    elbow = tuple(map(int, keypoints[elbow_idx]))
    wrist = tuple(map(int, keypoints[wrist_idx]))

    pixels_se, len_se = extract_pixels(frame, shoulder, elbow)
    pixels_ew, len_ew = extract_pixels(frame, elbow, wrist)
    
    if len(pixels_se) == 0 or len(pixels_ew) == 0:
        return None, color_history

    combined_pixels = np.concatenate((pixels_se, pixels_ew))
    radius = int((len_se + len_ew) * RADIUS_FACTOR)
    
    # Smooth the colors over the last three frames
    if len(color_history) == COLOR_SMOOTHING_FRAMES:
        color_history.pop(0)
    
    color_history.append(combined_pixels)
    
    # Check if all color history arrays have the same length
    if all(len(colors) == len(color_history[0]) for colors in color_history):
        smoothed_colors = np.mean(color_history, axis=0)
    else:
        # If the arrays have different lengths, use the current combined_pixels
        smoothed_colors = combined_pixels
    
    return create_circular_gradient(frame.shape, shoulder, radius, smoothed_colors), color_history

def get_body_mask(frame, seg_model):
    results = seg_model(frame, imgsz=YOLO_IMAGE_SIZE)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    for result in results:
        if result.masks is not None:
            for segment in result.masks.xy:
                cv2.fillPoly(mask, [segment.astype(int)], 255)
    
    # Apply blur to the mask
    blurred_mask = cv2.GaussianBlur(mask, (BLUR_RADIUS, BLUR_RADIUS), 0)
    
    # Convert to float32 for smoother edges
    return blurred_mask.astype(np.float32) / 255.0

def process_frame(frame, pose_model, seg_model, color_history):
    results = pose_model(frame, imgsz=YOLO_IMAGE_SIZE)

    kinesphere_overlay = np.zeros_like(frame, dtype=np.float32)
    kinesphere_masks = []

    for result in results:
        if result.keypoints is None or len(result.keypoints) == 0:
            continue
        
        for person_id, (person_keypoints, person_confidences) in enumerate(zip(result.keypoints.xy, result.keypoints.conf)):
            if person_keypoints.shape[0] < 17:
                continue

            person_mask = np.zeros(frame.shape[:2], dtype=bool)

            left_gradient, color_history[(person_id, 'left')] = process_arm(frame, person_keypoints, person_confidences, 5, 7, 9, color_history.get((person_id, 'left'), []))
            right_gradient, color_history[(person_id, 'right')] = process_arm(frame, person_keypoints, person_confidences, 6, 8, 10, color_history.get((person_id, 'right'), []))

            if left_gradient is not None:
                kinesphere_overlay = cv2.add(kinesphere_overlay, left_gradient * 0.5)
                person_mask |= (left_gradient.sum(axis=2) > 0)
            if right_gradient is not None:
                kinesphere_overlay = cv2.add(kinesphere_overlay, right_gradient * 0.5)
                person_mask |= (right_gradient.sum(axis=2) > 0)

            if np.any(person_mask):
                kinesphere_masks.append(person_mask)

    body_mask = get_body_mask(frame, seg_model)
    
    # Ensure body_mask is 2D
    if body_mask.ndim == 3:
        body_mask = body_mask[:,:,0]
    
    if np.max(kinesphere_overlay) > 0:
        kinesphere_only = cv2.normalize(kinesphere_overlay, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        kinesphere_only = cv2.pow(kinesphere_only / 255.0, GAMMA) * 255.0
        kinesphere_only = kinesphere_only.astype(np.uint8)

        # Convert to HSV and increase saturation
        kinesphere_hsv = cv2.cvtColor(kinesphere_only, cv2.COLOR_BGR2HSV)
        kinesphere_hsv[:,:,1] = np.clip(kinesphere_hsv[:,:,1] * 1.2, 0, 255).astype(np.uint8)
        kinesphere_only = cv2.cvtColor(kinesphere_hsv, cv2.COLOR_HSV2BGR)

        # Create overlap mask if rose overlay is enabled
        if ENABLE_ROSE_OVERLAY:
            overlap_mask = np.zeros(frame.shape[:2], dtype=bool)
            for i in range(len(kinesphere_masks)):
                for j in range(i+1, len(kinesphere_masks)):
                    overlap_mask |= (kinesphere_masks[i] & kinesphere_masks[j])

        # Apply kinesphere mask
        combined_mask = np.logical_or.reduce(kinesphere_masks)
        kinesphere_only = kinesphere_only * combined_mask[:,:,np.newaxis]

        # Apply rose color to overlapping regions if enabled
        if ENABLE_ROSE_OVERLAY:
            rose_overlay = np.full_like(frame, ROSE_COLOR)
            kinesphere_only = np.where(overlap_mask[:,:,np.newaxis], rose_overlay, kinesphere_only)

        # Create an alpha mask for kinesphere
        alpha_mask = combined_mask.astype(np.float32) * ALPHA

        # Blend the frame with the kinesphere
        blended_frame = frame.astype(np.float32) * (1 - alpha_mask[:,:,np.newaxis]) + \
                        kinesphere_only.astype(np.float32) * alpha_mask[:,:,np.newaxis]
        
        # Apply body mask with 40% opacity
        body_mask_3d = np.repeat(body_mask[:,:,np.newaxis], 3, axis=2)
        body_alpha = 0.45  # 40% opacity for body mask
        final_frame = blended_frame * (1 - body_mask_3d * body_alpha) + frame.astype(np.float32) * (body_mask_3d * body_alpha)
        final_frame = final_frame.astype(np.uint8)
    else:
        final_frame = frame

    return final_frame, color_history

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
    output_path = f"{base_name}_kinespheres_segmented.mp4"

    # Create kinesphere_frames directory if it doesn't exist
    frames_dir = "kinesphere_frames"
    os.makedirs(frames_dir, exist_ok=True)

    pose_model = YOLO(POSE_MODEL_PATH)
    seg_model = YOLO(SEG_MODEL_PATH)

    frame_count = 0
    color_history = defaultdict(list)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processing frame {frame_count}/{total_frames}")

        try:
            processed_frame, color_history = process_frame(frame, pose_model, seg_model, color_history)
            # Save frame as high-quality JPEG
            frame_path = os.path.join(frames_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            # Save original frame if processing fails
            frame_path = os.path.join(frames_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    cap.release()

    # Create video from frames
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith(".jpg")]
    frame_files.sort()

    if frame_files:
        first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
        height, width = first_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame_file in frame_files:
            frame = cv2.imread(os.path.join(frames_dir, frame_file))
            out.write(frame)

        out.release()

        # Apply bitrate adjustment
        bitrate = 20000000
        temp_output = f"{base_name}_temp.mp4"
        os.rename(output_path, temp_output)
        os.system(f"ffmpeg -i {temp_output} -b:v {bitrate} -maxrate {bitrate} -bufsize {bitrate//2} {output_path}")
        os.remove(temp_output)

        print(f"Processing complete. Output saved as {output_path}")

        # Clean up frames directory
        shutil.rmtree(frames_dir)
        print(f"Cleaned up temporary frames in {frames_dir}")
    else:
        print("No frames were processed. Check for errors during frame processing.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video file with kinesphere and segmentation.")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("--disable-rose", action="store_true", help="Disable rose overlay for overlapping kinespheres")
    args = parser.parse_args()

    # Set the ENABLE_ROSE_OVERLAY based on the command-line argument
    ENABLE_ROSE_OVERLAY = not args.disable_rose

    process_video(args.input_video)