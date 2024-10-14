# Kinesphere visualisation
Python scripts to visualize a person's sphere of potential movement

yolov8-video5.py
The script takes a video as input, processes it frame by frame and outputs a video file. It uses two AI models: a pose detection model (YOLO) to identify human body keypoints, and a segmentation model to detect and outline human bodies.
For each person detected in a frame, the script creates a kinesphere - a space of potential movement visualized as a semi-transparent overlay that surrounds the person's shoulders. This is done by:
Extracting color information along the arm (from shoulder to wrist)
Creating a circular gradient based on these colors
Applying this gradient around the person's shoulders.
To prevent jarring color changes between frames, the script implements a color smoothing mechanism. It keeps a history of colors for each person's arms over several frames and averages them. The script uses the segmentation model to create a mask of the human bodies in the frame. This mask is used to ensure that the kinesphere effect doesn't hide the actual bodies. When kinespheres of different people overlap, the script can optionally apply a special red color to these overlapping areas, showing the kinesphere infringement as it happens. 
Assuming you have ultralytics and opencv installed, run the script as python3 yolov8-video2.py yourvideo.mp4

yolov8-video2.py
This is a simpler script, similar to yolov8-video5.py but for performance reasons it does not include masking, smoothing and kinesphere overlap functions.

yolov8-webcam2.py
This is the same as yolov8-video2.py but for using webcam as input, and outputting near-live kinesphere video on the screen.
