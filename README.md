Real-Time Object Detection and Tracking

Detects and tracks objects in a video (file or webcam) using YOLOv8 for detection and DeepSORT for multi-object tracking, drawing bounding boxes, class labels, and persistent tracking IDs on each frame.

Features

Real-time video input from a webcam or a video file
Pre-trained YOLOv8 model for object detection
Frame-by-frame processing with bounding box drawing
DeepSORT tracking for stable IDs across frames (handles occlusion better than plain SORT)
Live display window and optional saved output video


Requirements

Python 3.8+
A webcam (optional) or a video file


Installation

bashpip install ultralytics deep-sort-realtime opencv-python

The first run will auto-download the YOLOv8 weights (yolov8n.pt by default, ~6MB).

Usage

Run on a video file:

bashpython object_tracking.py --source my_video.mp4

Run on webcam (default camera):

bashpython object_tracking.py --source 0

Save the annotated output to a new video file:

bashpython object_tracking.py --source my_video.mp4 --save tracked_output.mp4

Run headless (no display window, e.g. on a server):

bashpython object_tracking.py --source my_video.mp4 --save tracked_output.mp4 --no-display

Use a larger/more accurate YOLO model:

bashpython object_tracking.py --source my_video.mp4 --model yolov8s.pt

(Options in increasing size/accuracy: yolov8n.pt < yolov8s.pt < yolov8m.pt < yolov8l.pt < yolov8x.pt)

Adjust detection confidence threshold:

bashpython object_tracking.py --source my_video.mp4 --conf 0.5

Command-line arguments

ArgumentDefaultDescription--source0Video file path, or 0 for webcam--modelyolov8n.ptYOLOv8 weights file--conf0.4Detection confidence threshold (0–1)--saveNonePath to save the annotated output video--no-displayoffDisable the live preview window

Controls


Press q while the display window is focused to stop processing early.


How it works


Each frame is read from the video source.
YOLOv8 detects objects and returns bounding boxes, confidence scores, and class labels.
Detections are passed to DeepSORT, which matches them to existing tracks (or creates new ones) using motion prediction and appearance features.
Confirmed tracks are drawn on the frame with their bounding box, class label, and a persistent track ID.
The frame is shown live and/or written to the output video file.
