"""
Real-Time Object Detection and Tracking
----------------------------------------
Detector : YOLOv8 (ultralytics)
Tracker  : DeepSORT (deep-sort-realtime)

Usage:
    python object_tracking.py --source path/to/video.mp4
    python object_tracking.py --source 0                # use webcam
    python object_tracking.py --source video.mp4 --save output.mp4

Install dependencies first:
    pip install ultralytics deep-sort-realtime opencv-python
"""

import argparse
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 + DeepSORT object detection and tracking")
    parser.add_argument("--source", type=str, default="0",
                         help="Path to video file, or '0' for webcam (default: 0)")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                         help="YOLOv8 model weights (default: yolov8n.pt, auto-downloads)")
    parser.add_argument("--conf", type=float, default=0.4,
                         help="Detection confidence threshold (default: 0.4)")
    parser.add_argument("--save", type=str, default=None,
                         help="Optional path to save annotated output video")
    parser.add_argument("--no-display", action="store_true",
                         help="Disable live display window (useful for headless runs)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Interpret "0" as webcam index, otherwise treat as file path
    source = 0 if args.source == "0" else args.source

    # Load detector
    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)

    # Load tracker
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))
        print(f"Saving annotated output to: {args.save}")

    frame_idx = 0
    print("Starting detection + tracking. Press 'q' to quit (if display enabled).")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # --- 1. Run YOLOv8 detection on the frame ---
        results = model(frame, conf=args.conf, verbose=False)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            w, h = x2 - x1, y2 - y1
            # DeepSORT expects [left, top, w, h], confidence, class label
            detections.append(([x1, y1, w, h], conf, label))

        # --- 2. Update tracker with this frame's detections ---
        tracks = tracker.update_tracks(detections, frame=frame)

        # --- 3. Draw bounding boxes + labels + tracking IDs ---
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            label = track.get_det_class() or "object"

            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            text = f"ID {track_id} | {label}"
            cv2.putText(frame, text, (l, max(t - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- 4. Display / save output ---
        if not args.no_display:
            cv2.imshow("Object Detection and Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if writer is not None:
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Done. Processed {frame_idx} frames.")


if __name__ == "__main__":
    main()
