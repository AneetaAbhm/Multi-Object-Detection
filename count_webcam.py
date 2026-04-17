import sys
import time
from pathlib import Path
from collections import Counter

import cv2
from ultralytics import YOLO

# ====================== CONFIGURATION ======================
MODEL_PATH  = r"yolo_trained_model\best.pt"   # update path after training
CONFIDENCE  = 0.5                              # detection threshold
RESOLUTION  = (1280, 720)                      # webcam resolution
CAMERA_IDX  = 0                                # 0 = default webcam
# ===========================================================


def load_model(model_path: str) -> YOLO:
    path = Path(model_path)
    if not path.exists():
        print(f"Model not found: {path}")
        print("Update MODEL_PATH at the top of this file.")
        sys.exit(1)
    model = YOLO(str(path))
    print(f"Model loaded  | Classes: {list(model.names.values())}")
    return model


def open_camera(idx: int, resolution: tuple) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"Cannot open webcam (index {idx})")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    return cap


def draw_panel(frame, class_counts: Counter, labels: dict) -> None:
    """Draw a semi-transparent info panel on the left side of the frame."""
    h, w      = frame.shape[:2]
    panel_w   = 260

    # Semi-transparent black background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Title
    cv2.putText(frame, "YOLO Detection", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

    # Divider line
    cv2.line(frame, (10, 50), (panel_w - 10, 50), (100, 100, 100), 1)

    # Count per detected class
    y = 85
    for cls_id, count in sorted(class_counts.items()):
        class_name = labels[cls_id]
        color      = (0, 255, 0) if "person" in class_name.lower() \
                                 or "cars"   in class_name.lower() \
                     else (255, 220, 80)
        cv2.putText(frame, f"{class_name:<18}: {count}",
                    (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        y += 32

    # Total object count
    total = sum(class_counts.values())
    cv2.putText(frame, f"Total objects : {total}",
                (15, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Quit hint at bottom
    cv2.putText(frame, "Press Q to quit", (15, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)


def run_webcam(model: YOLO) -> None:
    labels = model.names
    cap    = open_camera(CAMERA_IDX, RESOLUTION)

    print("\nWebcam started — press Q to quit\n")

    fps_start = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed — check webcam connection.")
            break

        # Run YOLO detection
        results     = model(frame, verbose=False, conf=CONFIDENCE)
        result      = results[0]

        # Count detections per class
        class_counts = Counter(int(cls) for cls in result.boxes.cls)

        # Draw YOLO bounding boxes (built-in)
        annotated = result.plot()

        # Draw info panel
        draw_panel(annotated, class_counts, labels)

        # FPS counter
        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps        = frame_count / elapsed
            fps_start  = time.time()
            frame_count = 0
        else:
            fps = 0

        if fps > 0:
            cv2.putText(annotated, f"FPS: {fps:.1f}", (15, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        cv2.imshow("Urban Object Detection — Webcam", annotated)

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")


if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    run_webcam(model)