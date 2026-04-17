import sys
import time
from pathlib import Path
from collections import Counter

import cv2
from ultralytics import YOLO

# ====================== CONFIGURATION ======================
MODEL_PATH  = r"yolo_trained_model\best.pt"   # update after training
CONFIDENCE  = 0.25                             # detection threshold
CAMERA_IDX  = 0                                # 0 = default webcam
RESOLUTION  = (1280, 720)
# ===========================================================


def load_model(model_path: str) -> YOLO:
    path = Path(model_path)
    if not path.exists():
        print(f"Model not found: {path}")
        print("Update MODEL_PATH at the top of this file.")
        sys.exit(1)
    model = YOLO(str(path))
    return model


def pick_alert_class(labels: dict) -> tuple:
    """Show available classes and let user type which one to watch."""
    print("\nAvailable classes:")
    for idx, name in labels.items():
        print(f"  [{idx}] {name}")

    print("\nEnter the class name or number to watch for alerts:")
    user_input = input("  >>> ").strip()

    # Accept number input
    if user_input.isdigit():
        idx = int(user_input)
        if idx in labels:
            return idx, labels[idx]
        else:
            print(f"Invalid index: {idx}")
            sys.exit(1)

    # Accept name input (case-insensitive)
    name_to_id = {v.lower(): k for k, v in labels.items()}
    if user_input.lower() in name_to_id:
        idx = name_to_id[user_input.lower()]
        return idx, labels[idx]

    print(f"Class '{user_input}' not found.")
    print(f"Available: {list(labels.values())}")
    sys.exit(1)


def run_alert_webcam(model: YOLO, alert_id: int, alert_class: str, conf: float) -> None:
    labels = model.names

    cap = cv2.VideoCapture(CAMERA_IDX)
    if not cap.isOpened():
        print(f"Cannot open webcam (index {CAMERA_IDX})")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

    print(f"\nWatching for  : '{alert_class}'")
    print("Webcam started — press Q to quit\n")

    alert_start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed — check webcam connection.")
            break

        results  = model.predict(frame, conf=conf, verbose=False)
        result   = results[0]

        detected = [int(cls) for cls in result.boxes.cls]
        counts   = Counter(detected)
        is_alert = alert_id not in detected

        annotated = result.plot()
        h, w      = annotated.shape[:2]

        # ── Alert banner ─────────────────────────────────────────
        overlay      = annotated.copy()
        banner_color = (0, 0, 200) if is_alert else (0, 150, 0)
        cv2.rectangle(overlay, (0, 0), (w, 70), banner_color, -1)
        cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)

        if is_alert:
            if alert_start_time is None:
                alert_start_time = time.time()
            duration = int(time.time() - alert_start_time)
            msg = f"  ALERT: '{alert_class}' NOT DETECTED  ({duration}s)"
        else:
            alert_start_time = None
            count = counts.get(alert_id, 0)
            msg   = f"  OK: '{alert_class}' detected ({count})"

        cv2.putText(annotated, msg, (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # ── Bottom info ──────────────────────────────────────────
        total = sum(counts.values())
        cv2.putText(annotated, f"Objects in frame: {total}",
                    (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
        cv2.putText(annotated, "Press Q to quit",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

        cv2.imshow("Urban Detection — Alert System", annotated)

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")


if __name__ == "__main__":
    model  = load_model(MODEL_PATH)
    labels = model.names

    alert_id, alert_class = pick_alert_class(labels)
    run_alert_webcam(model, alert_id, alert_class, CONFIDENCE)