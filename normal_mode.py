import cv2
from collections import Counter
from ultralytics import YOLO
from pathlib import Path

def load_model(model_path):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return YOLO(str(path))

model = load_model(r"yolo_trained_model\best.pt")   # loaded once

def generate_normal_frames(conf=0.25, camera_idx=0, resolution=(1280,720), model_path=None):
    labels = model.names
    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, verbose=False)
        result = results[0]
        class_counts = Counter(int(cls) for cls in result.boxes.cls)

        annotated = result.plot()

        # === Exact panel from your alert_detct.py ===
        h, w = annotated.shape[:2]
        panel_w = 280
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)

        cv2.putText(annotated, "YOLO Detection", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
        cv2.line(annotated, (10, 50), (panel_w - 10, 50), (100, 100, 100), 1)

        y = 85
        for cls_id, count in sorted(class_counts.items()):
            class_name = labels[cls_id]
            color = (0, 255, 0) if any(x in class_name.lower() for x in ["person", "car"]) else (255, 220, 80)
            cv2.putText(annotated, f"{class_name:<18}: {count}",
                        (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            y += 32

        total = sum(class_counts.values())
        cv2.putText(annotated, f"Total objects : {total}",
                    (15, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)

        cv2.putText(annotated, "Press ESC in browser tab to stop", (w-400, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()