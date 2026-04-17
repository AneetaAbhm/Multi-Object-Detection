import cv2
import time
from collections import Counter
from ultralytics import YOLO
from pathlib import Path

model = YOLO(r"yolo_trained_model\best.pt")   # loaded once

def generate_alert_frames(alert_class: str, conf=0.25, camera_idx=0, resolution=(1280,720), model_path=None):
    labels = model.names
    
    # Get alert_id
    name_to_id = {v.lower(): k for k, v in labels.items()}
    alert_id = name_to_id.get(alert_class.lower())
    if alert_id is None:
        print(f"Class '{alert_class}' not found!")
        return

    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    alert_start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, verbose=False)
        result = results[0]

        detected = [int(cls) for cls in result.boxes.cls]
        counts = Counter(detected)
        is_alert = alert_id not in detected

        annotated = result.plot()
        h, w = annotated.shape[:2]

        # === Exact alert banner from your count_webcam.py ===
        overlay = annotated.copy()
        banner_color = (0, 0, 200) if is_alert else (0, 150, 0)
        cv2.rectangle(overlay, (0, 0), (w, 80), banner_color, -1)
        cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)

        if is_alert:
            if alert_start_time is None:
                alert_start_time = time.time()
            duration = int(time.time() - alert_start_time)
            msg = f"  ALERT: '{alert_class}' NOT DETECTED  ({duration}s)"
        else:
            alert_start_time = None
            count = counts.get(alert_id, 0)
            msg = f"  OK: '{alert_class}' detected ({count})"

        cv2.putText(annotated, msg, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

        # Bottom info
        total = sum(counts.values())
        cv2.putText(annotated, f"Objects in frame: {total}",
                    (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()