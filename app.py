from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, HTMLResponse
from pathlib import Path

app = FastAPI(title="YOLO Detection System")

MODEL_PATH = r"yolo_trained_model\best.pt"
CONFIDENCE = 0.35
CAMERA_IDX = 0
RESOLUTION = (640, 480)

# Import the mode generators
from normal_mode import generate_normal_frames
from alert_mode import generate_alert_frames

@app.get("/", response_class=HTMLResponse)
async def home():
    # Get available classes from the model (we'll load it once here)
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    class_list = sorted(model.names.values())

    html_content = f"""
    <html>
    <head>
        <title>YOLO Detection - Normal & Alert</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                background: #f8f9fa; 
                color: #333; 
                margin: 0; 
                padding: 30px; 
            }}
            .container {{ max-width: 1100px; margin: auto; }}
            h1 {{ color: #2c3e50; text-align: center; }}
            .card {{
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                padding: 25px;
                margin: 20px 0;
            }}
            .mode-btn {{
                padding: 14px 30px;
                margin: 10px;
                font-size: 18px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                width: 280px;
            }}
            .normal {{ background: #3498db; color: white; }}
            .alert  {{ background: #e74c3c; color: white; }}
            select, button {{ padding: 10px; font-size: 16px; margin: 10px 5px; }}
            img {{ 
                border: 4px solid #ddd; 
                border-radius: 10px; 
                margin-top: 20px; 
                box-shadow: 0 6px 20px rgba(0,0,0,0.15);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1> YOLO Multi Object Detection System</h1>

            <!-- Normal Mode -->
            <div class="card">
                <h2>Normal Mode</h2>
                <p>Shows all detections with detailed count panel (same as your original alert_detct.py)</p>
                <a href="/normal"><button class="mode-btn normal">Start Normal Mode</button></a>
                <img src="/normal" width="100%" alt="Normal Stream">
            </div>

            <!-- Alert Mode -->
            <div class="card">
                <h2>Alert Mode</h2>
                <p>Watch a specific class. Red banner appears when it is NOT detected.</p>
                
                <form action="/alert" method="get" target="_blank">
                    <select name="alert_class" required>
                        <option value="">-- Select Class to Monitor --</option>
    """
    for cls in class_list:
        html_content += f'                        <option value="{cls}">{cls}</option>\n'
    
    html_content += """
                    </select>
                    <button type="submit" class="mode-btn alert">Start Alert Mode</button>
                </form>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content


@app.get("/normal")
async def normal_stream():
    return StreamingResponse(
        generate_normal_frames(CONFIDENCE, CAMERA_IDX, RESOLUTION),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/alert")
async def alert_stream(alert_class: str = Query(..., description="Class name to monitor")):
    return StreamingResponse(
        generate_alert_frames(alert_class, CONFIDENCE, CAMERA_IDX, RESOLUTION),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    import uvicorn
    print(" Server running at http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)