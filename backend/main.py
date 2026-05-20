import asyncio
import base64
import tempfile
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

from utils.detector import PersonDetector
from utils.tracker import ByteTracker
from utils.crowd_analyzer import CrowdAnalyzer, get_density_category
from utils.alerts import send_whatsapp_alert
from database import init_db, insert_stat, get_recent_stats, get_summary

CSRNET_WEIGHTS = "weights/csrnet_shanghaitech.pth"
YOLO_MODEL = "yolo11n.pt"

detector = None
tracker = None
analyzer = None


def load_models():
    global detector, tracker, analyzer
    detector = PersonDetector(model_path=YOLO_MODEL)
    tracker = ByteTracker()
    analyzer = CrowdAnalyzer(
        model_path=CSRNET_WEIGHTS if os.path.exists(CSRNET_WEIGHTS) else None
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    load_models()
    yield


app = FastAPI(title="Crowd Detection & Tracking API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def process_frame(frame: np.ndarray, frame_id: int) -> dict:
    detections = detector.detect(frame)
    tracks = tracker.update(detections)
    count, density_map, heatmap = analyzer.estimate_density(frame)
    category = get_density_category(count)
    alert = analyzer.check_alert(count)

    annotated = analyzer.draw_tracks(frame, tracks)
    overlay = analyzer.overlay_heatmap(annotated, heatmap, alpha=0.4)

    _, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 75])
    frame_b64 = base64.b64encode(buf).decode()

    _, hbuf = cv2.imencode(".jpg", heatmap, [cv2.IMWRITE_JPEG_QUALITY, 75])
    heatmap_b64 = base64.b64encode(hbuf).decode()

    insert_stat(frame_id, count, float(density_map.sum()), category, len(tracks), alert)

    # Send WhatsApp alert (cooldown handled inside)
    if alert:
        send_whatsapp_alert(int(count), category)

    return {
        "frame_id": frame_id,
        "person_count": round(count, 1),
        "density_score": round(float(density_map.sum()), 2),
        "density_category": category,
        "active_tracks": len(tracks),
        "track_ids": [t["id"] for t in tracks],
        "alert": alert,
        "frame": frame_b64,
        "heatmap": heatmap_b64,
    }


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    tracker.reset()
    frame_id = 0
    try:
        while True:
            data = await websocket.receive_bytes()
            arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            result = await asyncio.get_event_loop().run_in_executor(None, process_frame, frame, frame_id)
            frame_id += 1
            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass


@app.post("/analyze/upload")
async def analyze_upload(file: UploadFile = File(...)):
    """Analyze an uploaded video file, returns per-frame stats."""
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    tracker.reset()
    results = []
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % 5 == 0:  # process every 5th frame
            results.append(process_frame(frame, frame_id))
        frame_id += 1
    cap.release()
    os.remove(tmp_path)
    return {"total_frames_processed": len(results), "results": results}


@app.post("/reload")
def reload_models():
    """Hot-reload models after training completes."""
    load_models()
    return {"status": "reloaded", "weights_found": os.path.exists(CSRNET_WEIGHTS)}


@app.get("/stats/recent")
def recent_stats(limit: int = 100):
    return get_recent_stats(limit)


@app.get("/stats/summary")
def stats_summary():
    return get_summary()
