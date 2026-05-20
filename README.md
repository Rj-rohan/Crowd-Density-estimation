# Crowd Detection & Tracking System

Real-time crowd monitoring using YOLOv11 + ByteTrack + CSRNet (trained on ShanghaiTech).

## Architecture

```
Video/Camera → YOLOv11 (person detection) → ByteTracker (unique IDs + paths)
                                          ↓
                              CSRNet (density map estimation)
                                          ↓
                    Heatmap | Count | Density Category | Alerts
                                          ↓
                              FastAPI WebSocket → React Dashboard
```

## Setup

### Backend

```bash
cd backend
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
```

---

## Dataset: ShanghaiTech

Already placed under `backend/ShanghaiTech/` with structure:
```
ShanghaiTech/
  part_A/
    train_data/
      images/     processed_IMG_N.jpg
      ground-truth/  GT_IMG_N.mat
    test_data/
      images/
      ground-truth/
  part_B/         (same structure)
```

### Generate density maps

```bash
cd backend
python data/generate_density_maps.py --root ShanghaiTech --part A --split train_data
python data/generate_density_maps.py --root ShanghaiTech --part A --split test_data
```

This reads each `GT_IMG_N.mat` (head point annotations) and writes `maps/IMG_N.npy` Gaussian density maps.

---

## Training CSRNet

```bash
cd backend
python train_csrnet.py --part A --epochs 100
# Saves best model to weights/csrnet_shanghaitech.pth
```

Use `--part B` for the sparse-crowd part.

---

## Evaluation

```bash
cd backend
python evaluate.py --part A
# Reports MAE and MSE on test_data split
```

---

## Running the System

### Start backend
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Start frontend
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| WS | `/ws/stream` | Real-time frame processing |
| POST | `/analyze/upload` | Upload & analyze video file |
| GET | `/stats/recent?limit=100` | Recent crowd statistics |
| GET | `/stats/summary` | Aggregated summary |

---

## Density Thresholds

| Category | Count |
|----------|-------|
| Low | ≤ 20 |
| Medium | 21–50 |
| High | > 50 |

Alert triggered when count > 50.

---

## Performance Metrics

- **MAE** (Mean Absolute Error) — crowd count accuracy
- **MSE** (Mean Squared Error) — penalizes large errors
- **Tracking accuracy** — measured via ID consistency across frames
