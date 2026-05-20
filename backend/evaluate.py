"""
Evaluate CSRNet (MAE, MSE) and ByteTracker (MOTA, ID switches, Precision, Recall).

Usage:
  python evaluate.py --part A
  python evaluate.py --part A --weights weights/csrnet_shanghaitech.pth
"""

import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from models.csrnet import CSRNet
from train_csrnet import ShanghaiTechDataset
from utils.tracker import ByteTracker
from utils.detector import PersonDetector


# ── CSRNet Evaluation ─────────────────────────────────────────────────────────

def evaluate_csrnet(weights="weights/csrnet_shanghaitech.pth", data_root="ShanghaiTech", part="A"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CSRNet().to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()

    loader = DataLoader(ShanghaiTechDataset(data_root, part, "test_data"), batch_size=1, shuffle=False)
    mae_sum, mse_sum, n = 0, 0, 0

    with torch.no_grad():
        for imgs, targets in loader:
            pred = model(imgs.to(device)).sum().item()
            gt = targets.sum().item()
            err = pred - gt
            mae_sum += abs(err)
            mse_sum += err ** 2
            n += 1

    mae = mae_sum / n
    mse = (mse_sum / n) ** 0.5
    return {"mae": round(mae, 2), "mse": round(mse, 2), "samples": n}


# ── Tracking Accuracy (MOTA) ──────────────────────────────────────────────────

def evaluate_tracking(video_path: str, gt_counts: list[int] = None):
    """
    Runs YOLOv11 + ByteTracker on a video and computes:
      - MOTA  = 1 - (FN + FP + ID_switches) / total_gt
      - ID switches
      - Precision, Recall (based on per-frame detection vs gt count)

    gt_counts: optional list of ground-truth person counts per frame.
               If None, uses YOLO detections as pseudo-GT (measures tracker consistency).
    """
    import cv2
    detector = PersonDetector()
    tracker = ByteTracker()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Cannot open video: {video_path}"}

    prev_ids = set()
    id_switches = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    total_det = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        current_ids = {t["id"] for t in tracks}

        # ID switches = IDs that disappeared and new ones appeared in same frame
        lost = prev_ids - current_ids
        gained = current_ids - prev_ids
        id_switches += min(len(lost), len(gained))
        prev_ids = current_ids

        det_count = len(tracks)
        gt_count = gt_counts[frame_idx] if gt_counts and frame_idx < len(gt_counts) else det_count

        fp = max(0, det_count - gt_count)
        fn = max(0, gt_count - det_count)
        total_fp += fp
        total_fn += fn
        total_gt += gt_count
        total_det += det_count
        frame_idx += 1

    cap.release()

    if total_gt == 0:
        return {"error": "No ground truth data"}

    mota = round(1 - (total_fn + total_fp + id_switches) / max(total_gt, 1), 4)
    precision = round(total_det / max(total_det + total_fp, 1), 4)
    recall = round((total_gt - total_fn) / max(total_gt, 1), 4)

    return {
        "mota": mota,
        "id_switches": id_switches,
        "precision": precision,
        "recall": recall,
        "total_frames": frame_idx,
        "total_gt_count": total_gt,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default="A", choices=["A", "B"])
    parser.add_argument("--weights", default="weights/csrnet_shanghaitech.pth")
    parser.add_argument("--video", default=None, help="Optional video path for tracking evaluation")
    args = parser.parse_args()

    print("\n== CSRNet Evaluation (ShanghaiTech Part", args.part, ")==")
    csrnet_results = evaluate_csrnet(weights=args.weights, part=args.part)
    print("  MAE     :", csrnet_results['mae'])
    print("  MSE     :", csrnet_results['mse'])
    print("  Samples :", csrnet_results['samples'])

    if args.video:
        print("\n== Tracking Evaluation", args.video, "==")
        tracking_results = evaluate_tracking(args.video)
        print("  MOTA        :", tracking_results.get('mota'))
        print("  ID Switches :", tracking_results.get('id_switches'))
        print("  Precision   :", tracking_results.get('precision'))
        print("  Recall      :", tracking_results.get('recall'))
        print("  Frames      :", tracking_results.get('total_frames'))
