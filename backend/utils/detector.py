from ultralytics import YOLO
import numpy as np


class PersonDetector:
    def __init__(self, model_path="yolo11n.pt", conf=0.4, device="cpu"):
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device

    def detect(self, frame: np.ndarray) -> list:
        """Returns list of [x1, y1, x2, y2, score] for persons only (class 0)."""
        results = self.model(frame, conf=self.conf, classes=[0], device=self.device, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0])
                detections.append([x1, y1, x2, y2, score])
        return detections
