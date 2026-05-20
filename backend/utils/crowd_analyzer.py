import cv2
import numpy as np
import torch
from models.csrnet import CSRNet

DENSITY_THRESHOLDS = {"low": 20, "medium": 50}  # people count thresholds
ALERT_THRESHOLD = 50  # trigger alert above this count


def get_density_category(count: float) -> str:
    if count <= DENSITY_THRESHOLDS["low"]:
        return "Low"
    elif count <= DENSITY_THRESHOLDS["medium"]:
        return "Medium"
    return "High"


class CrowdAnalyzer:
    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.device = device
        self.model = CSRNet().to(device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def estimate_density(self, frame: np.ndarray):
        """Returns (count, density_map_normalized, heatmap_bgr)."""
        img = cv2.resize(frame, (640, 480))
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            density_map = self.model(img_tensor).squeeze().cpu().numpy()

        count = float(density_map.sum())
        density_norm = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)
        heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

        return count, density_map, heatmap_resized

    def overlay_heatmap(self, frame: np.ndarray, heatmap: np.ndarray, alpha=0.5) -> np.ndarray:
        return cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)

    def check_alert(self, count: float) -> str | None:
        if count > ALERT_THRESHOLD:
            return f"⚠️ Warning: High Crowd Density Detected ({int(count)} people)"
        return None

    def draw_tracks(self, frame: np.ndarray, tracks: list) -> np.ndarray:
        out = frame.copy()
        for t in tracks:
            x1, y1, x2, y2 = [int(v) for v in t["bbox"]]

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label with filled background
            label = f" ID:{t['id']}  {t['score']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            label_y = y1 - 4 if y1 - 4 > th else y1 + th + 4
            cv2.rectangle(out, (x1, label_y - th - 4), (x1 + tw + 2, label_y + 2), (0, 255, 0), -1)
            cv2.putText(out, label, (x1 + 1, label_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

            # Movement path trail
            path = t["path"]
            for i in range(1, len(path)):
                cv2.line(out, (int(path[i-1][0]), int(path[i-1][1])),
                         (int(path[i][0]), int(path[i][1])), (0, 165, 255), 2)

        return out
