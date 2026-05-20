import numpy as np
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment


class Track:
    _id_counter = 0

    def __init__(self, bbox, score):
        Track._id_counter += 1
        self.id = Track._id_counter
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.score = score
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.path = [self._center()]

    def _center(self):
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)

    def update(self, bbox, score):
        self.bbox = bbox
        self.score = score
        self.hits += 1
        self.time_since_update = 0
        self.path.append(self._center())
        if len(self.path) > 50:
            self.path.pop(0)

    def predict(self):
        self.age += 1
        self.time_since_update += 1


def iou(b1, b2):
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0


class ByteTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: list[Track] = []

    def reset(self):
        self.tracks.clear()
        Track._id_counter = 0

    def update(self, detections):
        """
        detections: list of [x1, y1, x2, y2, score]
        returns: list of (track_id, bbox, path)
        """
        for t in self.tracks:
            t.predict()

        if len(detections) == 0:
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return self._active_tracks()

        det_bboxes = [d[:4] for d in detections]
        det_scores = [d[4] for d in detections]

        if self.tracks:
            cost = np.array([[1 - iou(t.bbox, d) for d in det_bboxes] for t in self.tracks])
            row_ind, col_ind = linear_sum_assignment(cost)

            matched_tracks, matched_dets = set(), set()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < (1 - self.iou_threshold):
                    self.tracks[r].update(det_bboxes[c], det_scores[c])
                    matched_tracks.add(r)
                    matched_dets.add(c)

            for i, det in enumerate(det_bboxes):
                if i not in matched_dets:
                    self.tracks.append(Track(det, det_scores[i]))
        else:
            for bbox, score in zip(det_bboxes, det_scores):
                self.tracks.append(Track(bbox, score))

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return self._active_tracks()

    def _active_tracks(self):
        return [
            {"id": t.id, "bbox": t.bbox, "path": t.path, "score": t.score}
            for t in self.tracks
            if t.time_since_update == 0  # only show tracks seen in this frame
        ]
