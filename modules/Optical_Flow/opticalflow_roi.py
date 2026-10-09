import time

import cv2
import numpy as np

_ROI_FEATURE_PARAMS = dict(qualityLevel=0.3, minDistance=7, blockSize=7)
_ROI_LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

def flow_on_crops(prev_gray, curr_gray, max_point=30):
    """Lucas-Kanade entre dois crops do mesmo retângulo. Vetor zero se não houver par válido."""
    if prev_gray is None or curr_gray is None:
        return 0.0, 0.0, 0.0, 0.0
    if prev_gray.shape != curr_gray.shape or prev_gray.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    if min(prev_gray.shape[:2]) < 8:
        return 0.0, 0.0, 0.0, 0.0
    points = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=max_point, **_ROI_FEATURE_PARAMS
    )
    if points is None:
        return 0.0, 0.0, 0.0, 0.0
    next_points, status, _err = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, points, None, **_ROI_LK_PARAMS
    )
    if next_points is None or status is None:
        return 0.0, 0.0, 0.0, 0.0
    good = (next_points - points).reshape(-1, 2)[status.reshape(-1) == 1]
    if len(good) == 0:
        return 0.0, 0.0, 0.0, 0.0
    dx, dy = good.mean(axis=0)
    magnitude = float(np.hypot(dx, dy))
    angle = float(np.degrees(np.arctan2(dy, dx)))
    return float(dx), float(dy), magnitude, angle

def process_roi_batch(rois, max_point=30):
    """rois: lista de (track_id, crop_anterior, crop_atual)."""
    t0 = time.time()
    results = []
    for track_id, prev_crop, curr_crop in rois:
        dx, dy, magnitude, angle = flow_on_crops(prev_crop, curr_crop, max_point)
        results.append((track_id, dx, dy, magnitude, angle))
    return results, time.time() - t0


class RoiPointTracker:
    """Pontos de Lucas-Kanade que continuam no mesmo track, como no fluxo antigo."""

    def __init__(self, max_point=30):
        self.max_point = max_point
        self._points = {}

    def step(self, items, prev_gray, curr_gray):
        """items: lista de (track_id, box). A seta é a mediana dos pontos que sobreviveram."""
        t0 = time.time()
        active = set()
        results = []
        for track_id, box in items:
            active.add(track_id)
            results.append(self._step_track(track_id, box, prev_gray, curr_gray))
        for track_id in list(self._points):
            if track_id not in active:
                del self._points[track_id]
        return results, time.time() - t0

    def _step_track(self, track_id, box, prev_gray, curr_gray):
        pts = self._points.get(track_id)
        good_new = np.empty((0, 2), dtype=np.float32)
        disp = None
        if (
            pts is not None
            and len(pts) > 0
            and prev_gray is not None
            and curr_gray is not None
        ):
            nxt, status, _err = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, pts, None, **_ROI_LK_PARAMS
            )
            if nxt is not None and status is not None:
                keep = status.reshape(-1) == 1
                nxt_xy = nxt.reshape(-1, 2)
                old_xy = pts.reshape(-1, 2)
                keep = keep & self._inside(nxt_xy, box, curr_gray.shape)
                if np.any(keep):
                    good_new = nxt_xy[keep]
                    disp = good_new - old_xy[keep]

        if disp is not None and len(disp) > 0:
            dx = float(np.median(disp[:, 0]))
            dy = float(np.median(disp[:, 1]))
            magnitude = float(np.hypot(dx, dy))
            angle = float(np.degrees(np.arctan2(dy, dx)))
        else:
            dx = dy = magnitude = angle = 0.0

        need = self.max_point - len(good_new)
        if need > 0 and curr_gray is not None:
            seeded = self._seed(curr_gray, box, good_new, need)
            if seeded is not None:
                good_new = np.vstack([good_new, seeded]) if len(good_new) else seeded

        if len(good_new) > 0:
            self._points[track_id] = good_new.reshape(-1, 1, 2).astype(np.float32)
        elif track_id in self._points:
            del self._points[track_id]
        return track_id, dx, dy, magnitude, angle

    def _inside(self, xy, box, shape):
        height, width = shape[:2]
        x1, y1, x2, y2 = [float(v) for v in box]
        margin = max(20.0, 0.25 * max(x2 - x1, y2 - y1))
        x1 = max(0.0, x1 - margin)
        y1 = max(0.0, y1 - margin)
        x2 = min(width - 1.0, x2 + margin)
        y2 = min(height - 1.0, y2 + margin)
        return (xy[:, 0] >= x1) & (xy[:, 0] <= x2) & (xy[:, 1] >= y1) & (xy[:, 1] <= y2)

    def _seed(self, gray, box, existing, need):
        height, width = gray.shape[:2]
        x1 = max(0, int(np.floor(box[0])))
        y1 = max(0, int(np.floor(box[1])))
        x2 = min(width, int(np.ceil(box[2])))
        y2 = min(height, int(np.ceil(box[3])))
        if x2 - x1 < 8 or y2 - y1 < 8 or need <= 0:
            return None
        roi = gray[y1:y2, x1:x2]
        mask = np.full(roi.shape, 255, np.uint8)
        for x, y in np.asarray(existing, dtype=np.float32).reshape(-1, 2):
            cv2.circle(mask, (int(round(x - x1)), int(round(y - y1))), 7, 0, -1)
        corners = cv2.goodFeaturesToTrack(
            roi, mask=mask, maxCorners=int(need), **_ROI_FEATURE_PARAMS
        )
        if corners is None:
            return None
        corners = corners.reshape(-1, 2).astype(np.float32)
        corners[:, 0] += x1
        corners[:, 1] += y1
        return corners
