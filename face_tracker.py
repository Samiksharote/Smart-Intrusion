import time
import math

class FaceTracker:
    """Simple per-face tracker that debounces unknown detections by centroid proximity.

    Tracks unknown faces by centroid and requires N confirmations before reporting an intruder.
    """
    def __init__(self, threshold=3, expiry_seconds=2.0, distance_pixels=60):
        self.threshold = int(threshold)
        self.expiry = float(expiry_seconds)
        self.distance = float(distance_pixels)
        self.tracks = {}  # id -> {centroid, count, last_seen, alerted, last_location}
        self._next_id = 1

    def _centroid(self, location):
        top, right, bottom, left = location
        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0
        return (cx, cy)

    def _dist(self, a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def update(self, detected_faces):
        """Update tracker with current frame's detected faces.

        detected_faces: list of dicts with keys: 'name', 'location'
        Returns: list of tracks confirmed as intruders (each is dict with 'location' and 'centroid')
        """
        now = time.time()
        # expire old tracks
        to_delete = []
        for tid, t in self.tracks.items():
            if now - t['last_seen'] > self.expiry:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        confirmed = []

        # iterate unknown faces only
        unknowns = [f for f in detected_faces if f.get('name') == 'Unknown']
        for face in unknowns:
            loc = face.get('location')
            if not loc:
                continue
            c = self._centroid(loc)
            # find nearest existing track
            best = None
            best_dist = None
            for tid, t in self.tracks.items():
                d = self._dist(c, t['centroid'])
                if best is None or d < best_dist:
                    best = tid
                    best_dist = d

            if best is not None and best_dist is not None and best_dist <= self.distance:
                t = self.tracks[best]
                # if this track was already alerted, skip counting
                if t.get('alerted'):
                    # update last_seen and location
                    t['last_seen'] = now
                    t['centroid'] = c
                    t['last_location'] = loc
                    continue

                t['count'] += 1
                t['last_seen'] = now
                t['centroid'] = c
                t['last_location'] = loc

                if t['count'] >= self.threshold:
                    t['alerted'] = True
                    confirmed.append({'location': t['last_location'], 'centroid': t['centroid']})

            else:
                # create new track
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = {
                    'centroid': c,
                    'count': 1,
                    'last_seen': now,
                    'alerted': False,
                    'last_location': loc
                }
                if self.tracks[tid]['count'] >= self.threshold:
                    self.tracks[tid]['alerted'] = True
                    confirmed.append({'location': loc, 'centroid': c})

        return confirmed
