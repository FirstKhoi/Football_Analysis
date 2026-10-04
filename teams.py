from collections import Counter

import cv2
import numpy as np

def shirt_color(frame, bbox):
    """Mean Lab colour of the shirt: upper-middle of the box with grass pixels removed."""
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(x1, 0), max(y1, 0)
    h, w = y2 - y1, x2 - x1
    torso = frame[y1 + h // 5 : y1 + h // 2, x1 + w // 4 : x2 - w // 4]
    if torso.size == 0:
        return None
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    shirt = ~((hsv[:, 0] > 30) & (hsv[:, 0] < 90) & (hsv[:, 1] > 40))
    return lab[shirt].mean(axis=0) if shirt.any() else None

def assign_teams(tracks, frames):
    """'team' (0/1) and 'team_color' (BGR) for every player: KMeans(2) on shirt colours, majority vote per track."""
    players = tracks["players"]
    samples = [(track_id, shirt_color(frames[f], p["bbox"])) for f, frame in enumerate(players) for track_id, p in frame.items()]
    samples = [(track_id, c) for track_id, c in samples if c is not None]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(np.float32([c for _, c in samples]), 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    team_bgr = cv2.cvtColor(np.uint8([centers]), cv2.COLOR_LAB2BGR)[0].tolist()

    #Single frames are noisy (overlapping players, blur): the track's majority decides
    votes = {}
    for (track_id, _), label in zip(samples, labels.ravel()):
        votes.setdefault(track_id, Counter())[int(label)] += 1
    for frame in players:
        for track_id, p in frame.items():
            if track_id in votes:
                p["team"] = votes[track_id].most_common(1)[0][0]
                p["team_color"] = team_bgr[p["team"]]
