import cv2
import numpy as np

#Pitch calibration, hand-picked in frame CALIB_FRAME of input_videos/1.mp4: penalty box + goal area corners.
#Pitch coords in meters: x along the goal line (0 = goal centre), y = distance out from the goal line.
#ponytail: valid for this video only, re-pick the points for another video (or train a pitch-keypoint model)
CALIB_FRAME = 749
CALIB_PIXELS = np.float32([[448, 270], [823, 295], [320, 663], [313, 339], [444, 348], [178, 500]])
CALIB_METERS = np.float32([[-20.16, 0], [-20.16, 16.5], [20.16, 16.5], [-9.16, 0], [-9.16, 5.5], [9.16, 5.5]])

#Broadcast overlays that stay still while the camera moves (black box top-right, watermark bottom-right)
OVERLAYS = [(1190, 55, 1860, 160), (1350, 960, 1850, 1035)]

def estimate_camera_motion(frames, tracks):
    """Homography per frame mapping frame-0 pixels -> frame-i pixels (camera pans AND zooms)."""
    camera = [np.eye(3)]
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for i in range(1, len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        #Only track static scenery: hide overlays and everything that moves on its own
        mask = np.full(gray.shape, 255, np.uint8)
        for x1, y1, x2, y2 in OVERLAYS:
            mask[y1:y2, x1:x2] = 0
        for objects in tracks.values():
            for obj in objects[i - 1].values():
                x1, y1, x2, y2 = map(int, obj["bbox"])
                mask[max(y1 - 10, 0):y2 + 10, max(x1 - 10, 0):x2 + 10] = 0

        step = np.eye(3)
        p0 = cv2.goodFeaturesToTrack(prev, maxCorners=500, qualityLevel=0.01, minDistance=10, mask=mask)
        if p0 is not None:
            p1, status, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None)
            ok = status.ravel() == 1
            if ok.sum() >= 8:
                h, _ = cv2.findHomography(p0[ok], p1[ok], cv2.RANSAC, 2.0)
                if h is not None:
                    step = h
        #ponytail: chained frame-to-frame, drift grows with video length; re-anchor on pitch lines if clips get long
        camera.append(step @ camera[-1])
        prev = gray
    return camera

def add_positions(tracks, camera):
    """Players' foot point -> 'position' (pixels) and 'position_m' (pitch meters, camera motion removed)."""
    pixels_to_meters, _ = cv2.findHomography(CALIB_PIXELS, CALIB_METERS)
    for i, players in enumerate(tracks["players"]):
        #frame i pixels -> frame 0 -> calibration frame -> meters
        to_meters = pixels_to_meters @ camera[CALIB_FRAME] @ np.linalg.inv(camera[i])
        for player in players.values():
            x1, _, x2, y2 = player["bbox"]
            player["position"] = ((x1 + x2) / 2, y2)
            player["position_m"] = cv2.perspectiveTransform(np.float32([[player["position"]]]), to_meters)[0, 0].tolist()

def frames_per_player(players):
    frames = {}
    for f, frame in enumerate(players):
        for track_id in frame:
            frames.setdefault(track_id, []).append(f)
    return frames

def stitch_tracks(tracks, fps, max_gap_s=3, max_mps=7, slack_m=1.5):
    """Tracker lost a player and gave him a new ID: if ID B (same team) starts within reach (meters)
    of where ID A ended shortly before, B continues A and takes A's ID. Needs 'position_m' and 'team'."""
    players = tracks["players"]
    ends = {}  #surviving ID -> (last frame, last position, team)
    for track_id, frames in sorted(frames_per_player(players).items(), key=lambda kv: kv[1][0]):
        first, last = frames[0], frames[-1]
        start, team = players[first][track_id]["position_m"], players[first][track_id].get("team")
        best, best_d = track_id, None
        #ponytail: greedy nearest match, swap for Hungarian assignment if crossing players get mixed up
        for old_id, (end_f, end, old_team) in ends.items():
            gap = (first - end_f) / fps
            d = float(np.linalg.norm(np.subtract(start, end)))
            if old_team == team and 0 < gap <= max_gap_s and d <= slack_m + max_mps * gap and (best_d is None or d < best_d):
                best, best_d = old_id, d
        if best != track_id:
            for f in frames:
                players[f][best] = players[f].pop(track_id)
        ends[best] = (last, players[last][best]["position_m"], team)

def add_speed_and_distance(tracks, fps, window=10, max_kmh=40):
    """Speed (km/h) over `window` frames and running distance (m) for every player.
    window=5 inflated distance 5-17% from bbox jitter on 1.mp4; 10+ is stable."""
    players = tracks["players"]
    for track_id, frames in frames_per_player(players).items():
        total, i = 0.0, 0
        for j in range(1, len(frames)):
            if frames[j] - frames[i] < window and j < len(frames) - 1:
                continue
            a, b = players[frames[i]][track_id], players[frames[j]][track_id]
            meters = float(np.linalg.norm(np.subtract(b["position_m"], a["position_m"])))
            kmh = meters / ((frames[j] - frames[i]) / fps) * 3.6
            if kmh <= max_kmh:  #faster = ID switch or bad box, nobody sprints that fast
                total += meters
                for f in frames[i:j + 1]:
                    players[f][track_id].update(speed=kmh, distance=total)
            i = j

if __name__ == "__main__":
    #Player 1 walks 0.1 m/frame (9 km/h), player 2 teleports 1 m/frame (90 km/h -> rejected)
    tracks = {"players": [{1: {"position_m": [0.1 * f, 0]}, 2: {"position_m": [0, f]}} for f in range(11)]}
    add_speed_and_distance(tracks, fps=25)
    assert abs(tracks["players"][3][1]["speed"] - 9) < 1e-6
    assert abs(tracks["players"][10][1]["distance"] - 1.0) < 1e-6
    assert "speed" not in tracks["players"][3][2]

    #ID 7 lost after frame 2 at x=2, ID 9 appears at frame 10 at x=3 -> same player;
    #ID 8 is 30 m away, ID 6 is closer (checked first) but the other team
    tracks = {"players": [{7: {"position_m": [f, 0], "team": 0}} for f in range(3)] + [{} for _ in range(7)]
              + [{6: {"position_m": [2.5, 0], "team": 1}, 9: {"position_m": [3, 0], "team": 0}, 8: {"position_m": [30, 0], "team": 0}}]}
    stitch_tracks(tracks, fps=25)
    assert set(tracks["players"][10]) == {7, 8, 6}
    print("ok")
