import os
import pickle
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        #Looser matching: small boxes + panning camera give low IoU (121 -> 86 player IDs on 1.mp4)
        # ponytail: deprecated, removed in supervision 0.31 -> pin supervision<0.31
        self.tracker = sv.ByteTrack(lost_track_buffer=90, minimum_matching_threshold=0.9)

    def detect_frames(self, frames):
        batch_size = 8  #yolo26m @ 1280 on the Mac: 32 took +10GB RAM and ran slower
        detections = []
        for i in range(0, len(frames), batch_size):
            detections += self.model.predict(frames[i : i+batch_size], conf=0.1, verbose=False)
        return detections

    def get_object_tracks(self, frames, stub_path=None):
        #Cached result: delete the stub file after changing the video or the model
        if stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        cls_names_inv = {v:k for k, v in self.model.names.items()}
        tracks = {"players": [], "referees": [], "ball": []}
        tracked, votes = [], {}

        for detection in self.detect_frames(frames):
            #Convert to Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            #Goalkeeper -> player (few goalkeeper samples, model mixes them up)
            is_goalkeeper = detection_supervision.class_id == cls_names_inv["goalkeeper"]
            detection_supervision.class_id[is_goalkeeper] = cls_names_inv["player"]

            #Track objects (ByteTrack ignores class: one ID can be player in one frame, referee in the next)
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            tracked.append(detection_with_tracks)
            for cls_id, track_id in zip(detection_with_tracks.class_id, detection_with_tracks.tracker_id):
                votes.setdefault(int(track_id), Counter())[cls_id] += 1

            #Only one ball -> skip tracker (it drops small fast objects), fixed id 1
            tracks["ball"].append({})
            for bbox, cls_id in zip(detection_supervision.xyxy, detection_supervision.class_id):
                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][-1][1] = {"bbox": bbox.tolist()}

        #Model flips player <-> referee for a few frames (29 of 96 IDs on 1.mp4): one class per track, majority vote
        for detection_with_tracks in tracked:
            tracks["players"].append({})
            tracks["referees"].append({})
            for bbox, track_id in zip(detection_with_tracks.xyxy, detection_with_tracks.tracker_id):
                cls_id = votes[int(track_id)].most_common(1)[0][0]
                if cls_id == cls_names_inv["player"]:
                    tracks["players"][-1][int(track_id)] = {"bbox": bbox.tolist()}
                elif cls_id == cls_names_inv["referee"]:
                    tracks["referees"][-1][int(track_id)] = {"bbox": bbox.tolist()}

        if stub_path:
            os.makedirs(os.path.dirname(stub_path) or '.', exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        x1, _, x2, y2 = map(int, bbox)
        x_center, width = (x1 + x2) // 2, x2 - x1
        cv2.ellipse(frame, (x_center, y2), (width, int(0.35 * width)), 0, -45, 235, color, 2, cv2.LINE_4)
        if track_id is not None:
            cv2.rectangle(frame, (x_center - 20, y2 + 5), (x_center + 20, y2 + 25), color, cv2.FILLED)
            cv2.putText(frame, str(track_id), (x_center - 14, y2 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    def draw_annotations(self, frames, tracks):
        #Draws in place: 750 full-HD frames are ~4.6GB, a copy would double that
        for i, frame in enumerate(frames):
            for track_id, player in tracks["players"][i].items():
                self.draw_ellipse(frame, player["bbox"], player.get("team_color", (0, 0, 255)), track_id)
                if "speed" in player:
                    x, y = int((player["bbox"][0] + player["bbox"][2]) / 2) - 35, int(player["bbox"][3]) + 45
                    for text, dy in ((f'{player["speed"]:.1f} km/h', 0), (f'{player["distance"]:.1f} m', 20)):
                        cv2.putText(frame, text, (x, y + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            for referee in tracks["referees"][i].values():
                self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            for ball in tracks["ball"][i].values():
                x1, y1, x2, _ = map(int, ball["bbox"])
                x = (x1 + x2) // 2
                triangle = np.array([[x, y1], [x - 10, y1 - 20], [x + 10, y1 - 20]])
                cv2.drawContours(frame, [triangle], 0, (0, 255, 0), cv2.FILLED)
                cv2.drawContours(frame, [triangle], 0, (0, 0, 0), 2)
        return frames
