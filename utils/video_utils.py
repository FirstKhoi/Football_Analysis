import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps

def save_video(output_video_frames, output_video_path, fps):
    h, w = output_video_frames[0].shape[:2]
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))
    if not out.isOpened():
        raise IOError(f"Không ghi được: {output_video_path} (thư mục có tồn tại chưa?)")
    for frame in output_video_frames:
        out.write(frame)
    out.release()
