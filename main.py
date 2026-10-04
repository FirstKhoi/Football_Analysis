from utils import read_video, save_video
from trackers import Tracker
from motion import estimate_camera_motion, add_positions, stitch_tracks, add_speed_and_distance
from teams import assign_teams

def main():
    #read videos
    video_frames, fps = read_video('input_videos/1.mp4')

    #Initialize Tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, stub_path='stubs/track_stubs.pkl')

    #Camera motion -> position in meters -> teams -> re-join lost IDs -> speed & distance
    camera = estimate_camera_motion(video_frames, tracks)
    add_positions(tracks, camera)
    assign_teams(tracks, video_frames)
    stitch_tracks(tracks, fps)
    add_speed_and_distance(tracks, fps)

    #draw & save videos
    tracker.draw_annotations(video_frames, tracks)
    save_video(video_frames, 'output_videos/output_video.mp4', fps)

if __name__ == '__main__':
    main()
