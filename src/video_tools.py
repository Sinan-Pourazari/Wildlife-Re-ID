import cv2 as cv
#from yt_dlp import YoutubeDL
"""
def get_video_stream(video_url,):
    "For later use"

    with YoutubeDL({'quiet': True, 'format': 'best'}) as ydl:
        info = ydl.extract_info(video_url, download=False)
        stream_url = info['url']

    cap = cv.VideoCapture(stream_url)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv.imshow("YouTube Stream", frame)
        if cv.waitKey(1) == ord("q"):
            break
        cap.release()
        cv.destroyAllWindows()
"""
def extract_frame_segments(video_path, segment_size=20, max_frames=None):
        cap = cv.VideoCapture(video_path)
        frames = []
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames.append(frame)
            count += 1

            if max_frames is not None and count >= max_frames:
                break

        cap.release()

        # Now group frames into segments of segment_size
        frame_segments = [
            frames[i:i + segment_size]
            for i in range(0, len(frames), segment_size)
            if len(frames[i:i + segment_size]) == segment_size  # only full segments
        ]

        return frame_segments
# TODO write function that gets an video as it plays and returns singular frames as it streames
# TODO create function to bunlde frames of empty videos into bundels for new background voter members
#Todo find a way to stich processed images back into video feed or modify videofeed based on gatherd infornamtion (image indictaes animal at xy so se boundingbox at xy in video feat)

def array_to_mp4(array):
    height, width = array[0].shape[:2]  # auto-detect from first frame
    out = cv.VideoWriter("output.mp4", cv.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    for frame in array:
        out.write(frame)  # frame must be uint8 (0–255) and BGR format
    out.release()
