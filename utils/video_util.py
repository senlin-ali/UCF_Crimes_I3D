import cv2
import numpy as np


def get_frames(video_path, frames):
    cap = cv2.VideoCapture(video_path)
    clip = []
    for frame in frames:
        cap.set(1, frame)
        ret, img = cap.read()
        assert ret == True
        clip.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.array(clip)


def get_length(video_path):
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length

