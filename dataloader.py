import os
import random

import configuration as cfg
import parameters as params
from utils.video_util import *
from utils.data_util import *

from torch.utils.data import Dataset, DataLoader
import cv2


class DataGenerator(Dataset):

    def __init__(self, data_split):
        if data_split == 'train':
            self.data_file = cfg.train_split_file
            self.data_percentage = params.train_percent
        elif data_split == 'test':
            self.data_file = cfg.test_split_file
            self.data_percentage = params.validation_percent
        self.inputs = self.get_inputs()
        self.samples = self.build_samples()
        len_data = int(len(self.samples) * self.data_percentage)
        self.samples = self.samples[0:len_data]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        anomaly_input, normal_input = sample[0], sample[1]
        anomaly_clip = self.build_clip(anomaly_input[0], anomaly_input[1], anomaly_input[2])
        normal_clip = self.build_clip(normal_input[0], normal_input[1], normal_input[2])
        return anomaly_clip, normal_clip

    def get_inputs(self):
        inputs = {'anomaly': [], 'normal': []}
        videos = open(self.data_file, 'r')
        for video in videos.readlines():
            video = video.rstrip()
            if 'Normal' in video:
                inputs['normal'].append([video, -1, -1])
            else:
                annotation_file = os.path.join(cfg.annotations_folder, video + '.txt')
                annotations = open(annotation_file, 'r').readlines()
                start_frame = int(annotations[0].split(' ')[5])
                end_frame = int(annotations[-1].split(' ')[5])
                inputs['anomaly'].append([video, start_frame, end_frame])
        return inputs

    def build_samples(self):
        anomaly_inputs = self.inputs['anomaly']
        normal_inputs = self.inputs['normal']
        samples = []
        for i in range(params.samples_per_epoch):
            samples.append((random.choice(anomaly_inputs), random.choice(normal_inputs)))
        return samples

    def build_clip(self, video_id, annotation_start, annotation_end):
        video_folder = video_id[0:-3] if 'Normal' not in video_id else 'Training_Normal_Videos_Anomaly'
        video_file = os.path.join(cfg.dataset_folder, video_folder, video_id + '_x264.mp4')
        num_frames = get_length(video_file)
        len_slice = params.frames_per_clip * params.skip_rate
        if 'Normal' in video_id:
            start = random.choice(range(0, num_frames - len_slice))
            end = start + len_slice
        else:
            assert annotation_start < num_frames and annotation_end <= num_frames and annotation_start < annotation_end
            assert (annotation_end - annotation_start) > len_slice
            start = random.choice(range(annotation_start, annotation_end - len_slice))
            end = start + len_slice

        frames = range(start, end, params.skip_rate)
        clip = get_frames(video_file, frames)
        clip = [cv2.resize(frame, (224, 224)) for frame in clip] 
        clip = np.array(clip)/255.0
        assert len(clip) == params.frames_per_clip
        return clip

