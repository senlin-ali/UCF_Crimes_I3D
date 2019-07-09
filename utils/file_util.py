import os
import configuration as cfg


def get_video_paths():
    train_file = os.path.join(cfg.dataset_folder, cfg.train_split_file)
    train_videos = [os.path.join(cfg.dataset_folder, line.rstrip()) for line in open(train_file).readlines()]
    test_file = os.path.join(cfg.dataset_folder, cfg.test_split_file)
    test_videos = [os.path.join(cfg.dataset_folder, line.rstrip()) for line in open(test_file).readlines()]
    return train_videos, test_videos