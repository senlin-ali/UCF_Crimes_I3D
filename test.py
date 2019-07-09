from dataloader import *
from model import *
import torch
import cv2
import skvideo.io


def visualize_clip(clip, mask, save_file):
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    mask = cv2.resize(np.array(mask), (224, 224))
    writer = skvideo.io.FFmpegWriter(save_file, outputdict={'-framerate': '30', '-q:v': '2'})
    print("Generating video : ", save_file, end = '\n')

    counter = 0
    for frame in clip:
        frame = np.array(frame * 255.0).astype(np.uint8)
        if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
            continue
        overlay = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
        overlay[mask > np.mean(mask)] = [0, 0, 255]
        frame_img = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        writer.writeFrame(frame_img)
        counter += 1
    writer.close()
    

def test(saved_model, use_cuda, save_file):

    validation_data_generator = DataGenerator('train')
    validation_dataloader = DataLoader(validation_data_generator, batch_size=params.batch_size, shuffle=True, num_workers=4, drop_last=True)

    model = CombinedModel()
    model.load_state_dict(torch.load(saved_model)['state_dict'])

    model.eval()
    if use_cuda:
        model.cuda()

    (anomaly_clips, normal_clips) = next(iter(validation_dataloader))
    anomaly_clips = np.array(anomaly_clips, dtype='f')
    normal_clips = np.array(normal_clips, dtype='f')

    anomaly_clips = np.transpose(anomaly_clips, (0, 4, 1, 2, 3))
    normal_clips = np.transpose(normal_clips, (0, 4, 1, 2, 3))

    if use_cuda:
        anomaly_clips = Variable(torch.from_numpy(anomaly_clips)).cuda()
        normal_clips = Variable(torch.from_numpy(normal_clips)).cuda()
    else:
        anomaly_clips = Variable(torch.from_numpy(anomaly_clips))
        normal_clips = Variable(torch.from_numpy(normal_clips))

    positive_bag = model(anomaly_clips)
    negative_bag = model(normal_clips)

    anomaly_clip = anomaly_clips[0]
    anomaly_mask = positive_bag[0]

    anomaly_clip = np.transpose(anomaly_clip.data.numpy(), (1, 2, 3, 0))
    visualize_clip(anomaly_clip, anomaly_mask.data.numpy(), save_file)


if __name__ == '__main__':
    saved_model = '/home/praveen/CVPR20/UCF_Crimes_I3D/results/saved_models/run6/model_30.pth'
    output = '/home/praveen/CVPR20/UCF_Crimes_I3D/results/outputs/train_run6_model_30.mp4'
    test(saved_model, False, output)
