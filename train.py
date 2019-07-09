from dataloader import *
from model import *
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F


class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, positive_bag, negative_bag):
        loss = 1 - torch.max(positive_bag) + torch.max(negative_bag)
        if loss < 0:
            return 0
        return loss


class CustomLoss_With_Regularization(torch.nn.Module):
    def __init__(self, regularization_weight=0.1):
        super(CustomLoss_With_Regularization, self).__init__()
        self.regularization_weight = regularization_weight

    def forward(self, positive_bag, negative_bag):
        loss =  1 - torch.max(positive_bag) + torch.max(negative_bag)
        positive_bag = positive_bag.view(positive_bag.shape[0], -1)
        regularization_loss = -1.0 * (F.softmax(positive_bag, dim=1) * F.log_softmax(positive_bag, dim=1)).sum()
        loss = loss + (self.regularization_weight * regularization_loss)
        if loss < 0:
            return 0
        return loss


def train_epoch(run_id, epoch, data_loader, model, optimizer, criterion, writer, use_cuda):
    print('train at epoch {}'.format(epoch))

    losses = []

    model.train()
    if use_cuda:
        model.cuda()

    for i, (anomaly_clips, normal_clips) in enumerate(data_loader):
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

        optimizer.zero_grad()

        positive_bag = model(anomaly_clips)
        negative_bag = model(normal_clips)

        loss = criterion(positive_bag, negative_bag)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if i % 10 == 0:
            print("Training Epoch ", epoch , " Batch ", i, "- Loss : ", loss.item())
        del loss

    save_dir = os.path.join(cfg.saved_models_dir, run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if epoch % params.checkpoint == 0:
        save_file_path = os.path.join(save_dir, 'model_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)

    print('Training Epoch: %d, Loss: %.4f' % (epoch,  np.mean(losses)))

    writer.add_scalar('Training Loss', np.mean(losses), epoch)

    return model


def val_epoch(epoch, data_loader, model, criterion, writer, use_cuda):
    print('validation at epoch {}'.format(epoch))

    model.eval()
    if use_cuda:
        model.cuda()

    losses = []

    for i, (anomaly_clips, normal_clips) in enumerate(data_loader):
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

        loss = criterion(positive_bag, negative_bag)

        losses.append(loss.item())

        if i % 10 == 0:
            print("Validation Epoch ", epoch , " Batch ", i, "- Loss : ", loss.item())
        del loss
        del positive_bag, negative_bag

    print('Validation Epoch: %d, Loss: %.4f' % (epoch,  np.mean(losses)))

    writer.add_scalar('Validation Loss', np.mean(losses), epoch)


def train_model(run_id, saved_models_dir, use_cuda):

    writer = SummaryWriter(os.path.join(cfg.tf_logs_dir, str(run_id)))

    print("Run ID : " + run_id)

    print("Parameters used : ")
    print("batch_size: " + str(params.batch_size))
    print("lr: " + str(params.learning_rate))
    print("skip_frames: " + str(params.skip_rate))
    print("frames_per_clip: " + str(params.frames_per_clip))
    print("num_samples: " + str(params.samples_per_epoch))

    train_data_generator = DataGenerator('train')
    train_dataloader = DataLoader(train_data_generator, batch_size=params.batch_size, shuffle=True, num_workers=4, drop_last=True)

    validation_data_generator = DataGenerator('test')
    validation_dataloader = DataLoader(validation_data_generator, batch_size=params.batch_size, shuffle=True, num_workers=4, drop_last=True)

    if not os.path.exists(saved_models_dir):
        os.makedirs(saved_models_dir)

    print("Number of training samples : " + str(len(train_data_generator)))
    print("Number of validation samples : " + str(len(validation_data_generator)))

    steps_per_epoch = len(train_data_generator) / params.batch_size
    validation_steps = len(validation_data_generator) / params.batch_size

    print("Steps per epoch: " + str(steps_per_epoch))
    print("Steps per validation: " + str(validation_steps))

    model = CombinedModel()

    print("Using ADAM optimizer")
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

    criterion = CustomLoss()

    # loop for each epoch
    for epoch in range(params.num_epochs):
        model = train_epoch(run_id, epoch, train_dataloader, model, optimizer, criterion, writer, use_cuda)
        val_epoch(epoch, validation_dataloader, model, criterion, writer, use_cuda)

        data_generator = DataGenerator('train')
        train_dataloader = DataLoader(data_generator, batch_size=params.batch_size, shuffle=True, num_workers=4, drop_last=True)

        data_generator = DataGenerator('test')
        validation_dataloader = DataLoader(data_generator, batch_size=params.batch_size, shuffle=True, num_workers=4, drop_last=True)

