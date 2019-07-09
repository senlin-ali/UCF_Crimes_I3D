import argparse
import configuration as cfg
import parameters as params
from datetime import datetime
from train import *


def train_action_classifier(run_id, use_cuda):
    saved_models_dir = os.path.join(cfg.saved_models_dir, str(run_id))
    train_model(run_id, saved_models_dir, use_cuda)


def main(args):

    print("Run description : %s", args.run_description)

    # call a function depending on the 'mode' parameter
    if args.train_classifier:
        run_id = args.run_id + '_' + datetime.today().strftime('%d-%m-%y_%H%M')
        use_cuda = torch.cuda.is_available()
        train_action_classifier(run_id, use_cuda)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to train Anomaly Classifier model')

    # 'mode' parameter (mutually exclusive group) with five modes : train/test classifier, train/test generator, test
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('--train_classifier', dest='train_classifier', action='store_true',
                       help='Training the Classifier for Anomaly Detection')

    parser.add_argument("--gpu", dest='gpu', type=str, required=False,
                        help='Set CUDA_VISIBLE_DEVICES environment variable, optional')
    parser.add_argument('--run_id', dest='run_id', type=str, required=False,
                        help='Please provide an ID for the current run')
    parser.add_argument('--run_description', dest='run_description', type=str, required=False,
                        help='Please description of the run to write to log')

    # parse arguments
    args = parser.parse_args()

    # set environment variables to use GPU-0 by default
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # exit when the mode is 'train_action_classifier' and the parameter 'run_id' is missing
    if args.train_action_classifier:
        if args.run_id is None:
            parser.print_help()
            exit(1)

    main(args)
