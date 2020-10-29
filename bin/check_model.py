import getpass

import argparse
import logging
import os
import sys
import torch
from torch.nn import DataParallel

from utils.misc import get_cfg

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from data.dataset import ImageDataset  # noqa
from model.classifier import Classifier  # noqa

user = getpass.getuser()
parser = argparse.ArgumentParser(description='Test model')

parser.add_argument(
    '--model_path', default='../config/',
    metavar='MODEL_PATH',
    type=str, help="Path to the trained models")
parser.add_argument(
    '--cfg_name',
    default=''
)
parser.add_argument(
    '--model_name',
    # default='ady_small_pre_train.pth',
    # default='../bin/save-2020-10-27-train-from-scratch-chexpert-small-densenet/best1.ckpt',
    # default='../bin/save-2020-10-27-train-from-scratch-chexpert-small-densenet/best2.ckpt',
    # default='../bin/save-2020-10-27-train-from-scratch-chexpert-small-densenet/best3.ckpt',
    default='../bin/save-2020-10-28-23-16-51-687233/best1.ckpt',
    metavar='MODEL_NAME',
    type=str, help="Name of the trained model.")
parser.add_argument('--device_ids', default='0,1,2,3', type=str,
                    help="GPU indices comma separated, e.g. '0,1' ")


def run(args):
    cfg = get_cfg(args.model_path + 'ady_small.json')

    device_ids = list(map(int, args.device_ids.split(',')))
    num_devices = torch.cuda.device_count()
    if num_devices < len(device_ids):
        raise Exception(
            '#available gpu : {} < --device_ids : {}'
                .format(num_devices, len(device_ids)))
    device = torch.device('cuda:{}'.format(device_ids[0]))

    model = Classifier(cfg)
    model = DataParallel(model, device_ids=device_ids).to(device).eval()
    ckpt_path = os.path.join(args.model_path, args.model_name)
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'state_dict' in ckpt:
        model.module.load_state_dict(ckpt['state_dict'])
    else:
        model.module.load_state_dict(ckpt)

    if isinstance(ckpt, dict):
        for key in ckpt.keys():
            if key not in ['state_dict', '__len__']:
                print(f'Saved best {key}:', ckpt[key])


def main():
    logging.basicConfig(
        # level=logging.INFO,
        level=logging.ERROR,
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
