import os
import sys
import argparse
import logging
import json
import time
from easydict import EasyDict as edict
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch.nn.functional as F
import getpass
from utils.misc import get_cfg

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from data.dataset import ImageDataset  # noqa
from model.classifier import Classifier  # noqa

user = getpass.getuser()
parser = argparse.ArgumentParser(description='Test model')

parser.add_argument('--model_path', default='../config/', metavar='MODEL_PATH',
                    type=str, help="Path to the trained models")
parser.add_argument('--data_path', default=f'/home/{user}/data/',
                    metavar='DATA_PATH',
                    type=str, help="Path to the data set")
parser.add_argument('--out_csv_path', default='test/test.csv',
                    metavar='OUT_CSV_PATH', type=str,
                    help="Path to the output predictions in csv")
parser.add_argument('--num_workers', default=8, type=int,
                    help="Number of workers for each data loader")
parser.add_argument('--device_ids', default='0,1,2,3', type=str,
                    help="GPU indices comma separated, e.g. '0,1' ")

if not os.path.exists('test'):
    os.mkdir('test')


def get_pred(output, cfg):
    # BCE - Binary Cross-Entropy
    if cfg.criterion == 'BCE' or cfg.criterion == "FL":
        for num_class in cfg.num_classes:
            assert num_class == 1
        pred = torch.sigmoid(output.view(-1)).cpu().detach().numpy()
    elif cfg.criterion == 'CE':
        for num_class in cfg.num_classes:
            assert num_class >= 2
        prob = F.softmax(output)
        pred = prob[:, 1].cpu().detach().numpy()
    else:
        raise Exception('Unknown criterion : {}'.format(cfg.criterion))

    return pred


def test_epoch(cfg, args, model, dataloader, out_csv_path):
    torch.set_grad_enabled(False)
    model.eval()
    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device('cuda:{}'.format(device_ids[0]))
    steps = len(dataloader)
    dataiter = iter(dataloader)
    num_tasks = len(cfg.num_classes)

    test_header = [
        'Path',
        'Cardiomegaly',
        'Edema',
        'Consolidation',
        'Atelectasis',
        'Pleural Effusion']

    with open(out_csv_path, 'w') as f:
        f.write(','.join(test_header) + '\n')
        for step in range(steps):
            image, path = next(dataiter)
            image = image.to(device)
            output, __ = model(image)
            batch_size = len(path)
            pred = np.zeros((num_tasks, batch_size))

            for i in range(num_tasks):
                pred[i] = get_pred(output[i], cfg)

            for i in range(batch_size):
                batch = ','.join(map(lambda x: '{}'.format(x), pred[:, i]))
                result = path[i] + ',' + batch
                f.write(result + '\n')
                logging.info('{}, Image : {}, Prob : {}'.format(
                    time.strftime("%Y-%m-%d %H:%M:%S"), path[i], batch))


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
    ckpt_path = os.path.join(args.model_path, 'ady_small_pre_train.pth')
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'state_dict' in ckpt:
        model.module.load_state_dict(ckpt['state_dict'])
    else:
        model.module.load_state_dict(ckpt)

    dataloader_test = DataLoader(
        ImageDataset(in_csv_path=cfg.dev_csv, cfg=cfg, mode='test'),
        batch_size=cfg.dev_batch_size, num_workers=args.num_workers,
        drop_last=False, shuffle=False)

    test_epoch(cfg, args, model, dataloader_test, args.out_csv_path)

    if 'step' in ckpt:
        print('Save best is step :', ckpt['step'])
    if 'auc_dev_best' in ckpt:
        print('Save bset is AUC :', ckpt['auc_dev_best'])


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
