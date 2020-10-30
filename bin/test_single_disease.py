import getpass

import argparse
import logging
import numpy as np
import os
import sys
import time
import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from utils.misc import get_timestamp
from utils.misc import get_cfg

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from data.single_disease import SingleDiseaseDataset  # noqa
from model.classifier import Classifier  # noqa

user = getpass.getuser()
timestamp = get_timestamp()
parser = argparse.ArgumentParser(description='Test model')

parser.add_argument('--model_path', default='../config/', metavar='MODEL_PATH',
                    type=str, help="Path to the trained models")
parser.add_argument(
    '--in_csv_path',
    # default=f'/home/{user}/data/CheXpert-v1.0-small/valid.csv',
    default='valid',
    # default='train.csv',
    metavar='IN_CSV_PATH',
    type=str,
    help="Path to the csv file about the data (the metadata file).")
parser.add_argument('--data_path', default=f'/home/{user}/data/',
                    metavar='DATA_PATH',
                    type=str, help="Path to the data set")
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

    # test_header = [
    #     'Path',
    #     'Cardiomegaly',
    #     'Edema',
    #     'Consolidation',
    #     'Atelectasis',
    #     'Pleural Effusion']

    test_header = ['Path', cfg.disease]

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


def run(args, cfg):
    device_ids = list(map(int, args.device_ids.split(',')))
    num_devices = torch.cuda.device_count()
    if num_devices < len(device_ids):
        raise Exception(
            '#available gpu : {} < --device_ids : {}'
                .format(num_devices, len(device_ids)))
    device = torch.device('cuda:{}'.format(device_ids[0]))

    model = Classifier(cfg)
    model = DataParallel(model, device_ids=device_ids).to(device).eval()
    model_name = f"../bin/save-single-disease-{cfg.disease}/best3.ckpt"
    ckpt_path = os.path.join(args.model_path, model_name)
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'state_dict' in ckpt:
        model.module.load_state_dict(ckpt['state_dict'])
    else:
        model.module.load_state_dict(ckpt)

    dataloader_test = DataLoader(
        SingleDiseaseDataset(
            # in_csv_path=cfg.dev_csv,
            in_csv_path=args.in_csv_path + '.csv',
            cfg=cfg,
            mode='test',
        ),
        batch_size=cfg.dev_batch_size, num_workers=args.num_workers,
        drop_last=False, shuffle=False)

    folder = f'test-{cfg.disease}-{args.in_csv_path}/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    out_csv_path = folder + 'test.csv'

    test_epoch(cfg, args, model, dataloader_test, out_csv_path)

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
    cfg = get_cfg(args.model_path + 'single_disease_small.json')
    diseases = ['Pleural Effusion', 'Cardiomegaly', 'Consolidation',
                'Atelectasis', 'Edema']
    for disease in diseases:
        print('disease: ', disease)
        cfg.disease = disease
        run(args=args, cfg=cfg)


if __name__ == '__main__':
    main()
