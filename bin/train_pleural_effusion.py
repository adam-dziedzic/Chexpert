import argparse
import json
import logging
import numpy as np
import os
import subprocess
import sys
import time
import torch
import torch.nn.functional as F
from shutil import copyfile
from sklearn import metrics
from tensorboardX import SummaryWriter
from torch.nn import DataParallel
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from data.pleural_effusion import PleuralEffusionDataset  # noqa
from model.classifier import Classifier  # noqa
from utils.misc import get_cfg, class_wise_loss_reweighting  # noqa
from utils.misc import get_timestamp
from utils.misc import get_auc
from utils.misc import lr_schedule  # noqa
from model.utils import get_optimizer  # noqa

import getpass

user = getpass.getuser()

timestamp = get_timestamp()

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--cfg_path',
                    # default='../config/ady_small.json',
                    default='../config/pleural_effusion_small.json',
                    metavar='CFG_PATH', type=str,
                    help="Path to the config file in json format")
parser.add_argument('--save_path', default=f'./save-{timestamp}',
                    metavar='SAVE_PATH',
                    type=str,
                    help="Path to the saved models")
parser.add_argument('--num_workers', default=8, type=int,
                    help="Number of workers for each data loader")
parser.add_argument('--device_ids', default='0,1,2,3', type=str,
                    help="GPU indices ""comma separated, e.g. '0,1' ")
parser.add_argument('--pre_train',
                    default=None, type=str,
                    help="If get parameters from pretrained model")
parser.add_argument('--resume', default=0, type=int,
                    help="If resume from previous run")
parser.add_argument('--logtofile', default=True, type=bool,
                    help="Save log in save_path/log.txt if set True")
parser.add_argument('--verbose',
                    default=True,
                    type=bool,
                    help="Detail info")


def get_loss(output, target, index, device, cfg):
    target = target[:, index]
    output = output[index]

    if cfg.criterion == 'BCE':
        if cfg.batch_weight:
            weight = (target.size()[0] - target.sum()) / target.sum()
        else:
            pos_weight = torch.from_numpy(
                np.array(cfg.pos_weight,
                         dtype=np.float32)).to(device).type_as(target)
            weight = pos_weight[index]

        for num_class in cfg.num_classes:
            assert num_class == 1
        if target.sum() == 0:
            loss = torch.tensor(0., requires_grad=True).to(device)
        else:
            loss = F.binary_cross_entropy_with_logits(
                output, target,
                pos_weight=weight)
        pred = torch.sigmoid(output).ge(0.5).float()
    elif cfg.criterion == 'CE':
        assert len(cfg.num_classes) == 1
        assert cfg.num_classes[0] == 4
        if cfg.weight_beta == -1.0:
            loss = F.cross_entropy(input=output, target=target)
        else:
            loss = F.cross_entropy(input=output, target=target,
                                   weight=torch.tensor(
                                       cfg.class_weights,
                                       device=output.device,
                                       dtype=output.dtype,
                                   )
                                   )
        pred = output.argmax(dim=1)
    else:
        raise Exception(f'Unknown criterion: {cfg.criterion}')

    acc = pred.eq(target).float().sum() / len(target)
    balanced_acc = metrics.balanced_accuracy_score(
        y_true=target.detach().cpu().numpy(),
        y_pred=pred.detach().cpu().numpy(),
    )
    return (loss, acc, balanced_acc)


def set_class_weights(cfg, dataset):
    train_class_count = dataset.sample_counts_per_class()
    samples_per_cls = list(train_class_count.values())
    if cfg.weight_beta == -1:
        class_weights = np.ones_like(samples_per_cls)
    else:
        class_weights = class_wise_loss_reweighting(
            beta=cfg.weight_beta,
            samples_per_cls=samples_per_cls,
        )
    cfg.class_weights = class_weights
    return class_weights


def train_epoch(summary, summary_dev, cfg, args, model, dataloader,
                dataloader_dev, optimizer, summary_writer, best_dict,
                dev_header):
    torch.set_grad_enabled(True)
    model.train()
    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device('cuda:{}'.format(device_ids[0]))
    if cfg.train_steps == 0:
        steps = len(dataloader)
    else:
        steps = cfg.train_steps
    print('steps: ', steps)
    dataiter = iter(dataloader)
    label_header = dataloader.dataset._label_header
    num_tasks = len(cfg.num_classes)

    time_now = time.time()
    loss_sum = np.zeros(num_tasks)
    acc_sum = np.zeros(num_tasks)
    balanced_acc_sum = np.zeros(num_tasks)
    for step in range(steps):
        image, target = next(dataiter)
        image = image.to(device)
        target = target.to(device)
        output, logit_map = model(image)

        # different number of tasks
        loss = 0
        for t in range(num_tasks):
            loss_t, acc_t, balanced_acc_t = get_loss(
                output, target, t, device, cfg)
            loss += loss_t
            loss_sum[t] += loss_t.item()
            acc_sum[t] += acc_t.item()
            balanced_acc_sum[t] += balanced_acc_t

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        summary['step'] += 1

        if summary['step'] % cfg.log_every == 0:
            time_spent = time.time() - time_now
            time_now = time.time()

            loss_sum /= cfg.log_every
            acc_sum /= cfg.log_every
            balanced_acc_sum /= cfg.log_every
            loss_str = ' '.join(map(lambda x: '{:.5f}'.format(x), loss_sum))
            acc_str = ' '.join(map(lambda x: '{:.3f}'.format(x), acc_sum))
            balanced_acc_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                            balanced_acc_sum))

            logging.info(
                '{}, Train, Epoch : {}, Step : {}, Loss : {}, '
                'Acc : {}, Balanced Acc : {}, Run Time : {:.2f} sec'
                    .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                            summary['epoch'] + 1, summary['step'], loss_str,
                            acc_str, balanced_acc_str, time_spent))

            for t in range(num_tasks):
                summary_writer.add_scalar(
                    'train/loss_{}'.format(label_header[t]), loss_sum[t],
                    summary['step'])
                summary_writer.add_scalar(
                    'train/acc_{}'.format(label_header[t]), acc_sum[t],
                    summary['step'])
                summary_writer.add_scalar(
                    'train/balanced_acc_{}'.format(label_header[t]),
                    balanced_acc_sum[t],
                    summary['step'])

            loss_sum = np.zeros(num_tasks)
            acc_sum = np.zeros(num_tasks)
            balanced_acc_sum = np.zeros(num_tasks)

        if summary['step'] % cfg.test_every == 0:
            time_now = time.time()
            summary_dev, predlist, true_list = test_epoch(
                summary_dev, cfg, args, model, dataloader_dev)
            time_spent = time.time() - time_now

            auclist = []
            for i in range(len(cfg.num_classes)):
                y_pred = predlist[i]
                y_true = true_list[i]

                auc = get_auc(
                    classification_type=cfg.classification_type,
                    y_true=y_true,
                    y_pred=y_pred,
                    num_classes=cfg.num_classes[i],
                )

                auclist.append(auc)
            summary_dev['auc'] = np.array(auclist)

            loss_dev_str = ' '.join(map(lambda x: '{:.5f}'.format(x),
                                        summary_dev['loss']))
            acc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                       summary_dev['acc']))
            balanced_acc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                                summary_dev['balanced_acc']))
            auc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                       summary_dev['auc']))

            logging.info(
                '{}, Dev, '
                'Step : {}, '
                'Loss : {}, '
                'Acc : {}, '
                'Balanced Acc : {}, '
                'Auc : {},'
                'Mean auc: {:.3f}, '
                'Run Time : {:.2f} sec'.format(
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    summary['step'],
                    loss_dev_str,
                    acc_dev_str,
                    balanced_acc_dev_str,
                    auc_dev_str,
                    summary_dev['auc'].mean(),
                    time_spent))

            for t in range(len(cfg.num_classes)):
                summary_writer.add_scalar(
                    'dev/loss_{}'.format(dev_header[t]),
                    summary_dev['loss'][t], summary['step'])
                summary_writer.add_scalar(
                    'dev/acc_{}'.format(dev_header[t]), summary_dev['acc'][t],
                    summary['step'])
                summary_writer.add_scalar(
                    'dev/balanced_acc_{}'.format(dev_header[t]),
                    summary_dev['balanced_acc'][t],
                    summary['step'])
                summary_writer.add_scalar(
                    'dev/auc_{}'.format(dev_header[t]), summary_dev['auc'][t],
                    summary['step'])

            save_best = False

            mean_acc = summary_dev['acc'][cfg.save_index].mean()
            if mean_acc >= best_dict['acc_dev_best']:
                best_dict['acc_dev_best'] = mean_acc
                if cfg.best_target == 'acc':
                    save_best = True

            mean_balanced_acc = summary_dev['balanced_acc'][
                cfg.save_index].mean()
            if mean_balanced_acc >= best_dict['balanced_acc_dev_best']:
                best_dict['balanced_acc_dev_best'] = mean_balanced_acc
                if cfg.best_target == 'balanced_acc':
                    save_best = True

            mean_auc = summary_dev['auc'][cfg.save_index].mean()
            if mean_auc >= best_dict['auc_dev_best']:
                best_dict['auc_dev_best'] = mean_auc
                if cfg.best_target == 'auc':
                    save_best = True

            mean_loss = summary_dev['loss'][cfg.save_index].mean()
            if mean_loss <= best_dict['loss_dev_best']:
                best_dict['loss_dev_best'] = mean_loss
                if cfg.best_target == 'loss':
                    save_best = True

            if save_best:
                torch.save(
                    {'epoch': summary['epoch'],
                     'step': summary['step'],
                     'acc_dev_best': best_dict['acc_dev_best'],
                     'balanced_acc_dev_best': best_dict[
                         'balanced_acc_dev_best'],
                     'auc_dev_best': best_dict['auc_dev_best'],
                     'loss_dev_best': best_dict['loss_dev_best'],
                     'state_dict': model.module.state_dict()},
                    os.path.join(args.save_path, 'best{}.ckpt'.format(
                        best_dict['best_idx']))
                )
                best_dict['best_idx'] += 1
                if best_dict['best_idx'] > cfg.save_top_k:
                    best_dict['best_idx'] = 1
                logging.info(
                    '{}, Best, '
                    'Step : {}, '
                    'Loss : {}, '
                    'Acc : {}, '
                    'Balanced Acc : {}, '
                    'Auc :{}, '
                    'Best Auc : {:.3f}'.format(
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        summary['step'],
                        loss_dev_str,
                        acc_dev_str,
                        balanced_acc_dev_str,
                        auc_dev_str,
                        best_dict['auc_dev_best']))
        model.train()
        torch.set_grad_enabled(True)
    summary['epoch'] += 1

    return summary, best_dict


def test_epoch(summary, cfg, args, model, dataloader):
    torch.set_grad_enabled(False)
    model.eval()
    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device('cuda:{}'.format(device_ids[0]))
    steps = len(dataloader)
    dataiter = iter(dataloader)
    num_tasks = len(cfg.num_classes)

    loss_sum = np.zeros(num_tasks)
    acc_sum = np.zeros(num_tasks)
    balanced_acc_sum = np.zeros(num_tasks)

    predlist = list(x for x in range(len(cfg.num_classes)))
    true_list = list(x for x in range(len(cfg.num_classes)))
    for step in range(steps):
        image, target = next(dataiter)
        image = image.to(device)
        target = target.to(device)
        output, logit_map = model(image)
        # different number of tasks
        for t in range(len(cfg.num_classes)):

            loss_t, acc_t, balanced_acc_t = get_loss(
                output, target, t, device, cfg)
            # AUC
            if cfg.criterion == 'BCE':
                output_tensor = torch.sigmoid(
                    output[t].view(-1)).cpu().detach().numpy()
            elif cfg.criterion == 'CE':
                output_tensor = torch.softmax(
                    output[t], dim=1)
                output_tensor = output_tensor.cpu().detach().numpy()
            else:
                raise Exception(f"Unknown criterion: {cfg.criterion}")
            target_tensor = target[:, t].view(-1).cpu().detach().numpy()
            if step == 0:
                predlist[t] = output_tensor
                true_list[t] = target_tensor
            else:
                predlist[t] = np.concatenate(
                    (predlist[t], output_tensor),
                    axis=0)
                true_list[t] = np.append(true_list[t], target_tensor)

            loss_sum[t] += loss_t.item()
            acc_sum[t] += acc_t.item()
            balanced_acc_sum[t] += balanced_acc_t
    summary['loss'] = loss_sum / steps
    summary['acc'] = acc_sum / steps
    summary['balanced_acc'] = balanced_acc_sum / steps

    return summary, predlist, true_list


def run(args):
    cfg = get_cfg(args.cfg_path)
    if args.verbose is True:
        print(json.dumps(cfg, indent=4))

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if args.logtofile is True:
        logging.basicConfig(filename=args.save_path + '/log.txt',
                            filemode="w", level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)

    if not args.resume:
        with open(os.path.join(args.save_path, 'cfg.json'), 'w') as f:
            json.dump(cfg, f, indent=1)

    device_ids = list(map(int, args.device_ids.split(',')))
    num_devices = torch.cuda.device_count()
    if num_devices < len(device_ids):
        raise Exception(
            '#available gpu : {} < --device_ids : {}'
                .format(num_devices, len(device_ids)))
    device = torch.device('cuda:{}'.format(device_ids[0]))

    model = Classifier(cfg)
    if args.verbose is True:
        from torchsummary import summary
        if cfg.fix_ratio:
            h, w = cfg.long_side, cfg.long_side
        else:
            h, w = cfg.height, cfg.width
        summary(model.to(device), (3, h, w))
    model = DataParallel(model, device_ids=device_ids).to(device).train()
    if args.pre_train is not None:
        if os.path.exists(args.pre_train):
            ckpt = torch.load(args.pre_train, map_location=device)
            model.module.load_state_dict(ckpt)
    optimizer = get_optimizer(model.parameters(), cfg)

    copy_src_folder = False
    if copy_src_folder:
        src_folder = os.path.dirname(os.path.abspath(__file__)) + '/../'
        dst_folder = os.path.join(args.save_path, 'classification')
        rc, size = subprocess.getstatusoutput(
            'du --max-depth=0 %s | cut -f1' % src_folder)
        if rc != 0:
            raise Exception('Copy folder error : {}'.format(rc))
        print('size: ', size)
        rc, err_msg = subprocess.getstatusoutput(
            'cp -R %s %s' % (src_folder, dst_folder))
        if rc != 0:
            raise Exception('copy folder error : {}'.format(err_msg))

    copyfile(cfg.data_path + cfg.train_csv,
             os.path.join(args.save_path, 'train.csv'))
    copyfile(cfg.data_path + cfg.dev_csv,
             os.path.join(args.save_path, 'dev.csv'))

    dataset_train = PleuralEffusionDataset(
        in_csv_path=cfg.train_csv, cfg=cfg,
        mode='train')
    cfg.class_weights = set_class_weights(cfg=cfg, dataset=dataset_train)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=cfg.train_batch_size, num_workers=args.num_workers,
        drop_last=True, shuffle=True)
    dataset_dev = PleuralEffusionDataset(
        in_csv_path=cfg.dev_csv, cfg=cfg, mode='dev')
    dataloader_dev = DataLoader(
        dataset_dev,
        batch_size=cfg.dev_batch_size, num_workers=args.num_workers,
        drop_last=False, shuffle=False)
    dev_header = dataloader_dev.dataset._label_header

    summary_train = {'epoch': 0, 'step': 0}
    summary_dev = {'loss': float('inf'), 'acc': 0.0}
    summary_writer = SummaryWriter(args.save_path)
    epoch_start = 0
    best_dict = {
        "acc_dev_best": 0.0,
        "balanced_acc_dev_best": 0.0,
        "auc_dev_best": 0.0,
        "loss_dev_best": float('inf'),
        "fused_dev_best": 0.0,
        "best_idx": 1
    }

    if args.resume:
        ckpt_path = os.path.join(args.save_path, 'train.ckpt')
        ckpt = torch.load(ckpt_path, map_location=device)
        model.module.load_state_dict(ckpt['state_dict'])
        summary_train = {'epoch': ckpt['epoch'], 'step': ckpt['step']}
        best_dict['acc_dev_best'] = ckpt['acc_dev_best']
        best_dict['balanced_acc_dev_best'] = ckpt['balanced_acc_dev_best']
        best_dict['loss_dev_best'] = ckpt['loss_dev_best']
        best_dict['auc_dev_best'] = ckpt['auc_dev_best']
        epoch_start = ckpt['epoch']

    for epoch in range(epoch_start, cfg.epoch):
        lr = lr_schedule(cfg.lr, cfg.lr_factor, summary_train['epoch'],
                         cfg.lr_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        summary_train, best_dict = train_epoch(
            summary_train, summary_dev, cfg, args, model,
            dataloader_train, dataloader_dev, optimizer,
            summary_writer, best_dict, dev_header)

        time_now = time.time()
        summary_dev, predlist, true_list = test_epoch(
            summary_dev, cfg, args, model, dataloader_dev)
        time_spent = time.time() - time_now

        auclist = []
        for i in range(len(cfg.num_classes)):
            y_pred = predlist[i]
            y_true = true_list[i]

            auc = get_auc(
                classification_type=cfg.classification_type,
                y_true=y_true,
                y_pred=y_pred,
                num_classes=cfg.num_classes[i],
            )

            auclist.append(auc)
        summary_dev['auc'] = np.array(auclist)

        loss_dev_str = ' '.join(map(lambda x: '{:.5f}'.format(x),
                                    summary_dev['loss']))
        acc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                   summary_dev['acc']))
        balanced_acc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                            summary_dev['balanced_acc']))
        auc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                   summary_dev['auc']))

        logging.info(
            '{}, Dev, '
            'Step : {}, '
            'Loss : {}, '
            'Acc : {}, '
            'Balanced Acc : {}, '
            'Auc : {}, '
            'Mean auc: {:.3f}, '
            'Run Time : {:.2f} sec'.format(
                time.strftime("%Y-%m-%d %H:%M:%S"),
                summary_train['step'],
                loss_dev_str,
                acc_dev_str,
                balanced_acc_dev_str,
                auc_dev_str,
                summary_dev['auc'].mean(),
                time_spent))

        for t in range(len(cfg.num_classes)):
            summary_writer.add_scalar(
                'dev/loss_{}'.format(dev_header[t]), summary_dev['loss'][t],
                summary_train['step'])
            summary_writer.add_scalar(
                'dev/acc_{}'.format(dev_header[t]), summary_dev['acc'][t],
                summary_train['step'])
            summary_writer.add_scalar(
                'dev/balanced_acc_{}'.format(dev_header[t]),
                summary_dev['balanced_acc'][t],
                summary_train['step'])
            summary_writer.add_scalar(
                'dev/auc_{}'.format(dev_header[t]), summary_dev['auc'][t],
                summary_train['step'])

        save_best = False

        mean_acc = summary_dev['acc'][cfg.save_index].mean()
        if mean_acc >= best_dict['acc_dev_best']:
            best_dict['acc_dev_best'] = mean_acc
            if cfg.best_target == 'acc':
                save_best = True

        mean_balanced_acc = summary_dev['balanced_acc'][cfg.save_index].mean()
        if mean_balanced_acc >= best_dict['balanced_acc_dev_best']:
            best_dict['balanced_acc_dev_best'] = mean_balanced_acc
            if cfg.best_target == 'balanced_acc':
                save_best = True

        mean_auc = summary_dev['auc'][cfg.save_index].mean()
        if mean_auc >= best_dict['auc_dev_best']:
            best_dict['auc_dev_best'] = mean_auc
            if cfg.best_target == 'auc':
                save_best = True

        mean_loss = summary_dev['loss'][cfg.save_index].mean()
        if mean_loss <= best_dict['loss_dev_best']:
            best_dict['loss_dev_best'] = mean_loss
            if cfg.best_target == 'loss':
                save_best = True

        if save_best:
            torch.save(
                {'epoch': summary_train['epoch'],
                 'step': summary_train['step'],
                 'acc_dev_best': best_dict['acc_dev_best'],
                 'balanced_acc_dev_best': best_dict['balanced_acc_dev_best'],
                 'auc_dev_best': best_dict['auc_dev_best'],
                 'loss_dev_best': best_dict['loss_dev_best'],
                 'state_dict': model.module.state_dict()},
                os.path.join(args.save_path,
                             'best{}.ckpt'.format(best_dict['best_idx']))
            )
            best_dict['best_idx'] += 1
            if best_dict['best_idx'] > cfg.save_top_k:
                best_dict['best_idx'] = 1
            logging.info(
                '{}, Best, '
                'Step : {}, '
                'Loss : {}, '
                'Acc : {}, '
                'Balanced Acc : {}, '
                'Auc : {}, '
                'Best Auc : {:.3f}'.format(
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    summary_train['step'],
                    loss_dev_str,
                    acc_dev_str,
                    balanced_acc_dev_str,
                    auc_dev_str,
                    best_dict['auc_dev_best']))
        torch.save({'epoch': summary_train['epoch'],
                    'step': summary_train['step'],
                    'acc_dev_best': best_dict['acc_dev_best'],
                    'balanced_acc_dev_best': best_dict['balanced_acc_dev_best'],
                    'auc_dev_best': best_dict['auc_dev_best'],
                    'loss_dev_best': best_dict['loss_dev_best'],
                    'state_dict': model.module.state_dict()},
                   os.path.join(args.save_path, 'train.ckpt'))
    summary_writer.close()


def main():
    args = parser.parse_args()
    if args.verbose is True:
        print('Using the specified args:')
        print(args)

    run(args)


if __name__ == '__main__':
    start_time = time.time()
    main()
    stop_time = time.time()
    print('elapsed time: ', stop_time - start_time)
