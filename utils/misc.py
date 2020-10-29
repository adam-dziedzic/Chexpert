import numpy as np
import getpass
import json
from easydict import EasyDict as edict
from datetime import datetime
from sklearn import metrics


def get_auc(classification_type, y_true, y_pred, num_classes=None):
    """
    Compute the AUC (Area Under the receiver operator Curve).
    :param classification_type: the type of classification.
    :param y_true: the true labels.
    :param y_pred: the scores or predicted labels.
    :return: AUC score.
    """
    if classification_type == 'binary':
        fpr, tpr, thresholds = metrics.roc_curve(
            y_true, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
    elif classification_type == 'multi-class':
        auc = metrics.roc_auc_score(
            y_true=y_true,
            y_score=y_pred,
            # one-vs-one, insensitive to class imbalances
            multi_class='ovo',
            average='macro',
            labels=[x for x in range(num_classes)]
        )
    else:
        raise Exception(
            f"Unexpected class_type: {classification_type}.")
    return auc


def count_samples_per_class(dataloader):
    steps = len(dataloader)
    dataiter = iter(dataloader)
    targets = []
    for step in range(steps):
        _, target = next(dataiter)
        targets += list(target.squeeze().squeeze().numpy())
    targets = np.array(targets)
    uniques = np.unique(targets)
    counts = {u: 0 for u in uniques}
    for u in targets:
        counts[u] += 1
    return counts


def get_timestamp():
    dateTimeObj = datetime.now()
    # timestampStr = dateTimeObj.strftime("%Y-%B-%d-(%H:%M:%S.%f)")
    timestampStr = dateTimeObj.strftime("%Y-%m-%d-%H-%M-%S-%f")
    return timestampStr


def get_cfg(cfg_path):
    with open(cfg_path) as f:
        cfg = edict(json.load(f))

    user = getpass.getuser()
    for k, v in cfg.items():
        if '{user}' in str(v):
            cfg[k] = v.replace('{user}', user)
    return cfg


def lr_schedule(lr, lr_factor, epoch_now, lr_epochs):
    """
    Learning rate schedule with respect to epoch
    lr: float, initial learning rate
    lr_factor: float, decreasing factor every epoch_lr
    epoch_now: int, the current epoch
    lr_epochs: list of int, decreasing every epoch in lr_epochs
    return: lr, float, scheduled learning rate.
    """
    count = 0
    for epoch in lr_epochs:
        if epoch_now >= epoch:
            count += 1
            continue

        break

    return lr * np.power(lr_factor, count)


def class_wise_loss_reweighting(beta, samples_per_cls):
    """
     https://towardsdatascience.com/handling-class-imbalanced-data-using-a-loss-specifically-made-for-it-6e58fd65ffab

    :param samples_per_cls: number of samples per class
    :return: weights per class for the loss function
    """
    num_classes = len(samples_per_cls)
    effective_sample_count = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_sample_count)
    # normalize the weights
    weights = weights / np.sum(weights) * num_classes
    return weights
