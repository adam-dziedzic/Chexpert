import numpy as np
from torch.utils.data import Dataset
import cv2
import os
from PIL import Image
from data.imgaug import GetTransforms
from data.utils import transform

np.random.seed(0)


class PleuralEffusionDataset(Dataset):
    def __init__(self, in_csv_path, cfg, mode='train'):
        self.cfg = cfg
        self._label_header = None
        self._image_paths = []
        self._labels = []
        self._mode = mode
        self.dict = {'1.0': 1, '': 0, '-1.0': 2, '0.0': 3}
        with open(cfg.data_path + in_csv_path) as f:
            header = f.readline().strip('\n').split(',')
            assert header[15] == 'Pleural Effusion'
            self._label_header = header[15]
            for line in f:
                fields = line.strip('\n').split(',')
                image_path = fields[0]

                # get full path to the image
                data_path_split = cfg.data_path.split('/')
                image_path_split = image_path.split('/')
                data_path_split = [x for x in data_path_split if x != '']
                image_path_split = [x for x in image_path_split if x != '']
                if data_path_split[-1] == image_path_split[0]:
                    full_image_path = data_path_split[:-1] + image_path_split
                    full_image_path = "/" + "/".join(full_image_path)
                else:
                    full_image_path = cfg.data_path + image_path

                self._image_paths.append(full_image_path)
                assert os.path.exists(full_image_path), full_image_path

                # get the label
                self._labels.append(fields[15])

        self._num_image = len(self._image_paths)

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        image = cv2.imread(self._image_paths[idx], 0)
        image = Image.fromarray(image)
        if self._mode == 'train':
            image = GetTransforms(image, type=self.cfg.use_transforms_type)
        image = np.array(image)
        image = transform(image, self.cfg)
        label = np.array(self._labels[idx]).astype(np.float32)

        path = self._image_paths[idx]

        if self._mode == 'train' or self._mode == 'dev':
            return (image, label)
        elif self._mode == 'test':
            return (image, path)
        elif self._mode == 'heatmap':
            return (image, path, label)
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))
