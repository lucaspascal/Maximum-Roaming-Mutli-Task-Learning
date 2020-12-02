from torch.utils.data.dataset import Dataset

import os
import torch
import fnmatch
import numpy as np


class Cityscapes(Dataset):
    """
    This file is directly modified from https://pytorch.org/docs/stable/torchvision/datasets.html
    """
    def __init__(self, root, train=True, res='low_res', classes=7):
        self.train = train
        self.root = os.path.expanduser(root)
        self.classes = classes

        # R\read the data file
        if train:
            self.data_path = os.path.join(root, res, 'train')
        else:
            self.data_path = os.path.join(root, res, 'val')

        # calculate data length
        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.npy'))

    def __getitem__(self, index):
        # get image name from the pandas df
        image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0))
        semantic = torch.from_numpy(np.load(self.data_path + '/label_{}/{:d}.npy'.format(self.classes, index)))
        depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0))
        
        h,w = semantic.shape
        semantic_full = torch.FloatTensor(8,h,w)
        semantic_full.zero_()
        semantic_full.scatter_(0, semantic.type(torch.long).unsqueeze(0)+1, 1)
        semantic_full = semantic_full[1:, :, :]
        
        return image.type(torch.FloatTensor), semantic_full, depth.type(torch.FloatTensor)

    def __len__(self):
        return self.data_len

