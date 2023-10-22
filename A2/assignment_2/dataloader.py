import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

import os
import random

from PIL import Image, ImageOps

#import any other libraries you need below this line

class Cell_data(Dataset):
    def __init__(self, data_dir, size, train='True', train_test_split=0.8, augment_data=True):
        ##########################inputs##################################
        # data_dir(string) - directory of the data#########################
        # size(int) - size of the images you want to use###################
        # train(boolean) - train data or test data#########################
        # train_test_split(float) - the portion of the data for training###
        # augment_data(boolean) - use data augmentation or not#############
        super(Cell_data, self).__init__()
        # todo
        # initialize the data class

    def __getitem__(self, idx):
        # todo

        # load image and mask from index idx of your data

        # data augmentation part
        if self.augment_data:
            augment_mode = np.random.randint(0, 4)
            if augment_mode == 0:
                # todo
                # flip image vertically
            elif augment_mode == 1:
                # todo
                # flip image horizontally
            elif augment_mode == 2:
                # todo
                # zoom image
            else:
                # todo
                # rotate image

        # todo
        # return image and mask in tensors

    def __len__(self):
        return len(self.images)

