import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

import os
import random

from PIL import Image, ImageOps

#import any other libraries you need below this line
import cv2
import torch.nn.functional as F
from torchvision.transforms import ElasticTransform
from torchvision.transforms.functional import rotate, hflip, vflip, adjust_gamma

class Cell_data(Dataset):
    def __init__(self, data_dir, size, train, augment_data, train_test_split=0.8):
        ##########################inputs##################################
        # data_dir(string) - directory of the data#########################
        # size(int) - size of the images you want to use###################
        # train(boolean) - train data or test data#########################
        # train_test_split(float) - the portion of the data for training###
        # augment_data(boolean) - use data augmentation or not#############
        super(Cell_data, self).__init__()
        # initialize the data class
        
        # Directories for images and masks
        image_dir = os.path.join(data_dir, 'scans')
        mask_dir = os.path.join(data_dir, 'labels')

        # List all files in the respective directories
        image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.bmp')])
        mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.bmp')])

        # Split into train and test datasets
        split_idx = int(len(image_files) * train_test_split)
        
        if train:
            self.images = image_files[:split_idx]
            self.masks = mask_files[:split_idx]
        else:
            self.images = image_files[split_idx:]
            self.masks = mask_files[split_idx:]
        
        self.size = size
        self.augment_data = augment_data
        self.train = train

    def _load_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # load the img as greyscale
        image = cv2.resize(image, (self.size, self.size))  # resize the img to our desired size
        image = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # min-max scaling the pixel values
        return torch.from_numpy(image).unsqueeze(0)  # Convert the image to tensor and add channel dimension (H, W) -> (C, H, W)
    
    def _load_mask(self, mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # load the mask as greyscale
        mask = cv2.resize(mask, (self.size, self.size))  # resize the mask to our desired size
        return torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)  # Convert the mask to tensor and add channel dimension (H, W) -> (C, H, W)
    
    def _apply_augmentation(self, image, mask, augment_mode):
        if augment_mode == 1:
            # Flip image horizontally
            image = hflip(image)
            mask = hflip(mask)
        elif augment_mode == 2:
            # Flip image vertically
            image = vflip(image)
            mask = vflip(mask)
        elif augment_mode == 3:
            # Zooming image
            # Calculate the crop size in pixels
            k = random.choice([0.7, 0.8, 0.9])
            crop_size = int(self.size * k)  # Assuming to crop (1-k%)/2 off each side
            center = self.size // 2
            half_crop = crop_size // 2
            # Calculate cropping indices
            top = center - half_crop
            bottom = center + half_crop
            left = center - half_crop
            right = center + half_crop
            # Perform cropping
            image_cropped = image[:, top:bottom, left:right]
            mask_cropped = mask[:, top:bottom, left:right]
            # Perform resizing
            # 'F.interpolate' requires a batch dimension, so the shape should be (B, C, H, W)
            # so we need to unsqueeze it first, and then squeeze it at the end
            image_resized = F.interpolate(image_cropped.unsqueeze(0), size=(self.size, self.size), mode='bilinear', align_corners=False).squeeze(0)
            mask_resized = F.interpolate(mask_cropped.unsqueeze(0), size=(self.size, self.size), mode='nearest').squeeze(0)
            image = image_resized
            mask = mask_resized
        elif augment_mode == 4:
            # Rotate image
            k = random.choice([90, 180, 270])
            image = rotate(image, angle=k)
            mask = rotate(mask, angle=k)
        elif augment_mode == 5:
            # Gamma Correction
            gamma = random.uniform(0.5, 1.5)
            image = adjust_gamma(image, gamma)
        else:
            # Non-rigid transformation
            elastic_transformer = ElasticTransform(alpha=10.0, sigma=10.0)
            image = elastic_transformer(image)
            mask = elastic_transformer(mask)

        return image, mask

    def __getitem__(self, idx):

        # Test set, return the original image
        if not self.train:
            image = self._load_image(self.images[idx])
            mask = self._load_mask(self.masks[idx])
        
        # Train set, return original + 6 augmented images
        else:
            # Calculate actual image index and augmentation mode
            actual_idx = idx // 7
            augment_mode = idx % 7
            image = self._load_image(self.images[actual_idx])
            mask = self._load_mask(self.masks[actual_idx])

            # If augment_mode is not 0 and augment_data is True, apply the augmentation
            if augment_mode != 0 and self.augment_data:
                image, mask = self._apply_augmentation(image, mask, augment_mode)
        
        # Threshold the mask to ensure values are strictly 0.0 and 1.0
        mask = (mask > 0).float()

        return image, mask

    def __len__(self):
        # If it's a train set, multiply by 7, else return the original length for test set
        return len(self.images) * 7 if self.train else len(self.images)

