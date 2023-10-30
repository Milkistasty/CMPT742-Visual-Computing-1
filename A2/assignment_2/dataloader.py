import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

import os
import random

from PIL import Image, ImageOps

#import any other libraries you need below this line
from scipy.ndimage.filters import gaussian_filter
from torchvision.transforms.functional import adjust_gamma
import cv2

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
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # load the img
        image = cv2.resize(image, (self.size, self.size))  # resize the img
        image = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # min-max scaling the pixel values
        return Image.fromarray(image.astype(np.float32))  # convert the image to PIL format
    
    def _load_mask(self, mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # load the mask
        mask = cv2.resize(mask, (self.size, self.size))  # resize the mask
        return Image.fromarray(mask.astype(float))  # convert the mask to PIL format

    def _apply_augmentation(self, image, mask, augment_mode):
        if augment_mode == 1:
            # flip image vertically
            image = ImageOps.flip(image)
            mask = ImageOps.flip(mask)
        elif augment_mode == 2:
            # flip image horizontally
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)
        elif augment_mode == 3:
            # zoom image
            left = self.size * 0.1
            top = self.size * 0.1
            right = self.size * 0.9
            bottom = self.size * 0.9
            image = image.crop((left, top, right, bottom)).resize((self.size, self.size))
            mask = mask.crop((left, top, right, bottom)).resize((self.size, self.size))
        elif augment_mode == 4:
            # rotate image
            image = image.rotate(90)
            mask = mask.rotate(90)
        elif augment_mode == 5:
            # Gamma Correction
            gamma = random.uniform(0.5, 1.5)
            image = adjust_gamma(image, gamma)
        else:
            # Non-rigid transformation
            # uses Gaussian filters to generate smooth displacements on a coarse grid
            # Then, it applies these displacements to the entire image
            alpha = 10   # as described in the U-Net paper
            sigma = 10   # as described in the U-Net paper
            # Convert image and mask to tensors
            image = torch.tensor(np.array(image)).float()
            mask = torch.tensor(np.array(mask)).float()

            shape = image.shape
            # Generate random displacements
            random_displacement = torch.rand(*shape) * 2 - 1
            # Used gaussian_filter from scipy to apply Gaussian filtering for non-rigid transformation
            dx = torch.tensor(gaussian_filter(random_displacement.numpy(), sigma)) * alpha
            dy = torch.tensor(gaussian_filter(random_displacement.numpy(), sigma)) * alpha
    
            grid_y, grid_x = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]))
            indices_x = (grid_x + dx).clamp(0, shape[1]-1)
            indices_y = (grid_y + dy).clamp(0, shape[0]-1)
            # Use grid_sample for the non-rigid transformation. It expects a normalized grid in the range of [-1, 1]
            grid = torch.stack((indices_x*2/shape[1] - 1, indices_y*2/shape[0] - 1), dim=-1).unsqueeze(0)
            image = torch.nn.functional.grid_sample(image.unsqueeze(0).unsqueeze(0), grid).squeeze(0).squeeze(0)
            mask = torch.nn.functional.grid_sample(mask.unsqueeze(0).unsqueeze(0), grid).squeeze(0).squeeze(0)
            
            # # After applying the non-rigid transformation, converted tensors back to PIL images for uniformity
            image = Image.fromarray(image.numpy().astype(np.float32))
            mask = Image.fromarray(mask.numpy().astype(float))

        return image, mask

    def __getitem__(self, idx):

        # Test set, always return the original image
        if not self.train:
            image = self._load_image(self.images[idx])
            mask = self._load_mask(self.masks[idx])
        
        # Train set, for each image return original + 6 augmented images
        else:
            # Calculate actual image index and augmentation mode
            actual_idx = idx // 7
            augment_mode = idx % 7
            image = self._load_image(self.images[actual_idx])
            mask = self._load_mask(self.masks[actual_idx])

            # If augment_mode is not 0, apply the corresponding augmentation
            if augment_mode != 0 and self.augment_data:
                image, mask = self._apply_augmentation(image, mask, augment_mode)

        # return image and mask in tensors
        image = torch.tensor(np.array(image, dtype=np.float32)).unsqueeze(0)
        mask = torch.tensor(np.array(mask)).unsqueeze(0)
        # Threshold the mask to ensure values are strictly 0.0 and 1.0
        mask = (mask > 0).float()

        return image, mask

    def __len__(self):
        # If it's a train set, multiply by 7, else return the original length for test set
        return len(self.images) * 7 if self.train else len(self.images)

