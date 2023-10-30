import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

import os
import random

from PIL import Image, ImageOps

#import any other libraries you need below this line
from scipy.ndimage.filters import gaussian_filter

class Cell_data(Dataset):
    def __init__(self, data_dir, size, train, train_test_split=0.8, augment_data=True):
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

    def __getitem__(self, idx):
        # load image and mask from index idx of your data
        image_path = self.images[idx]
        mask_path = self.masks[idx]

        image = Image.open(image_path).resize((self.size, self.size)).convert('L')  # Grayscale
        mask = Image.open(mask_path).resize((self.size, self.size))

        # data augmentation part
        if self.augment_data:
            augment_mode = torch.randint(0, 6, (1,)).item()
            if augment_mode == 0:
                # flip image vertically
                image = ImageOps.flip(image)
                mask = ImageOps.flip(mask)
            elif augment_mode == 1:
                # flip image horizontally
                image = ImageOps.mirror(image)
                mask = ImageOps.mirror(mask)
            elif augment_mode == 2:
                # zoom image
                # Zooming by cropping and resizing
                left = self.size * 0.1
                top = self.size * 0.1
                right = self.size * 0.9
                bottom = self.size * 0.9
                image = image.crop((left, top, right, bottom)).resize((self.size, self.size))
                mask = mask.crop((left, top, right, bottom)).resize((self.size, self.size))
            elif augment_mode == 3:
                # rotate image
                # Rotation by 90 degrees
                image = image.rotate(90)
                mask = mask.rotate(90)
            elif augment_mode == 4:
                # Gamma Correction
                # randomly selects a gamma value between 0.5 and 1.5 
                # and applies the gamma correction formula to the image and mask tensors
                gamma = random.uniform(0.5, 1.5)
                # After applying gamma correction, 
                # converted tensors back to PIL images to allow subsequent augmentation to be applied
                image = Image.fromarray(torch.pow(torch.tensor(np.array(image)), gamma).numpy())
                mask = Image.fromarray(torch.pow(torch.tensor(np.array(mask)), gamma).numpy())
            else:
                # Non-rigid transformation
                # uses Gaussian filters to generate smooth displacements on a coarse grid
                # Then, it applies these displacements to the entire image
                alpha = 10  # as described in the U-Net paper
                sigma = 10  # as described in the U-Net paper
                
                # Convert image and mask to tensors for manipulation
                image_tensor = torch.tensor(np.array(image)).float() / 255.0  # Convert to float and normalize
                mask_tensor = torch.tensor(np.array(mask)).float() / 255.0

                shape = image_tensor.shape
    
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
                image = torch.nn.functional.grid_sample(image_tensor.unsqueeze(0).unsqueeze(0), grid).squeeze(0).squeeze(0)
                mask = torch.nn.functional.grid_sample(mask_tensor.unsqueeze(0).unsqueeze(0), grid).squeeze(0).squeeze(0)

                # After applying the non-rigid transformation, converted tensors back to PIL images for uniformity
                image = Image.fromarray((image.numpy() * 255).astype(np.uint8))  # Convert back to [0, 255]
                mask = Image.fromarray((mask.numpy() * 255).astype(np.uint8))

        # return image and mask in tensors
        # Convert to tensor and normalize
        image = torch.tensor(np.array(image, dtype=np.float32)).unsqueeze(0) / 255.0
        mask = torch.tensor(np.array(mask)).unsqueeze(0)

        # Threshold the mask to ensure values are strictly 0 and 1
        mask = (mask > 0).float()

        return image, mask

    def __len__(self):
        return len(self.images)

