from PIL import Image
import numpy as np
import cv2
import os

def compute_normal_map(height_map, scale=1.0):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    dx = cv2.filter2D(height_map, -1, sobel_x)
    dy = cv2.filter2D(height_map, -1, sobel_y)
    dz = np.ones_like(height_map) * scale

    normal_map = np.dstack((dx, dy, dz))
    norm = np.linalg.norm(normal_map, axis=2)
    normal_map = normal_map / np.expand_dims(norm, 2)
    normal_map = (normal_map + 1) / 2 * 255
    return normal_map.astype(np.uint8)

def convert_images_to_normal_maps(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            height_map = np.array(Image.open(img_path).convert("L"))
            normal_map = compute_normal_map(height_map)

            output_path = os.path.join(output_folder, filename)
            Image.fromarray(normal_map).save(output_path)

input_folder = 'C:/Users/Alienware/Desktop/github/MPOSE2021/image/'
output_folder = 'C:/Users/Alienware/Desktop/github/MPOSE2021/normalmap/'

convert_images_to_normal_maps(input_folder, output_folder)
