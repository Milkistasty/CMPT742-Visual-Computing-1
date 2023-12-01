import cv2
import os

def convert_images_to_canny(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, edges)

input_folder = 'C:/Users/Alienware/Desktop/github/MPOSE2021/image/'
output_folder = 'C:/Users/Alienware/Desktop/github/MPOSE2021/canny/'

convert_images_to_canny(input_folder, output_folder)
