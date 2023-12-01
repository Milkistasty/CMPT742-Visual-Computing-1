import cv2
import tkinter as tk
from tkinter import filedialog

def select_files():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_paths = filedialog.askopenfilenames(title='Select Images', filetypes=[('Image Files', '*.png;*.jpg;*.jpeg')])
    
    # Close the root window
    root.quit()
    root.destroy()

    return list(file_paths)

# Select image files
file_paths = select_files()

# Check if any files were selected
if not file_paths:
    print("No files selected")
    exit()

# Read the first image to get dimensions
first_img = cv2.imread(file_paths[0])
if first_img is None:
    print("Error reading the first image")
    exit()

height, width, layers = first_img.shape

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out = cv2.VideoWriter('C:/Users/Alienware/Desktop/VC CMPT742/FinalProject/final_result/selected_frames_from_sd/output_video.mp4', fourcc, 10, (width, height))

# Write each image into the video file
for file_path in file_paths:
    img = cv2.imread(file_path)
    if img is not None:
        out.write(img)

# Release the VideoWriter
out.release()

print("output_video.mp4")