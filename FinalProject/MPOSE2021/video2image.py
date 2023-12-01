import cv2
import os

def extract_and_resize_frames(video_path, output_folder, size=(512, 512)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(video_path):
        if filename.endswith(".avi"):
            video_file = os.path.join(video_path, filename)
            cap = cv2.VideoCapture(video_file)

            frame_count = 1
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # isld_ dataset 480*270, enlarging, so use cv2.INTER_LINEAR
                if filename.startswith("isld_"):
                    interpolation_method = cv2.INTER_LINEAR
                # else shrinking, so use cv2.INTER_AREA
                else:
                    interpolation_method = cv2.INTER_AREA

                resized_frame = cv2.resize(frame, size, interpolation=interpolation_method)

                output_filename = f"{os.path.splitext(filename)[0]}_frame{frame_count}.png"
                output_filepath = os.path.join(output_folder, output_filename)

                cv2.imwrite(output_filepath, resized_frame)
                frame_count += 1

            cap.release()

# def extract_frames(video_path, output_folder):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     for filename in os.listdir(video_path):
#         if filename.endswith(".avi"):
#             video_file = os.path.join(video_path, filename)
#             cap = cv2.VideoCapture(video_file)

#             frame_count = 1
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 output_filename = f"{os.path.splitext(filename)[0]}_frame{frame_count}.png"
#                 output_filepath = os.path.join(output_folder, output_filename)

#                 cv2.imwrite(output_filepath, frame)
#                 frame_count += 1

#             cap.release()

video_path = 'C:/Users/Alienware/Desktop/github/MPOSE2021/video/'  # video folder dir
output_folder = 'C:/Users/Alienware/Desktop/github/MPOSE2021/image/'  # output image dir

extract_and_resize_frames(video_path, output_folder)
# extract_frames(video_path, output_folder)
