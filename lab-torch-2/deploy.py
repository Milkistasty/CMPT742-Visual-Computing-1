import cv2
import torch
import torchvision.transforms as transforms
import time
import sys
import torch.nn as nn
import torch.nn.functional as F

# Load the scripted model
model = torch.jit.load(sys.argv[1])
model.eval()

square_crop_size = 1080

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(square_crop_size),  # Center crop to a square
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Initialize webcam
cap = cv2.VideoCapture(1)

prev_frame_time = 0
new_frame_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop the frame to a square
    height, width, _ = frame.shape
    if height > width:
        frame = frame[(height - width) // 2:(height + width) // 2, :]
    else:
        frame = frame[:, (width - height) // 2:(width + height) // 2]

    # Time when we finish processing for this frame
    new_frame_time = time.time()

    # Preprocess the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = transform(frame).unsqueeze(0)

    # Model prediction
    with torch.no_grad():
        outputs = model(frame_tensor)
        _, predicted = torch.max(outputs, 1)

    # Add prediction text to frame
    text = 'Face' if predicted.item() == 1 else 'No Face'
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Calculate FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps_text = f'FPS: {int(fps)}'
    cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Webcam', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
