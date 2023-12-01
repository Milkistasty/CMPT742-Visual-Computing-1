import cv2
import numpy as np
import matplotlib.pyplot as plt

# STEP 1: LOADING THE IMAGES
image_1 = cv2.imread("image_1.png" ,1)
image_2 = cv2.imread("image_2.png" ,1)

# STEP 2: LOADING THE ALGORITHM 
algorithm = cv2.SIFT_create()

kp1, des1 = algorithm.detectAndCompute(image_1, None)
kp2, des2 = algorithm.detectAndCompute(image_2, None)

# STEP 3: FINDING THE MATCHES
detector = cv2.BFMatcher()
matches = detector.knnMatch(des1, des2, k=2)

# STEP 4: SELECTING APPROPRIATE MATCHES  
selected_matches = []
for match1, match2 in matches:
    if match1.distance < 0.8 * match2.distance:
        selected_matches.append(match1)

# STEP 5: DRAWING COLORED MATCHING LINES
result = cv2.drawMatchesKnn(image_1,kp1,image_2,kp2,[selected_matches],None,flags=2)
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# STEP 6: SAVING THE RESULT
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111)
ax.imshow(result)
plt.savefig("part_1.png")
plt.close()
