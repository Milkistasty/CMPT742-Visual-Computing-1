import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

def find_matching_keypoints(image1, image2):
    #Input: two images (numpy arrays)
    #Output: two lists of corresponding keypoints (numpy arrays of shape (N, 2))
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(image1, None)
    kp2, desc2 = sift.detectAndCompute(image2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    good = []
    pts1 = []
    pts2 = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    return pts1, pts2

def drawlines(img1,img2,lines,pts1,pts2):
    #img1: image on which we draw the epilines for the points in img2
    #lines: corresponding epilines
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def normalize_points(pts):
    # Compute the centroid of the points
    centroid = np.mean(pts, axis=0)
    
    # Shift the origin of the points to the centroid
    shifted_pts = pts - centroid

    # Compute the mean distance of all points from the centroid
    mean_dist = np.mean(np.sqrt(np.sum(shifted_pts**2, axis=1)))

    # Scale the mean distance to sqrt(2)
    scale = np.sqrt(2) / mean_dist

    # Construct the normalization matrix
    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1]])

    # Normalize the points
    pts = T @ np.vstack((pts.T, np.ones(pts.shape[0])))
    return pts[0:2].T, T

def FindFundamentalMatrix(pts1, pts2):
    #Input: two lists of corresponding keypoints (numpy arrays of shape (N, 2))
    #Output: fundamental matrix (numpy array of shape (3, 3))

    """
    The matrix settings and general structure refers to CMU slides
    https://www.cs.cmu.edu/~16385/s17/Slides/12.4_8Point_Algorithm.pdf

    The code refers to GaTech, Project 3: Camera Calibration and Fundamental Matrix Estimation with RANSAC
    https://sites.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/tguo40/index.html
    
    """

    #todo: Normalize the points
    pts1, T1 = normalize_points(pts1)
    pts2, T2 = normalize_points(pts2)

    #todo: Form the matrix A
    A = np.zeros((len(pts1), 9))
    for i in range(len(pts1)):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A[i] = [x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1]
        
    #todo: Find the fundamental matrix
    # Find the SVD of (A^T A)
    U, S, V = np.linalg.svd(A)
    # Entries of F are the elements of column of V corresponding to the least singular value
    F = V[-1].reshape(3, 3)

    # Enforce rank 2 constraint on F
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))

    # Un-normalize F
    F = np.dot(T2.T, np.dot(F, T1))

    return F

def FindFundamentalMatrixRansac(pts1, pts2, num_trials = 1000, threshold = 0.01):
    #Input: two lists of corresponding keypoints (numpy arrays of shape (N, 2))
    #Output: fundamental matrix (numpy array of shape (3, 3))

    #todo: Run RANSAC and find the best fundamental matrix
    best_inliers_count = 0
    best_fundamental_matrix = None

    for _ in range(num_trials):
        # Randomly sample 8 points
        indices = np.random.choice(len(pts1), 8, replace=False)
        sampled_pts1 = pts1[indices]
        sampled_pts2 = pts2[indices]

        # Compute the fundamental matrix using these 8 points
        F = FindFundamentalMatrix(sampled_pts1, sampled_pts2)

        # Count the number of inliers
        inliers_count = 0
        for i in range(len(pts1)):
            pt1 = np.append(pts1[i], 1)  # homogeneous coordinates
            pt2 = np.append(pts2[i], 1)

            # Compute the distance from the point to the epipolar line
            line = F @ pt1
            distance = abs(pt2 @ line) / np.linalg.norm(line[:2])

            # Check if the distance is below the threshold
            if distance < threshold:
                inliers_count += 1

        # Update the best fundamental matrix if needed
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_fundamental_matrix = F

    return best_fundamental_matrix

if __name__ == '__main__':
    #Set parameters
    data_path = './data'
    use_ransac = True

    #Load images
    image1_path = os.path.join(data_path, 'myleft.jpg')
    image2_path = os.path.join(data_path, 'myright.jpg')
    image1 = np.array(Image.open(image1_path).convert('L'))
    image2 = np.array(Image.open(image2_path).convert('L'))


    #Find matching keypoints
    pts1, pts2 = find_matching_keypoints(image1, image2)

    #Builtin opencv function for comparison
    F_true = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)[0]

    #todo: FindFundamentalMatrix
    if use_ransac:
        F = FindFundamentalMatrixRansac(pts1, pts2)
    else:
        F = FindFundamentalMatrix(pts1, pts2)

    # Find epilines corresponding to points in second image,  and draw the lines on first image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F_true)  # replace F_true with F to test the performance of our algorithm
    lines1 = lines1.reshape(-1, 3)
    img1, img2 = drawlines(image1, image2, lines1, pts1, pts2)
    fig, axis = plt.subplots(1, 2)

    axis[0].imshow(img1)
    axis[0].set_title('Image 1')
    axis[0].axis('off')
    axis[1].imshow(img2)
    axis[1].set_title('Image 2')
    axis[1].axis('off')

    plt.show()


    # Find epilines corresponding to points in first image, and draw the lines on second image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F_true)  # replace F_true with F to test the performance of our algorithm
    lines2 = lines2.reshape(-1, 3)
    img1, img2 = drawlines(image2, image1, lines2, pts2, pts1)
    fig, axis = plt.subplots(1, 2)

    axis[0].imshow(img1)
    axis[0].set_title('Image 1')
    axis[0].axis('off')
    axis[1].imshow(img2)
    axis[1].set_title('Image 2')
    axis[1].axis('off')

    plt.show()





