import cv2
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from align_target import align_target

def poisson_blend(source_image, target_image, target_mask):
    #source_image: image to be cloned
    #target_image: image to be cloned into
    #target_mask: mask of the target image
    # Compute the Laplacian of the source image
    # if use cv2.laplacian will need to compute 3 channel seperately and combine them in the end
    laplacian_source = cv2.Laplacian(source_image, cv2.CV_64F)
    
    # Indices of the mask where it is non-zero
    mask_indices = np.where(target_mask > 0)
    
    # Total number of pixels in the mask
    num_pixels = len(mask_indices[0])
    
    # Create matrix A and vector b
    # Create A as a sparse matrix
    A = lil_matrix((num_pixels, num_pixels))
    
    # Create a mapping from (x, y) coordinates to index in A and b
    coord_to_idx = {(x,y): idx for idx, (x,y) in enumerate(zip(mask_indices[1], mask_indices[0]))}
    
    # Setting up the linear system of equations for Poisson blending based on the Laplacian
    # Fill in matrix A
    for idx, (y, x) in enumerate(zip(mask_indices[0], mask_indices[1])):
        A[idx, idx] = 4
        # Check each neighbor
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if (ny, nx) in coord_to_idx:
                A[idx, coord_to_idx[(ny, nx)]] = -1

    # Split the channels
    target_channels = cv2.split(target_image)
    # Initialize blended_channels with target_image
    blended_channels = cv2.split(target_image.copy())

    # Iterate over each channel
    for channel in range(3):  # Assuming RGB image
        b = np.zeros(num_pixels)
        for idx, (y, x) in enumerate(zip(mask_indices[0], mask_indices[1])):
            b[idx] = laplacian_source[y, x, channel]
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if not (ny, nx) in coord_to_idx:
                    b[idx] = target_channels[channel][ny, nx]
                    

        # Convert A to CSR format before solving
        A = A.tocsr()
        # Solve for the pixel values for the current channel
        blended_values_channel = spsolve(A, b)
        
        for idx, (y, x) in enumerate(zip(mask_indices[0], mask_indices[1])):
            # Bypass blending for pixels inside the mask and copy them directly from the source image
            blended_channels[channel][y, x] = blended_values_channel[idx]

    # Merge the channels to get the full blended image
    blended_image = cv2.merge(blended_channels)

    # Compute the least squares error for the whole image
    error = np.linalg.norm(A @ blended_values_channel - b)
    
    return blended_image, error


if __name__ == '__main__':
    #read source and target images
    source_path = 'source1.jpg'
    target_path = 'target.jpg'
    source_image = cv2.imread(source_path)
    target_image = cv2.imread(target_path)

    #align target image
    im_source, mask = align_target(source_image, target_image)

    ##poisson blend
    blended_image, least_square_error = poisson_blend(im_source, target_image, mask)
    cv2.imwrite('blended_image_1.jpg', blended_image)
    print("Least Squares Error:", least_square_error)