import cv2
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from align_target import align_target


# for poisson blending, we need to change the boundary condition for A matrix

def poisson_blend(source_image, target_image, target_mask):
    #source_image: image to be cloned
    #target_image: image to be cloned into
    #target_mask: mask of the target image

    # Compute the Laplacian of the source image
    # if use cv2.laplacian will need to compute 3 channel seperately and combine them in the end
    laplacian_source = cv2.Laplacian(source_image, cv2.CV_64F)
    cv2.imshow("Laplacian", laplacian_source)
    cv2.waitKey(0)

    # Indices of the mask where it is non-zero
    mask_indices = np.where(target_mask > 0)
    
    # Total number of pixels in the mask
    num_pixels = len(mask_indices[0])
    
    """ 
        the A and b matrix setting, 
        refers to https://github.com/willemmanuel/poisson-image-editing/blob/master/poisson.py 
    """

    # Create matrix A and vector b
    # Create A as a sparse matrix
    A = lil_matrix((num_pixels, num_pixels))
    
    height, width = target_mask.shape

    # Setting up the linear system of equations for Poisson blending based on the Laplacian
    # Fill in matrix A
    # For pixels inside the mask and not on the boundary: 
    # The Laplacian from the source image is used, and the matrix 
    # A is set up based on neighbors inside the mask
    # For pixels on the boundary of the mask: 
    # The corresponding row in A is set to an identity row, 
    # and b is set to the value from the target image for that pixel
    for idx, (y, x) in enumerate(zip(mask_indices[0], mask_indices[1])):
        boundary_pixel = False
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height or target_mask[ny, nx] == 0:
                boundary_pixel = True
                break

        if boundary_pixel:
            A[idx, idx] = 1
        else:
            A[idx, idx] = 4
            neighbors = [(y+dy, x+dx) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
            for ny, nx in neighbors:
                neighbor_idx = np.where((mask_indices[0] == ny) & (mask_indices[1] == nx))[0]
                A[idx, neighbor_idx] = -1

    # Split the channels
    target_channels = cv2.split(source_image)
    # Initialize blended_channels with target_image
    blended_channels = cv2.split(source_image.copy())

    # Iterate over each channel
    for channel in range(3):
        b = np.zeros(num_pixels)
        for idx, (y, x) in enumerate(zip(mask_indices[0], mask_indices[1])):
            boundary_pixel = False
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= width or ny < 0 or ny >= height or target_mask[ny, nx] == 0:
                    boundary_pixel = True
                    break

            if boundary_pixel:
                b[idx] = target_channels[channel][y, x]
            else:
                b[idx] = laplacian_source[y, x, channel]
                    
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
    # cv2.imwrite('blended_image_2.jpg', blended_image)
    print("Least Squares Error:", least_square_error)

    cv2.imshow('blended_image_2.jpg', blended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    