import cv2
import numpy as np

from external_energy import *
from internal_energy_matrix import *

"""
    1. Define what is internal energy
        ...

    2. Conversion
        Euler-Lagrange equation
        Minimize using Euler-Lagrange equation: 
            second order partial differential equation
            finds solutions that make the function stationary -> functions derivative = 0

    3. Solve with gradient descent
        3.1 Set 0 to be derivative of s over time
        -a * x''(s) + b * x''''(s) - dE(ext) / d(x, y) = dx / dt

        when the snake has converged to minimum, there will be no further change
        derivative of s over time will be 0 -> equation is satisfied

        3.2 Finite differences to get derivatives on N spline points
        e.g. for X direction:
            x(i)'' = x(i-1) - 2 * x(i) + x(i+1)
            x(i)'''' = x(i-2) - 4 * x(i-1) + 6 * x(i) - 4 * x(i+1) + x(i+2)
            
        3.3 Formulate A
            d[xt(i)] / dt = -a * [x(i-1) - 2 * x(i) + x(i+1)] + b * [x(i-2) - 4 * x(i-1) + 6 * x(i) - 4 * x(i+1) + x(i+2)] - f(x) 
            
        This can be rewritten in matrix form
            x(i-2) : b
            x(i-1) : -a - 4b
            x(i) : 2a + 6b
            x(i+1) : -a - 4b
            x(i+2) : b

        d[x(t)] / dt = Ax(t) - fx(x(t-1), y(t-1))
        [-a * x''(s) + b * x''''(s) - dE(ext) / d(x, y)] = [Ax(t) - fx(x(t-1), y(t-1))] = d[x(t)] / dt

        A = N by N, N is the number of discrete points for the snake

    4. iterate
        now we assume that the movement of the snake in each iteration is very small
            d[x(t)] / dt can be approximated by (x(t) - x(t-1))
        so we have:
        -gamma * (x(t) - x(t-1)) = Ax(t) - fx(x(t-1), y(t-1))
        -gamma * x(t) = Ax(t) - gamma * x(t-1) - fx(x(t-1), y(t-1))
        -(A + gamma * I) * x(t) = -gamma * x(t-1) - fx(x(t-1), y(t-1))
        x(t) = (A + gamma * I) ^ (-1) * (gamma * x(t-1) + fx(x(t-1), y(t-1)) by tranform, finally we have this equation, the new x coor. can be computed

"""

def click_event(event, x, y, flags, params):
    global xs, ys
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        #save point
        xs.append(x)
        ys.append(y)

        #display point
        cv2.circle(img_copy, (x, y), 3, 128, -1)
        cv2.imshow('img_copy', img_copy)

# Bilinear interpolation function
def bilinear_interpolation(image, x, y):
    # Ensure the coordinates are within the image bounds
    x = np.clip(x, 0, image.shape[1] - 2)
    y = np.clip(y, 0, image.shape[0] - 2)
   
    # Getting the integer and fractional parts
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1
    dx = x - x0
    dy = y - y0

    # weighted average of the four surrounding pixel values
    # (1 - dx) * (1 - dy) * image[y0, x0]: top-left pixel
    # dx * (1 - dy) * image[y0, x1]: top-right pixel
    # (1 - dx) * dy * image[y1, x0]: bottom-left pixel
    # dx * dy * image[y1, x1]: bottom-right pixel
    interpolated_value = ((1 - dx) * (1 - dy) * image[y0, x0] + 
                          dx * (1 - dy) * image[y0, x1] + 
                          (1 - dx) * dy * image[y1, x0] + 
                          dx * dy * image[y1, x1])
      
    return interpolated_value

def interpolate(xs, ys, num_points):
    interpolated_xs, interpolated_ys = [], []

    # Create a closed loop
    xs.append(xs[0])
    ys.append(ys[0])

    # Compute the total distance of the closed loop
    distances = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    total_distance = np.sum(distances)

    # Compute the cumulative distance for each point along the loop
    cumulative_distances = np.zeros(len(xs))
    cumulative_distances[1:] = np.cumsum(distances)

    # Create an array of distances at which to evaluate the interpolated coordinates
    interp_distances = np.linspace(0, total_distance, num_points)

    # Use np.interp to interpolate the x and y coordinates
    interpolated_xs = np.interp(interp_distances, cumulative_distances, xs)
    interpolated_ys = np.interp(interp_distances, cumulative_distances, ys)

    # display point
    for _, (x, y) in enumerate(zip(interpolated_xs, interpolated_ys)):
        cv2.circle(img_copy, (int(x), int(y)), 3, 128, -1)
    cv2.imshow('img_copy', img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return interpolated_xs, interpolated_ys

if __name__ == '__main__':
    # point initialization
    img_path = 'images/star.png'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blurring the image at the beginning, might help with the quality of the output
    blur_size = (3, 3) 
    img = cv2.GaussianBlur(img, blur_size, 0)
    img_copy = img.copy()
                                                             
    # selected points are i n xs and ys
    xs = []
    ys = []
    cv2.imshow('img_copy', img_copy)

    cv2.setMouseCallback('img_copy', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # interpolate
    # implement part 1: interpolate between the selected points
    num_points = 100
    interpolated_xs, interpolated_ys = interpolate(xs, ys, num_points)

    # Parameters for energy minimization
    alpha = 0.04  # Controls the snake's tension
    beta = 5  # Controls the snake's rigidity or stiffness
    gamma = 0.5  # how much influence our previous time step has
    kappa = 0.5  # weighting for the external energy
    sigma1 = 1  # sigma used to calculate the gradient of the image
    sigma2 = 1  # sigma used to calculate the gradient for the term energy
    sigma3 = 1  # sigma used to calculate the gradient of the external energy
    w_line = 0.5  # weighting for line energy
    w_edge = 2.5  # weighting for edge energy
    w_term = 0.5  # weighting for term energy
    num_iterations = 100  # number of iterations for energy minimization
    # maxPixelMove = 5  # number of max pixel movement allowed for each point

    # Get Internal Energy                                 
    M = get_matrix(alpha, beta, gamma, num_points)

    # Get external energy
    E = external_energy(img, w_line, w_edge, w_term)
    # Blur the external energy again
    E = cv2.GaussianBlur(E, blur_size, 0)
 
    # Compute the gradients of External Energy
    gradient_x = cv2.Sobel(E, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(E, cv2.CV_64F, 0, 1, ksize=3)
                   
    # Check images for gradients and External Energy
    cv2.imshow("External Energy", E)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    cv2.imshow("gradient_x", gradient_x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("gradient_y", gradient_y)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
    # Optimization loop
    xt = np.array(interpolated_xs, dtype=float) 
    yt = np.array(interpolated_ys, dtype=float) 

    for iter in range(num_iterations):
        # Calculate gradients of external energy using bilinear interpolation
        fx = np.zeros(num_points)
        fy = np.zeros(num_points)

        for i in range(num_points):
            fx[i] = bilinear_interpolation(gradient_x, xt[i], yt[i])
            fy[i] = bilinear_interpolation(gradient_y, xt[i], yt[i])

        # Update the new xt and yt using the optimization equation
        xt = M @ (gamma * xt - kappa * fx)
        yt = M @ (gamma * yt - kappa * fy)
   
        # Clipping xt and yt ensures that the snake's coordinates don't move outside the image boundaries
        xt = np.clip(xt, 0, img.shape[1] - 1)
        yt = np.clip(yt, 0, img.shape[0] - 1)

        # Display the intermediate result at each iteration
        img_temp = img.copy()
        for _, (x, y) in enumerate(zip(xt, yt)):
            cv2.circle(img_temp, (int(x), int(y)), 3, 128, -1)
        cv2.imshow(f'Iteration {iter}', img_temp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.imwrite('images/brain_result_eye.png', img_temp)
