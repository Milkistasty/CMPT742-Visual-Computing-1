import cv2
import numpy as np

def get_matrix(alpha, beta, gamma, num_points):
    """Return the matrix for the internal energy minimization.
    # Arguments
        alpha: The alpha parameter.
        beta: The beta parameter.
        gamma: The gamma parameter.
        num_points: The number of points in the curve.
    # Returns
        The matrix for the internal energy minimization. (i.e. A + gamma * I)
    """
    """
        it tries to force the snake to be small and smooth
    """
    # Component of A
    # x(i)
    # a -- Main diagonal
    a = (2 * alpha + 6 * beta) * np.ones(num_points)
    # x(i-1)
    # b -- Second diagonal below the main diagonal
    b = (-alpha - 4 * beta) * np.ones(num_points - 1)
    # x(i+2)
    # c -- Third diagonal below the main diagonal
    c = beta * np.ones(num_points - 2)

    # Create the pentadiagonal matrix A
    A = np.diag(a) + np.diag(b, k=1) + np.diag(b, k=-1) + np.diag(c, k=2) + np.diag(c, k=-2)

    # Compute M = (A + gamma * I) ^ (-1) 
    M = np.linalg.inv(A + gamma * np.identity(num_points))

    return M
