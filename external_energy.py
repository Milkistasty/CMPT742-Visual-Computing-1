import cv2
import numpy as np

def line_energy(image):
    #implement line energy (i.e. image intensity)
    return image.copy()

def edge_energy(image):
    #implement edge energy (i.e. gradient magnitude)
    # calculate the gradient using sobel convolution kernels
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # same as x[i] - x[i-1] for each i except the boundary point
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return gradient_magnitude

def term_energy(image):
    #implement term energy (i.e. curvature)
    # Calculate first-order derivatives
    Cx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Cy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate second-order derivatives
    Cxx = cv2.Sobel(Cx, cv2.CV_64F, 1, 0, ksize=3)
    Cyy = cv2.Sobel(Cy, cv2.CV_64F, 0, 1, ksize=3)
    Cxy = cv2.Sobel(Cx, cv2.CV_64F, 0, 1, ksize=3)

    # formula of curvature
    term_energy = (Cxx * (Cy**2) - 2 * Cxy * Cx * Cy + Cyy * (Cx**2)) / (((Cx**2) + (Cy**2))**1.5 + np.finfo(float).eps)
    return term_energy

def external_energy(image, w_line, w_edge, w_term):
    #implement external energy
    line = line_energy(image)
    edge = edge_energy(image)
    term = term_energy(image)

    # Apply min-max scaling
    line = (line - np.min(line)) / (np.max(line) - np.min(line) + np.finfo(float).eps)
    edge = (edge - np.min(edge)) / (np.max(edge) - np.min(edge) + np.finfo(float).eps)
    term = (term - np.min(term)) / (np.max(term) - np.min(term) + np.finfo(float).eps)

    # Eexternal = W(line) * E(line) + W(edge) * E(edge) + W(term) * E(term)
    external_energy = w_line * line + w_edge * edge + w_term * term
    return external_energy