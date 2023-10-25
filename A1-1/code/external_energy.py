import cv2
import numpy as np
from scipy.ndimage import convolve

"""
% Gaussian based image derivatives
%
%  J=ImageDerivatives2D(I,sigma,type)
%
% inputs,
%   I : The 2D image
%   sigma : Gaussian Sigma
%   type : 'x', 'y', 'xx', 'xy', 'yy'
%
% outputs,
%   J : The image derivative
%
% Function is written by D.Kroon University of Twente (July 2010)
Retrieved at https://www.mathworks.com/matlabcentral/fileexchange/28149-snake-active-contour?s_tid=prof_contriblnk
"""
def gaussian_derivative_filter(sigma, derivative_type):
    """
    Generate Gaussian derivative filters.
    :param sigma: Gaussian Sigma.
    :param derivative_type: Type of derivative ('x', 'y', 'xx', 'xy', 'yy').
    :return: Gaussian derivative filter.
    """
    # Create grid
    x, y = np.meshgrid(np.arange(-3*sigma, 3*sigma+1), np.arange(-3*sigma, 3*sigma+1))
    
    # Compute Gaussian derivative filter based on the type
    if derivative_type == 'x':
        DGauss = -(x / (2 * np.pi * sigma**4)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    elif derivative_type == 'y':
        DGauss = -(y / (2 * np.pi * sigma**4)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    elif derivative_type == 'xx':
        DGauss = (x**2 / sigma**2 - 1) * np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**4)
    elif derivative_type in ['xy', 'yx']:
        DGauss = (x * y) * np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**6)
    elif derivative_type == 'yy':
        DGauss = (y**2 / sigma**2 - 1) * np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**4)
    else:
        raise ValueError(f"Unknown derivative type: {derivative_type}")
    
    return DGauss

def line_energy(image):
    #implement line energy (i.e. image intensity)
    return image.copy()

def edge_energy(image, sigma):
    #implement edge energy (i.e. gradient magnitude)
    # calculate the gradient using sobel convolution kernels
    grad_x = convolve(image, gaussian_derivative_filter(sigma, 'x'), mode='reflect') * 64
    grad_y = convolve(image, gaussian_derivative_filter(sigma, 'y'), mode='reflect') * 64 
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return gradient_magnitude

def term_energy(image, sigma):
    #implement term energy (i.e. curvature)
    # Calculate first-order derivatives
    Cx = convolve(image, gaussian_derivative_filter(sigma, 'x'), mode='reflect') * 64 
    Cy = convolve(image, gaussian_derivative_filter(sigma, 'y'), mode='reflect') * 64 

    # Calculate second-order derivatives
    Cxx = convolve(Cx, gaussian_derivative_filter(sigma, 'x'), mode='reflect') * 64 
    Cyy = convolve(Cy, gaussian_derivative_filter(sigma, 'y'), mode='reflect') * 64 
    Cxy = convolve(Cx, gaussian_derivative_filter(sigma, 'y'), mode='reflect') * 64 

    # formula of curvature
    term_energy = (Cxx * (Cy**2) - 2 * Cxy * Cx * Cy + Cyy * (Cx**2)) / (((Cx**2) + (Cy**2))**1.5 + np.finfo(float).eps)
    return term_energy

def external_energy(image, w_line, w_edge, w_term, sigma1, sigma2):
    #implement external energy
    line = line_energy(image)
    edge = edge_energy(image, sigma1)
    term = term_energy(image, sigma2)

    # Apply min-max scaling
    # line = (line - np.min(line)) / (np.max(line) - np.min(line) + np.finfo(float).eps)
    # edge = (edge - np.min(edge)) / (np.max(edge) - np.min(edge) + np.finfo(float).eps)
    # term = (term - np.min(term)) / (np.max(term) - np.min(term) + np.finfo(float).eps)

    # Eexternal = W(line) * E(line) + W(edge) * E(edge) + W(term) * E(term)
    external_energy = w_line * line + w_edge * edge + w_term * term
    return external_energy