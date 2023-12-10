import cv2 as cv

import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from main import read_yaml
import matplotlib.pyplot as plt

yaml_const = read_yaml('config.yaml')

def convert_img_2_binary(img_path):
    """Converts a grayscale image into binary.
        
        Args:
            img_path (str): Image path
        Returns:
            np.array, np.array: that represents a binary image and the original image grayscaled version.
    """
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    block_size = yaml_const['PROCESSING_CONST']['BIN_BLOCK_SIZE']
    block_const = yaml_const['PROCESSING_CONST']['BIN_CONST']

    binary_img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, block_size, block_const)

    return img, binary_img

def get_projections(binary_image):
   """Calculates the row and columns projections.

   Args:
       binary_image (numpy.ndarray): Binarized handwritten image

   Returns:
       tuple (np.array, np.array): First element is the sum of rows the second is the columns
   """

   return (np.sum(binary_image, axis=1), np.sum(binary_image, axis=0))

def get_top_margin(rows, rows_max):
   """Calculates the first row where the condition is setisfied

   Args:
       rows (numpy.ndarray): projection of rows
       rows_max (int): rows max value

   Returns:
       int: First row index where the condition is setisfied.
   """
   return np.where(rows >= (rows_max * yaml_const['PROCESSING_CONST']['MARGIN_CONST']))[0]

def get_bottom_margin(rows, rows_max):
   """Calculates the first row where the condition is setisfied

   Args:
       rows (numpy.ndarray): projection of rows
       rows_max (int): rows max value

   Returns:
       int: First row index where the condition is setisfied.
   """
   return np.where(rows >= (rows_max * yaml_const['PROCESSING_CONST']['MARGIN_CONST']))[-1]

def get_right_margin(cols, cols_max):
   """Calculates the last column where the condition is setisfied

   Args:
       cols (numpy.ndarray): projection of columns
       cols_max (int): columns max value

   Returns:
       int: Last column index where the condition is setisfied.
   """
   return np.where(cols > (cols_max * yaml_const['PROCESSING_CONST']['MARGIN_CONST']))[-1]
   

def get_left_margin(cols, cols_max):
   """Calculates the first row where the condition is setisfied

   Args:
       rows (numpy.ndarray): projection of columns
       rows_max (int): rows max value

   Returns:
       int: First row index where the condition is setisfied.
   """
   return np.where(cols > (cols_max * yaml_const['PROCESSING_CONST']['MARGIN_CONST']))[0]

def get_margins(binary_image):
   """Calculates the minimal margins of an image

   Args:
       binary_image (numpy.array): nparray represeting a binary image.

   Returns:
         tuple: In order of top, bottom right and left margin starting positions. 
   """
   rows, cols = get_projections(binary_image)

   row_max = rows.max()
   col_max = cols.max()

   top_margin = get_top_margin(rows, row_max)
   bottom_margin = get_bottom_margin(rows, row_max)
   right_margin = get_right_margin(cols, col_max)
   left_margin = get_left_margin(cols, col_max)

   return (top_margin[0], bottom_margin[-1], right_margin[-1], left_margin[0])

def get_conscious_margin(binary_image, top_start):
    """Calculates the conscious margins for the user

    Args:
        binary_image (np.array): Binarized handwritten image
        top_start (int): top margin starting point

    Returns_
        int: Conscious margin distance
    """
    img_cpy = binary_image.copy()
    resized_top = img_cpy[top_start:top_start+yaml_const['PROCESSING_CONST']['AVG_ROW_SIZE']]
    _, col_left = get_projections(resized_top)

    cols_max_left = col_left.max()

    return np.where(col_left > (cols_max_left * 0.01))[0][0]

def get_2nd_point_4_slope(binary_image, bottom_start):
    """Calculates the last rows left margin.

    Args:
        binary_image (np.array): Binarized handwritten image
        bottom_start (int): Bottom margin y axis value

    Returns:
        int: Last rows left margin distance.
    """
   
    img_cpy = binary_image.copy()
    resized_bottom = img_cpy[bottom_start-yaml_const['PROCESSING_CONST']['AVG_ROW_SIZE']:bottom_start]
    _, col_left = get_projections(resized_bottom)

    cols_min_left = col_left.max()

    return np.where(col_left > (cols_min_left * 0.01))[0][0]

def diletation(binary_img, iterations = 1, struct_tuple = tuple(yaml_const['PROCESSING_CONST']['STRUCT_SIZE'])):
    """Diletates an image

    Args:
        binary_img (np.array): Binarized handwritten image
        iterations (int, optional): Number of iterations. Defaults to 1.

    Returns:
        np.array: Dilettad binary image
    """

    img_cpy = binary_img.copy()
    
    struct = cv.getStructuringElement(cv.MORPH_RECT, struct_tuple) 
    dilatted = cv.dilate(img_cpy, struct, iterations=iterations)

    return dilatted


def hole_filling(binary_img):
    """Dilettates an image, then fills the holes in the text.

    Args:
        binary_img (np.array): Binarized handwritten image

    Returns:
        np.array: Binary image, with no holes in the text
    """
    img_cpy = binary_img.copy()

    dilatted = diletation(img_cpy, yaml_const['PROCESSING_CONST']['DILET_ITER'])

    img_to_fill = dilatted.copy()

    h, w = img_to_fill.shape[:2]

    mask = np.zeros((h+2, w+2), np.uint8)

    cv.floodFill(img_to_fill, mask, (0, 0), 255)

    inverted = cv.bitwise_not(img_to_fill)

    return dilatted | inverted


def internal_pixel_removal_2(binary_img):

    img_cpy = binary_img.copy()

    filled = hole_filling(img_cpy)
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (4, 4))
    eroded = cv.erode(filled,  kernel, iterations=1)
    
    ret_img = filled - eroded
    return ret_img


def detect_line(line_slice):

    sliced = line_slice.copy()
    gap = yaml_const['PROCESSING_CONST']['AVG_LINE_GAP']
    contours = internal_pixel_removal_2(sliced)
    
    rho_res = 1 
    theta_res = 0.25 * math.pi / 180

    lines = cv.HoughLines(contours, rho_res, theta_res, 200)

    cv.imshow('Sliced', contours)
    cv.imwrite('tmp/test_contours.png', contours)
    cv.waitKey()

    return lines


def cartesian_to_polar(x, y):

    theta =  np.arctan2(y, x)
    rho = x * np.cos(theta) + y * np.sin(theta)
    return rho, theta

def Palagyis_megoldas(lines):
    
    img_diag = np.sqrt(2970 ** 2 + 4200 ** 2)

    rho_res = 1
    theta_res =  0.25 * np.pi / 180

    height = int(np.round(img_diag / rho_res))
    width = int(np.round(np.pi / theta_res))


    accumulator = np.zeros((height, width), dtype=np.uint8)

    if lines is not None:
        for line in lines:
            r, theta = line[0]
            r_index =  int(np.round(r))
            theta_index = int(np.round(theta))
            accumulator[r_index, theta_index] = 1

    accumulator = diletation(accumulator, struct_tuple=yaml_const['PROCESSING_CONST']['DILET_FOR_HOUGH']) #Hát ha
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(accumulator, 4, cv.CV_32S)

    # for label in labels:


    print(f"Accumulator array: {np.sum(accumulator)}, line count: {len(lines)}, Label count: {num_labels}")
    print(f'Indexes {np.where(accumulator == 1)}')
    cv.imshow('accumulator', accumulator)
    
    # return centroids
    return np.array([cartesian_to_polar(x, y) for x, y in centroids])

def detect_lines(binary_img, gray_img):
 
    img_cpy = binary_img.copy()
    original = gray_img.copy()
    
    original_rgb = cv.cvtColor(original, cv.COLOR_GRAY2BGR)

    gap_y = yaml_const['PROCESSING_CONST']['AVG_LINE_GAP']

    cv.imshow('blured', img_cpy)


    lines = detect_line(img_cpy)
#     if lines is not None:
# # 
#         for line in lines:
# # 
#             rho, theta = line[0]
#             # rho, theta = line
#             a = np.cos(theta)
#             b = np.sin(theta)
#             x0 = a * rho
#             y0 = b * rho
#             x1 = int(x0 + 2970 * (-b))
#             y1 = int(y0 + 4200 * (a))
#             x2 = int(x0 - 2970 * (-b))
#             y2 = int(y0 - 4200 * (a))
# #   
#             cv.line(original_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

    lines2 = Palagyis_megoldas(lines)
    # lines = lines_clustering(lines, 1, 2)

    if lines is not None:

        for line in lines2:

            # rho, theta = line[0]
            rho, theta = line
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 2970 * (-b))
            y1 = int(y0 + 4200 * (a))
            x2 = int(x0 - 2970 * (-b))
            y2 = int(y0 - 4200 * (a))

            cv.line(original_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)


    return original_rgb
