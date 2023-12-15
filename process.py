import cv2 as cv

import numpy as np
import math
from main import read_yaml

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
   
   if len(top_margin) != 0 and len(bottom_margin) != 0 and len(right_margin) != 0 and len(left_margin) != 0:
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
    """Removes the internal pixels from the filled image.

    Args:
        binary_img np.ndarray: Binary image

    Returns:
        np.ndarray: Contour image
    """

    img_cpy = binary_img.copy()

    filled = hole_filling(img_cpy)
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (4, 4))
    eroded = cv.erode(filled,  kernel, iterations=1)
    
    ret_img = filled - eroded
    # cv.imshow('contour', ret_img)
    return ret_img

def detect_line(line_slice):
    """Runs Hough transform on contour image.

    Args:
        line_slice np.ndarray: Contour image

    Returns:
        np.ndarray: Returns the polar coordinates of detected lines. 
    """

    sliced = line_slice.copy()
    max_cell = yaml_const['PROCESSING_CONST']['HOUGH_MAX_CELL']

    contours = internal_pixel_removal_2(sliced)
    
    rho_res = 1 
    theta_res = 0.25 * math.pi / 180

    lines = cv.HoughLines(contours, rho_res, theta_res, max_cell)

    return lines


# def cartesian_to_polar(x, y):

#     theta =  np.arctan2(y, x)
#     rho = x * np.cos(theta) + y * np.sin(theta)
#     return rho, theta

# def Palagyis_megoldas(lines):
    
#     img_diag = int(np.ceil(np.sqrt(2970 ** 2 + 4200 ** 2)))

#     rho_res = 1
#     theta_res =  0.25 * np.pi / 180

#     height = int(np.round(img_diag / rho_res))
#     width = int(np.round(np.pi / theta_res))


#     accumulator = np.zeros((height, width), dtype=np.uint8)

#     if lines is not None:
#         for line in lines:
#             r, theta = line[0]
#             r_index =  int(r + img_diag)
#             theta_index = int(theta / theta_res)
#             accumulator[r_index, theta_index] = 1
#             print(accumulator[r_index, theta_index])

#     struct = cv.getStructuringElement(cv.MORPH_ELLIPSE, yaml_const['PROCESSING_CONST']['DILET_FOR_HOUGH'])
#     # accumulator = cv.dilate(accumulator, struct, iterations=1)
#     cv.imwrite('akkumulátor_dilettált.png', accumulator)
#     num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(accumulator, 4, cv.CV_32S)

#     # for label in labels:


#     print(f"Accumulator array: {np.sum(accumulator)}, line count: {len(lines)}, Label count: {num_labels}")
#     print(f'Indexes {np.where(accumulator == 1)}')
#     # cv.imshow('accumulator', accumulator)
    
#     # return centroids
#     return np.array([cartesian_to_polar(x, y) for x, y in centroids])

def detect_lines(binary_img, gray_img):
    """Calls the necessary function for line detection.

    Args:
        binary_img np.ndarray: Binary image.
        gray_img np.ndarray: Gray scale image, for testing purposes only.
    Returns:
        np.ndarray, np.ndarray: 1st value is for testin purposes (image with detected lines), 2nd the detected lines orientation
    """
 
    img_cpy = binary_img.copy()
    original = gray_img.copy()
    # cv.imshow('original', original)
    original_rgb = cv.cvtColor(original, cv.COLOR_GRAY2BGR)
    # cv.imshow('original RGB', original_rgb)

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
#             x1 = int(x0 + 5000 * (-b))
#             y1 = int(y0 + 5000 * (a)
#             x2 = int(x0 - 5000 * (-b))
#             y2 = int(y0 - 5000 * (a))
# #   
#             cv.line(original_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # lines2 = Palagyis_megoldas(lines)
    x_thresh = yaml_const['PROCESSING_CONST']['X_THRESH']
    y_thresh = yaml_const['PROCESSING_CONST']['Y_THRESH']
    lines2 = average_nearby_lines(lines, x_thresh, y_thresh* np.pi/180)

    thetas = []
    if lines2 is not None:
# 
        for line in lines2:
# 
            # rho, theta = line[0]
            rho, theta = line
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 5000 * (-b))
            y1 = int(y0 + 5000 * (a))
            x2 = int(x0 - 5000 * (-b))
            y2 = int(y0 - 5000 * (a))
            thetas.append(np.degrees(theta))
            cv.line(original_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv.imshow("line", original_rgb)
    cv.waitKey()
    cv.destroyAllWindows()
    
    return original_rgb, thetas

def average_nearby_lines(lines, rho_threshold, theta_threshold):
    """Averages the polar coordinates of lines that are close enough to each other.

    Args:
        lines np.ndarray: List of (rho, theta) tuples from HoughLines.
        rho_threshold float: The maximum allowed difference in rho for lines to be considered close.
        theta_threshold float: The maximum allowed difference in theta for lines to be considered close.

    Returns:
        np.ndarray: Array of averaged polar coordinates.
    """

    if lines is None:
        return []

    lines = filter_lines_by_angle(lines, yaml_const['PROCESSING_CONST']['VERTICAL_THRESH'])

    # Group lines based on the threshold
    grouped_lines = []
    for line in lines:
        rho, theta = line
        found_group = False
        for group in grouped_lines:
            if abs(group['mean_rho'] - rho) < rho_threshold and abs(group['mean_theta'] - theta) < theta_threshold:
                group['lines'].append((rho, theta))
                group['mean_rho'] = np.mean([l[0] for l in group['lines']])
                group['mean_theta'] = np.mean([l[1] for l in group['lines']])
                found_group = True
                break
        if not found_group:
            grouped_lines.append({'lines': [(rho, theta)], 'mean_rho': rho, 'mean_theta': theta})

    # Extract the averaged lines
    averaged_lines = [(group['mean_rho'], group['mean_theta']) for group in grouped_lines]

    return averaged_lines

def filter_lines_by_angle(lines, vert_thresh):
    """Return only those line that are inside the threshold

    Args:
        lines np.ndarray: Array of polar coordinates
        vert_thresh int: Threshold value. 

    Returns:
        np.ndarray: Array of filtered polar coordinates 
    """

    filtered_lines = []
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 5000 * (-b))
        y1 = int(y0 + 5000 * (a))
        x2 = int(x0 - 5000 * (-b))
        y2 = int(y0 - 5000 * (a))

        delt_x = abs(x2 - x1)
        delt_y = abs(y2 - y1)

        if delt_y == 0:
            continue
        elif (delt_x / delt_y) > vert_thresh:
            filtered_lines.append((rho, theta))

    return filtered_lines
