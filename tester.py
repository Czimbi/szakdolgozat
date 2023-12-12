import cv2 as cv
import numpy as np
import math
import process

def draw_line(binary_img, margin_top, margin_bottom, margin_right, margin_left_start, margin_left_end):
    img_cpy = binary_img.copy()
    img_clr = cv.cvtColor(img_cpy, cv.COLOR_GRAY2BGR)
    top_line = cv.line(img_clr, (0, margin_top), (img_cpy.shape[1], margin_top), (255, 0, 0), 5)
    bottom_line = cv.line(top_line, (0, margin_bottom), (img_cpy.shape[1], margin_bottom), (255, 255, 0), 5)
    right_line = cv.line(bottom_line, (margin_right, margin_top), (margin_right, margin_bottom), (0, 255, 0), 5)
    left_line = cv.line(right_line, (margin_left_start, margin_top), (margin_left_end, margin_bottom), (0, 0, 255), 5)

    return left_line

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv.resize(image, dim, interpolation=inter)

def slice_line(binary_img, avg_line_height, top, bottom):

    line_num = math.floor((bottom - top) / avg_line_height)

    for line in range(1, line_num):
        cv.imshow(f'sliced{line}', binary_img[top + (avg_line_height * (line - 1)) : top + (avg_line_height * (line + 1)),:]) 
        cv.waitKey()

def test_canny(binary_img):
    
    img = binary_img.copy()

    contours = process.internal_pixel_removal_2(img)

    edges = cv.Canny(contours, 1, 1)

    cv.imshow('egdes', edges)