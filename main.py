import cv2 as cv
from pdf2image import convert_from_path
import os
import numpy as np

import pandas as pd              

import process
import tester

import tkinter as tk
from tkinter import filedialog

import yaml


def read_yaml(file_path: str) -> dict:
    """Reads yaml config file.

    Args:
        file_path (str): Path to config.yaml

    Returns:
        dict: dictionary containing config.yaml constants
    """
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

#Global constants
TITLE = "Makrostruktúrák elemzése."
SELECT_FILE_TEXT = "Válasszon fájlt:"
SELECT_FILE_BROWSE = "Tallózás"
GEOMETRY = "500x250"
START = "Elemzés"

#Global variables
features_df = pd.DataFrame()

#Reads config.yaml file.
yaml_const = read_yaml('config.yaml')

def convert_to_img(path: str) -> str:
    """Gets input image path and converts PDF to JPEG

    Args:
        path (str): Handwritten image path

    Returns:
        str: JPEG image path
    """
    img = convert_from_path(path, dpi=254)
     
    if '/' in path:
        splitted = path.split('/')
    else:
        splitted = path.split('\\')

    file_name = splitted[-1]

    img_path = yaml_const['PATH_CONST']['OUTPUTPATH'] + f'{file_name[:-4]}.jpg'

    img[0].save(img_path, 'JPEG')
    
    return img_path, file_name[:-4]

def empty_tmp():
    files = [f for f in os.listdir('tmp/')]
    for f in files:
        os.remove(os.path.join('tmp/', f))


def write_path():
    """Callback function. 

        Only let's select PDF files then writes file path into selected_file_btn label
    """
    path = filedialog.askopenfilename(
        filetypes=[('PDF Files', "*.pdf")]
    )
    selected_file_label.config(text=path)

def measure_margins(binary_img):
    """Calculates margins with method from process file and writes the values into a dataframe.

    Args:
        binary_img (np.array): Binarized handwritten image 
    """
    img_cpy = binary_img.copy()
    top, bottom, right, left = process.get_margins(img_cpy)

    features_df['Felső margó'] = [top / 100]
    features_df['Alsó margó'] = [bottom / 100]
    features_df['Minimális bal margó'] = [left / 100]

    conscious_left = process.get_conscious_margin(img_cpy, top)

    features_df['Tudatos bal margó'] = [conscious_left / 100] 

    second_point = process.get_2nd_point_4_slope(img_cpy, bottom)

    slope = (bottom - top) / (conscious_left - second_point)

    features_df['Bal margó iránya'] = [slope]



def start_analysis():
    path = selected_file_label.cget('text')
    img_path, file_name = convert_to_img(path)

    gray_scale_img, binary_img = process.convert_img_2_binary(img_path)

    measure_margins(binary_img)
    csv_path = yaml_const['PATH_CONST']['ANALYSISPATH'] + file_name + '.csv'
    features_df.to_csv(csv_path, sep=';', columns=features_df.columns)



if __name__ == '__main__':
    # """ 
    #     TODO proper documentation 
    # """

    # main_window = tk.Tk()
    # main_window.title(TITLE)
    # main_window.geometry(GEOMETRY)

    # select_file_label = tk.Label(main_window, text=SELECT_FILE_TEXT)

    # selected_file_label = tk.Label(main_window)

    # select_file_btn = tk.Button(main_window, text=SELECT_FILE_BROWSE, command=write_path)

    # start_analysis_btn = tk.Button(main_window, text=START, command=start_analysis)

    # main_window.columnconfigure(0)
    # main_window.columnconfigure(1)

    # main_window.rowconfigure(0)
    # main_window.rowconfigure(1)

    # select_file_label.grid(row=0, column=0)
    # selected_file_label.grid(row=0, column=1, columnspan=4)

    # select_file_btn.grid(row=1, column=0)
    # start_analysis_btn.grid(row=1, column=1)

    # main_window.mainloop()

    # route = './data'
    # # files = [os.path.join(route, f) for f in os.listdir(route) if os.path.isfile(os.path.join(route, f))]
    # # for i, f in enumerate(files):
    # #     img = convert_from_path(files[i], dpi=254)
    # #     # 7 removes the ./data/ substring
    # #     img[0].save(f'tmp/{f[7:-4]}.jpg', 'JPEG')
    # """Convert from pdf to JPEG 
    # """

    route = './tmp'
    files = [os.path.join(route, f) for f in os.listdir(route) if os.path.isfile(os.path.join(route, f))]

    # for i, f in enumerate(files):
    #     resized = cv.resize(process.convert_img_2_binary(f), (1000, 800))
    #     cv.imshow(f'Binary Image {i}', resized)
    #     cv.waitKey()   
    # cv.destroyAllWindows()
    """Test loop for all the images.
    """ 

    for i, f in enumerate(files):
        gray_img, img = process.convert_img_2_binary(f)
        top, bottom, right, left = process.get_margins(img) 
        internal_removed = process.detect_lines(img, gray_img, top, bottom)
        img_show = tester.ResizeWithAspectRatio(internal_removed, 1250, 800)
        cv.imshow(f'With margin {i}', img_show)
        # tester.slice_line(img, yaml_const['PROCESSING_CONST']['AVG_LINE_HEIGHT'],
        #                 top, bottom)
        # tester.test_canny(img)
        cv.waitKey()
        cv.destroyAllWindows()



    # files = [f for f in os.listdir('tmp/')]
    # for f in files:        
    #     os.remove(os.path.join('tmp/', f))
    # """Empty tmp folder.
    # """
