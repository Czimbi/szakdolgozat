from pdf2image import convert_from_path
import os

TMP_PATH = 'tmp/'

# def convert_folder_to_images(path=''):
#     """
#     TODO proper documentation according to sphinx stanards
#     """
#     if(path == ''):
#         raise ValueError('path attribute is empty!')
    
#     files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
#     print(files)
#     for i, f in enumerate(files):
#         img = convert_from_path(files[i])
#         # 7 removes the ./data/ substring
#         img[0].save(f'tmp/{f[7:-4]}.jpg', 'JPEG')

def empty_tmp_folder():
    """
        TODO proper documentation.
    """
    files = [f for f in os.listdir(TMP_PATH)]
    for f in files:
        os.remove(os.path.join(TMP_PATH, f))

