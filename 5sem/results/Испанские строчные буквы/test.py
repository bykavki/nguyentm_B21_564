import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image, ImageFont
from PIL.ImageOps import invert
from fontTools.ttLib import TTFont
from features import *


def binarized(img):
    return np.array(img) > 255//2

def cut(img):
    img =  np.array(img)
    left = 0
    right = img.shape[1] - 1
    while all(img[:, left] == 0):
        left += 1
    while all(img[:, right] == 0):
        right -= 1
    return Image.fromarray(img[:, left:right]).convert("L")



if __name__ == '__main__':
    img = Image.open('alphabet/' + '01.png').convert('L')
    img_bin = binarized(img)
    img_bin = np.array(cut(img_bin))
    print(get_norm_weight(img_bin))
    

    img_bin = binarized(Image.open('alphabet/' + '01.png').convert('L'))


    print(get_norm_weight(img_bin))





