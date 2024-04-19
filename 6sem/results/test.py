import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image, ImageFont
from PIL.ImageOps import invert
from fontTools.ttLib import TTFont


def binarized(img):
    return np.array(img) > 255//2


if __name__ == '__main__':
    directory = os.fsencode('output')
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith('.png'):
            img = Image.open('output/' + filename).convert('L')
            img_bin = binarized(img)
            Image.fromarray(img_bin).convert('L').save('binarized/' + filename)


