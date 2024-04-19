import numpy as np
import os

from semitone import semitone
from PIL import Image, ImageDraw

'''
def get_median(img, pos, kernel):
    pixels = []
    ker_i, ker_j = 0, 0
    img_i, img_j = pos
    img_i = img_i - 2
    img_j = img_j - 2
    for ker_i in range(5):
        for ker_j in range(5):
            if kernel[ker_i][ker_j] == 1:
                pixels.append(img[img_i][img_j])
    #print(pixel)
    return np.median(pixels)
'''


def median_filter(img, kernel_name):
    kernel = {
        'hill':np.array(
        [[0,0,1,0,0],
         [0,1,1,1,0],
         [1,1,1,1,1],
         [0,1,1,1,0],
         [0,0,1,0,0]]),
        'hollow':np.array(
        [[1,1,0,1,1],
         [1,0,0,0,1],
         [0,0,1,0,0],
         [1,0,0,0,1],
         [1,1,0,1,1]]),
    }
    res = np.zeros(img.shape)
    w, l = img.shape
    for i in range(2, w - 2):
        for j in range(2, l - 2):
            res[i][j] = np.median(img[i-2:i+3,j-2:j+3][kernel[kernel_name]==1])
    return res

if __name__ == "__main__":

    directory = os.fsencode('input')
    kernel_names = ['hill', 'hollow']

    for file in os.listdir('input'):
        filename = os.fsdecode(file)
        if filename.endswith('.png'):
            for kernel in kernel_names:    
                name = filename.split('.')[0]
                img = Image.open('input/' + filename).convert('RGB')
                img = semitone(np.array(img))
                img_filtered = median_filter(img, kernel)
                Image.fromarray(img_filtered).convert('RGB').save(f'output/{name}_{kernel}.png')
                Image.fromarray(abs(img_filtered - img)).convert('RGB').save(f'output/{name}_{kernel}_diff.png')