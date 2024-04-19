import numpy as np
import os

from PIL import Image
from semitone import semitone

def get_pixel(image, grad_type, x, y):
    if grad_type == 'x':
        return (image[x-1, y+1] + 2 * image[x, y+1] + image[x+1, y+1]) - (image[x-1, y-1] + 2 * image[x, y-1] + image[x+1, y-1])
    elif grad_type == 'y':
        return (image[x+1, y-1] + 2 * image[x+1, y] + image[x+1, y+1]) - (image[x-1, y-1] + 2 * image[x-1, y] + image[x-1, y+1])
    elif grad_type == 'xy':
        return abs((image[x-1, y+1] + 2 * image[x, y+1] + image[x+1, y+1]) - (image[x-1, y-1] + 2 * image[x, y-1] + image[x+1, y-1])) + \
               abs((image[x+1, y-1] + 2 * image[x+1, y] + image[x+1, y+1]) - (image[x-1, y-1] + 2 * image[x-1, y] + image[x-1, y+1]))
                


def get_grad(img, grad_type):

    res = np.zeros(img.shape)

    for i in range(1, res.shape[0]-1):
        for j in range(1, res.shape[1]-1):
            res[i][j] = get_pixel(img, grad_type, i, j)
    return 255 * res / np.max(res)

def get_grad_binarized(img, threshold):
    res = np.zeros(img.shape)

    res[img > threshold] = 255
    res[img <= threshold] = 0
    
    return res


if __name__ == "__main__":

    for file in os.listdir('input'):
        filename = os.fsdecode(file)
        if filename.endswith('.png'):
            img = np.array(Image.open('input/' + filename).convert('RGB'))
            sem = semitone(img)
            x = get_grad(sem, 'x')
            y = get_grad(sem, 'y')
            xy = get_grad(sem, 'xy')
            Image.fromarray(x).convert('RGB').save('output/x/' + filename)
            Image.fromarray(y).convert('RGB').save('output/y/' + filename)
            Image.fromarray(xy).convert('RGB').save('output/xy/' + filename)
            Image.fromarray(sem).convert('RGB').save('output/semitone/' + filename)
            for threshold in range(5, 15):
                binarized = get_grad_binarized(xy, threshold=threshold)
                Image.fromarray(binarized).convert('L').save('output/Binarized/' + str(threshold) + '_' + filename)

        