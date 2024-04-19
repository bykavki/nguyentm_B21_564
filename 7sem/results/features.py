import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image, ImageFont
from PIL.ImageOps import invert
from fontTools.ttLib import TTFont



alphabet=list('abcdefghijklmnñopqrstuvwxyz')
font_size=52
font_path='times.ttf'

def binarized(img):
    return np.array(img) > 255//2


def cut(img):
    img = np.array(img)
    left = 0
    right = img.shape[1] - 1
    while all(img[:, left] == 0):
        left += 1
    while all(img[:, right] == 0):
        right -= 1
    return Image.fromarray(img[:, left:right+1]).convert("L")

def generate_letters():
    font = ImageFont.truetype(font_path, size=font_size)

    for index, letter in enumerate(alphabet):
        font = ImageFont.truetype(font_path, size=font_size)
        mask_image = font.getmask(alphabet[index], "L")
        img = Image.new("L", mask_image.size)
        img.im.paste((255), (0, 0) + mask_image.size, mask_image)
        if index == 1:
            test = np.array(img)
            print(test[:, test.shape[1]-2])
        img = cut(img)
        invert(img).save('alphabet/' + f"{str(index+1).zfill(2)}.png")


def get_weight(img_bin):
    return np.sum(img_bin)

def get_norm_weight(img_bin):
    return np.sum(img_bin) / img_bin.shape[0] / img_bin.shape[1]

def get_center(img_bin):
    weight = get_weight(img_bin)

    sum_x = 0
    sum_y = 0
    for x in range(img_bin.shape[0]):
        for y in range(img_bin.shape[1]):
            sum_x += x * img_bin[x, y]
            sum_y += y * img_bin[x, y]
    
    return sum_x / weight, sum_y / weight

def get_relative_center(img_bin):
    x, y = get_center(img_bin)
    return x / img_bin.shape[0], y/img_bin.shape[1]

def get_inertia_moment(img_bin):
    x_c, y_c = get_center(img_bin)
    I_x, I_y = 0, 0

    for x in range(img_bin.shape[0]):
        for y in range(img_bin.shape[1]):
            I_x += (x - x_c)**2 * img_bin[x, y]
            I_y += (y - y_c)**2 * img_bin[x, y]
    
    return I_x, I_y

def get_norm_inertia_moment(img_bin):
    I_x, I_y = get_inertia_moment(img_bin)
    norm = (img_bin.shape[0] * img_bin.shape[1])**2
    return I_x / norm, I_y / norm

def get_profiles(img_bin):
        return {
        'x': {
            'y': np.sum(img_bin, axis=0),
            'x': np.arange(
                start=1, stop=img_bin.shape[1] + 1).astype(int)
        },
        'y': {
            'y': np.arange(
                start=1, stop=img_bin.shape[0] + 1).astype(int),
            'x': np.sum(img_bin, axis=1)
        }
    }

def draw_profile(img_bin, index, type='x'):
    profiles = get_profiles(img_bin)

    if type == 'x':
        plt.bar(x=profiles['x']['x'], height=profiles['x']['y'], width=0.9)

        plt.ylim(0, 52)

    elif type == 'y':
        plt.barh(y=profiles['y']['y'], width=profiles['y']['x'], height=0.9)

        plt.ylim(52, 0)


    plt.xlim(0, 52)

    plt.savefig(f'results/profiles/{type}/letter_{str(index).zfill(2)}.png')
    plt.clf()



if __name__ == '__main__':
    generate_letters()

    weight = []
    norm_weight = []
    center = []
    rel_center = []
    inertia = []
    norm_intertia = []

    directory = os.fsencode('alphabet')
    indexes = []


    for i in range(1, 28):
        filename = str(i).zfill(2) + '.png'
        img = Image.open('alphabet/' + filename).convert('L')
        img_bin = binarized(np.array(img))
        weight.append(get_weight(img_bin))
        norm_weight.append(get_norm_weight(img_bin))
        print(get_norm_weight(img_bin), filename)
        center.append(get_center(img_bin))
        rel_center.append(get_relative_center(img_bin))
        inertia.append(get_inertia_moment(img_bin))
        norm_intertia.append(get_norm_inertia_moment(img_bin))
        draw_profile(img_bin, int(filename.split('.')[0]), 'x')
        draw_profile(img_bin, int(filename.split('.')[0]), 'y')

    data = {
        'weight': weight,
        'norm_weight': norm_weight,
        'center': center,
        'rel_center': rel_center,
        'inertia': inertia,
        'norm_inertia': norm_intertia
    }
    

    df = pd.DataFrame(data, index=alphabet)
    print(df.dtypes)
    df.to_csv('results/result.csv', index_label='letter')


