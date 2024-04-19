import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from features import generate_letters
from PIL import Image, ImageFont
from PIL.ImageOps import invert


alphabet=list('abcdefghijklmnñopqrstuvwxyz')
sentence='ai ai se eu te pego hein'
font_size=52
font_path='times.ttf'

def binarized(img):
    return img < 255//2

def generate_sentence():
    font = ImageFont.truetype(font_path, size=font_size)

    mask_image = font.getmask(sentence, "L")
    img = Image.new("L", mask_image.size)
    img.im.paste((255), (0, 0) + mask_image.size, mask_image)
    invert(img).save('input/' + f"sentence.bmp")


def get_profiles(img, axis):
      x = np.sum(img, axis=0)
      y = np.sum(img, axis=1)
      return x, y

def split_letters(img, profile):
    letters = []
    borders = []
    letter_start = 0
    
    is_space = True

    for i in range(img.shape[1]):
        if profile[i] == 0:
            if not is_space:
                is_space = True
                letter, up, bottom = cut(img[:, letter_start:i])
                letters.append(letter)
                borders.append([(up, letter_start), (up, i), (bottom, letter_start), (bottom, i)])

        else:
            if is_space:
                is_space = False
                letter_start = i
                borders.append(letter_start)
    letter, up, bottom = cut(img[:, letter_start:img.shape[1] - 1])
    letters.append(letter)

    return letters, borders

def cut(letter):

    up = 0
    bottom = letter.shape[0] - 1
    while all(letter[up] == 255):
        up += 1
    while all(letter[bottom] == 255):
        bottom -= 1
    

    return letter[up:bottom+1], up, bottom
          
      

if __name__ == '__main__':
        generate_sentence()
        img = Image.open('input/sentence.bmp').convert('L')
        
        x, y = get_profiles(binarized(np.array(img)), 1)

        letters, borders = split_letters(np.array(img), x)
        for index, letter in enumerate(letters):
            Image.fromarray(letter).convert('L').save(f"output/{index}.png")
                
