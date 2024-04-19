import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import fnmatch

from ast import literal_eval
from PIL import Image, ImageFont
from PIL.ImageOps import invert
from segmentation import *
from features import *


alphabet=list('abcdefghijklmnñopqrstuvwxyz')
sentence='ai ai se eu te pego hein'
font_size=52
font_path='times.ttf'
features=pd.read_csv('result.csv', index_col='letter')

def calculate_distance(img_bin):

    hypothesis = []
    distances = []

    weight = get_norm_weight(img_bin)
    x, y = get_relative_center(img_bin)
    I_x, I_y = get_norm_inertia_moment(img_bin)

    print(weight, x, y, I_x, I_y)
    
    for letter in alphabet:
        letter_features = features.loc[letter]
        distance = np.sqrt((literal_eval(letter_features['rel_center'])[0] - x)**2 + (literal_eval(letter_features['rel_center'])[1] - y)**2 +\
                           (literal_eval(letter_features['norm_inertia'])[0] - I_x)**2 + (literal_eval(letter_features['norm_inertia'])[1] - I_y)**2 +\
                           (letter_features['norm_weight'] - weight)**2)
       
        distances.append(distance)

    max_dist = max(distances)
    for letter, distance in zip(alphabet, distances):
        hypothesis.append((letter, (max_dist-distance)/max_dist))

    return sorted(hypothesis, key=lambda x: x[1], reverse=True)


def cut_vert(letters):
    cutted_letters = []

    for letter in letters:
        up = 0
        bottom = letter.shape[0] - 1
        while all(letter[up] == 0):
            up += 1
        while all(letter[bottom] == 0):
            bottom -= 1
        
        cutted_letters.append(letter[up:bottom+1])

    return cutted_letters

def cut_hor(letters):
    cutted_letters = []

    for letter in letters:
        left = 0
        right = letter.shape[1] - 1
        while all(letter[:, left]==False):
            left += 1
        while all(letter[:, right]==False):
            right -= 1
        
        cutted_letters.append(letter[:, left:right])

    return cutted_letters




if __name__ == '__main__':
    '''
    generate_sentence()
    img = Image.open('input/sentence.bmp').convert('L')
    
    img = np.array(img)
    Image.fromarray(binarized(img)).convert('L').save('test.bmp')
    x, y = profiles(binarized(img), 1)
    letters, borders = split_letters(img, x)
    cutted = cut_vert(letters)
    cutted = cut_hor(cutted)
    for index, letter in enumerate(cutted):
        Image.fromarray(letter).convert('L').save(f"letters/{index}.png")

    ans = ""
    with open('output/hypothesis.txt', 'w') as f:
        directory = os.fsencode('letters')
        for i in range(0, 18):
            filename = f'{i}.png'
            img = Image.open('letters/' + filename).convert('L')
            img_bin = binarized(img)
            distance = calculate_distance(img_bin)
            f.write(str(distance))
            ans += distance[0][0]
            f.write("\n")
        f.write(f"{ans}\n")
        
        count = 0
        for a, b in zip(ans, sentence.replace(" ", "")):
            if a==b:
                count += 1
        
        f.write(str(count/len(sentence.replace(' ', '')) * 100))
    '''
    generate_sentence(sentence, 'input/sentence.bmp', font_size)
    img = Image.open('input/sentence.bmp').convert('L')
    split(img)
    ans=""
    with open('output/hypothesis.txt', 'w') as f:
        for i in range(0, len(fnmatch.filter(os.listdir('letters'), '*.png'))):
            filename = f'{i}.png'
            img = Image.open('letters/' + filename).convert('L')
            img_bin = binarized(img)
            distance = calculate_distance(img_bin)
            f.write(str(distance))
            ans += distance[0][0]
            f.write("\n")
        f.write(f"{ans}\n")
        f.write(f"{sentence.replace(' ', '')}\n")
        count = 0
        for a, b in zip(ans, sentence.replace(" ", "")):
            if a==b:
                count += 1
        
        f.write(str(count/len(sentence.replace(' ', '')) * 100))



    

