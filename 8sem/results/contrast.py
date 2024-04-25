import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw


d=1
phi=[0, 90, 180, 270]

def haralick_matrix(img, d):
    res = np.zeros((256, 256))
    hist = np.zeros(256).astype('uint8')
    W, H = img.shape
    k = 4*W*H - 2*W - 2*H


    for x in range(d, img.shape[0] - d):
        for y in range(d, img.shape[1] - d):
            res[img[x - d, y], img[x, y]] += 1
            res[img[x + d, y], img[x, y]] += 1
            res[img[x, y - d], img[x, y]] += 1
            res[img[x, y + d], img[x, y]] += 1
            hist[img[x, y]] += 1
    
    return hist, np.uint8(res), k

def calc_features(haralick, k):
    features = {
        'ASM': 0,
        'MPR': 0,
        'ENT': 0,
        'TR': 0
    }
    matrix = haralick * k
    features['ASM'] = np.sum(matrix**2)
    features['MPR'] = np.max(matrix)
    features['TR'] = np.trace(matrix)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 0:
                features['ENT'] -= matrix[i, j] * np.log2(matrix[i, j])
    return features
                                

def power_transformation(img, c, f0, gamma):
    res = np.zeros(img.shape)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i, j] = c * (img[i, j] + f0)**gamma

    return res*255/np.max(res)

if __name__ == "__main__":
    data = {
        'filename': [],
        'img_type':[],
        'ASM': [],
        'MPR': [],
        'ENT': [],
        'TR': []
    }
    
    for file in os.listdir('input'):
        filename = os.fsdecode(file)
        if filename.endswith('.png'):
            img = Image.open('input/' + filename).convert('L')
            img = np.array(img)
            contrast = power_transformation(img/255, 1, 0, 0.5)
            hist, matrix, k = haralick_matrix(img, 1)
            features = calc_features(matrix, k)

            data['filename'].append(filename)
            data['img_type'].append('default')
            data['ASM'].append(features['ASM'])
            data['ENT'].append(features['ENT'])
            data['MPR'].append(features['MPR'])
            data['TR'].append(features['TR'])

            plt.bar(np.arange(hist.size), hist)
            plt.savefig('output/' + f"{filename}_default_bar.png")
            plt.clf()


            Image.fromarray(matrix/k * 255/np.max(matrix/k)).convert('L').save('output/' + f"matrix_default_{filename}")
            Image.fromarray(contrast).convert('L').save('contrast/' + filename)


            hist, matrix, k = haralick_matrix(np.uint8(contrast), 1)
            features = calc_features(matrix, k)

            data['filename'].append(filename)
            data['img_type'].append('contrast')
            data['ASM'].append(features['ASM'])
            data['ENT'].append(features['ENT'])
            data['MPR'].append(features['MPR'])
            data['TR'].append(features['TR'])


            plt.bar(np.arange(hist.size), hist)
            plt.savefig('output/' + f"{filename}_contrast_bar.png")
            plt.clf()
            Image.fromarray(matrix/k * 255/np.max(matrix/k)).convert('L').save('output/' + f"matrix_contrast_{filename}")

    
    df = pd.DataFrame(data)
    df = df.set_index('filename')
    df.to_csv('output/result.csv')


                