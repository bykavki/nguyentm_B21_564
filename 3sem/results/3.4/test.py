from PIL import Image

import numpy as np

input = np.array(Image.open('input.png').convert('L'))
output = np.array(Image.open('output.png').convert('L'))

w, l = input.shape
count = 0
for i in range(w):
    for j in range(l):
        if input[i][j] != output[i][j]:
            count += 1
print(count)