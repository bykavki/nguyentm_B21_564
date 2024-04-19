import numpy as np
import os

from PIL import Image

def semitone(image):
    return (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(np.int8)

if __name__ == "__main__":
    for file in os.listdir('input'):
        filename = os.fsdecode(file)
        if filename.endswith('.png'):
            img = np.array(Image.open('input/' + filename).convert('RGB'))
            img_sem = semitone(img)
            Image.fromarray(img_sem).convert('RGB').save('output/' + filename)

