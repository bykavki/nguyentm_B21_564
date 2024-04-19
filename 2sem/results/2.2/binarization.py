import numpy as np
import os

from semitone import semitone
from PIL import Image


def intra_class_variance(image, threshold):
    return np.nansum([
		np.mean(cls) * np.var(image,where=cls)
		for cls in [image>=threshold,image<threshold]
	])
 

def get_threshold(image):
    return min(
		range( np.min(image)+1, np.max(image) ),
		key = lambda th: intra_class_variance(image,th)
	)


def Otsu_binarization(image):
    semitone_img = semitone(image)
    threshold = get_threshold(semitone_img)

    new_image = np.zeros(semitone_img.shape)
    new_image[semitone_img > threshold] = 0
    new_image[semitone_img <= threshold] = 255

    return new_image.astype(np.uint8)

if __name__ == '__main__':
    for file in os.listdir('input'):
        filename = os.fsdecode(file)
        if filename.endswith('.png'):
            img = np.array(Image.open('input/' + filename).convert('RGB'))
            img_sem = Otsu_binarization(img)
            Image.fromarray(img_sem).convert('RGB').save('output/' + filename)
