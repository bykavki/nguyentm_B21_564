from semitone import semitone
from PIL import Image

import numpy as np

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
    #semitone_img = image
    semitone_img = semitone(image)
    threshold = get_threshold(semitone_img)

    new_image = np.zeros(semitone_img.shape)
    new_image[semitone_img > threshold] = 0
    new_image[semitone_img <= threshold] = 255

    return new_image.astype(np.uint8)

if __name__ == '__main__':
    file="84_3.png"
    src = Image.open(file).convert('RGB')
    res = Otsu_binarization(np.array(src))
    print(res)
    Image.fromarray(res).show()
