from PIL import Image
import numpy as np

def semitone(image):
    return (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(np.int8)

if __name__ == "__main__":
    file="84_3.jpeg"
    src = Image.open(file).convert('RGB')
    res = semitone(np.array(src))
    Image.fromarray(res).show()
