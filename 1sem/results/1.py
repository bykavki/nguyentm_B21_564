import numpy as np
import os

from PIL import Image

def resample(image, scale):
    height, width, channels = image.shape
    new_width = round(scale * width)
    new_height = round(scale * height)
    
    resampled = np.zeros(shape=(new_height, new_width, channels))
    
    for x in range(new_height):
        for y in range(new_width):
            resampled[x, y] = image[
                min(round(x / scale), height - 1),
                min(round(y / scale), width - 1)
            ]   
    
    return resampled

if __name__ == "__main__":
    names = ["muar", "color"]
    
    tasks = ["interpolation", "decimation", "two_step", "one_step"]
    
    scales = [3, 1/5, (3, 5), 3/5]
    

    '''
    for scale, task in zip(scales, tasks):
        for name in names:
            if type(scale) != tuple:
                image = np.array(Image.open(f"src/{name}.png").convert('RGB'))
                resampled = resample(image, scale)

                resampled_image = Image.fromarray(resampled.astype(np.uint8), 'RGB')
                resampled_image.save(f"result/{name}_{task}.png")
            else:
                image = np.array(Image.open(f"src/{name}.png").convert('RGB'))
                resampled1 = resample(image, scale[0])
                resampled2 = resample(resampled1, 1/scale[1])

                resampled_image = Image.fromarray(resampled2.astype(np.uint8), 'RGB')
                resampled_image.save(f"result/{name}_{task}.png")
    '''
    '''
    image = np.array(Image.open(f"src/muar.png").convert('RGB'))
    resampled = resample(image, 3)
    resampled_image = Image.fromarray(resampled.astype(np.uint8), 'RGB')
    resampled_image.save(f"test.png")
    '''
    for file in os.listdir('input'):
        filename = os.fsdecode(file)
        name = filename.split('.')[0]
        if filename.endswith('.png'):
            for scale, task in zip(scales, tasks):
                if type(scale) != tuple:
                    image = np.array(Image.open(f"input/{name}.png").convert('RGB'))
                    resampled = resample(image, scale)

                    resampled_image = Image.fromarray(resampled.astype(np.uint8), 'RGB')
                    resampled_image.save(f"output/{name}_{task}.png")
                else:
                    image = np.array(Image.open(f"input/{name}.png").convert('RGB'))
                    resampled1 = resample(image, scale[0])
                    resampled2 = resample(resampled1, 1/scale[1])

                    resampled_image = Image.fromarray(resampled2.astype(np.uint8), 'RGB')
                    resampled_image.save(f"output/{name}_{task}.png")
                
