import argparse
from enum import Enum
from skimage.transform import resize
import numpy as np
from skimage.util import random_noise
from PIL import Image
import os
from pathlib import Path


"""
Logic is from https://stackoverflow.com/a/71822506
Adding random nose in an even distribution across the image
Doing it in 3 layers helps creating variing size of grain, as we downscale and then upscale 
Poission distribution 
"""
def gen_noise_mask(rows, cols, intf, intm, intl, mode):
    # Full resolution
    noise_im1 = np.zeros((rows, cols))
    if mode == Modes.poisson.value: 
        noise_im1 = random_noise(noise_im1, mode=mode, clip=False)
    elif mode == Modes.gaussian.value:
        noise_im1 = random_noise(noise_im1, mode=mode, var=intf ** 2, clip=False)


    # Half resolution
    noise_im2 = np.zeros((rows // 2, cols // 2))
    if mode == Modes.poisson.value: 
        noise_im2 = random_noise(noise_im2, mode=mode, clip=False)
    elif mode == Modes.gaussian.value:
        noise_im2 = random_noise(noise_im2, mode=mode, var=intm ** 2, clip=False)
    noise_im2 = resize(noise_im2, (rows, cols))  # Upscale to original image size

    # Quarter resolution
    noise_im3 = np.zeros((rows // 4, cols // 4))
    if mode == Modes.poisson.value: 
        noise_im3 = random_noise(noise_im3, mode=mode, clip=False)
    elif mode == Modes.gaussian.value:
        noise_im3 = random_noise(noise_im3, mode=mode, var=intl ** 2, clip=False)
    noise_im3 = resize(noise_im3, (rows, cols))  # What is the interpolation method?

    noise_im = noise_im1 + noise_im2 + noise_im3  # Sum the noise in multiple resolutions (the mean of noise_im is around zero).
    return noise_im


def noiseGenerator(im: Image, intf, intm, intl, mode):
    im_arr = np.asarray(im)

    rows, cols, depth = im_arr.shape

    rgba_array = np.zeros((rows, cols, depth), 'float64')
    for d in range(0, depth): ## depth is the number of color channels, will typically be 3 for RGB, but could also contain alpha
        rgba_array[..., d] += gen_noise_mask(rows, cols, intf, intm, intl, mode)
    noisy_img = im_arr / 255 + rgba_array  # Add noise_im to the input image.
    noisy_img = np.round((255 * noisy_img)).clip(0, 255).astype(np.uint8) # clip image to 0-255 bounds
    return Image.fromarray(noisy_img)


class Modes(Enum):
    gaussian='gaussian'
    poisson='poisson'

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Grainer script to add a bit of character to your photos")
    parser.add_argument("-p", "--path", type=str, help="path to image")
    parser.add_argument("-if", "--intensity_fine", type=float, default=0.01, help="Intensity of the fine grain, unused by poission")
    parser.add_argument("-im", "--intensity_medium", type=float, default=0.01, help="Intensity of the medium grain, unused by poission")
    parser.add_argument("-il", "--intensity_large", type=float, default=0.01, help="Intensity of the large grain, unused by poission")
    parser.add_argument("-m", "--mode", type=Modes, default=Modes.gaussian, help="How to apply noise to image, gaussian, poisson or localvar")

    args = parser.parse_args()

    path = Path(args.path)

    dir = path.parent
    fileExtension = path.suffix.lower()
    if not fileExtension.endswith(".jpeg") and not fileExtension.endswith(".png") and not fileExtension.endswith(".jpg"): 
        raise TypeError("Given file path is not a compatible image. Must be .jpeg, .png or .jpg")
    
    image = Image.open(args.path)
    fine_intensity = args.intensity_fine
    medium_intensity = args.intensity_medium
    large_intensity = args.intensity_large
    mode = args.mode.value
    

    new_image = noiseGenerator(im=image, intf=fine_intensity, intm=medium_intensity, intl=large_intensity, mode=mode)


    new_file_name = f'{path.stem}-grained{path.suffix}'
    new_path = path.parent / new_file_name
    print(f'Saving new image at {new_path}')

    new_image.save(fp=new_path)









