import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import os

from estimate_backscatter import estimate_video_coefficients, estimate_video_backscatter
from load_raw import load_image, load_depthmap, preprocess_depthmap
import time

from random import sample

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, required=True, help='Path for Input Images')
    parser.add_argument('--depthmap-path', type=str, required=True, help='Path for Depthmaps corresponding to the Input Images')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save the backscatter removed images')
    args = parser.parse_args()

    dirs = os.listdir(args.image_path)

    print("Detected {} files in directory {}\n".format(len(dirs), args.image_path))

    K = 10
    coeff_est_files = sample(os.listdir(args.image_path), K)

    coeff_est_images = []
    coeff_est_depths = []

    for file in coeff_est_files:
        f = os.path.join(args.image_path, file)        
        if f[-3:].lower() == "png" or f[-3:].lower() == "jpg"  or f[-3:].lower() == "PEG":
            t = time.time()
            image = load_image(f)
            depths = preprocess_depthmap(image, load_depthmap(os.path.join(args.depthmap_path, file.split('.')[0]+'.png'), (image.shape[1], image.shape[0])))
            
            coeff_est_images.append(image)
            coeff_est_depths.append(depths)

    opt_coeffs = estimate_video_coefficients(coeff_est_images, coeff_est_depths)

    print("Estimated Coefficients")

    for file in tqdm(dirs, desc = 'dirs'):
        f = os.path.join(args.image_path, file)
        if f[-3:].lower() == "png" or f[-3:].lower() == "jpg"  or f[-3:].lower() == "PEG":
            t = time.time()
            image = load_image(f)
            depths = preprocess_depthmap(image, load_depthmap(os.path.join(args.depthmap_path, file.split('.')[0]+'.png'), (image.shape[1], image.shape[0])))

            Ba = estimate_video_backscatter(depths, opt_coeffs)

            Da = image - Ba
            Da = np.clip(Da, 0, 1)
            D = np.uint8(Da * 255.0)
            backscatter_removed = Image.fromarray(D)
            backscatter_removed.save(args.output_path + file.split('.')[0]+'.png')

    print("Done!")

# python preprocess_video.py --image-path /home/aditya/llve/StableLLVE/uw_video_extra/images/ --depthmap-path /home/aditya/llve/StableLLVE/uw_video_extra/depthmaps/ --output-path /home/aditya/llve/StableLLVE/uw_video_extra/train_video/