import numpy as np
from PIL import Image
import argparse
import os
from tqdm import tqdm
from load_raw import load_image
from metrics import ref_based, non_ref_based

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=str, required=True, help='Path for the Model Outputs')
    args = parser.parse_args()

    dirs = os.listdir(args.output_path)

    print("Detected {} files in directory {}\n".format(len(dirs), args.output_path))
    file_names = []
    overall_uciqe = []

    for file in tqdm(dirs):
        f = os.path.join(args.output_path, file)
        if f[-3:].lower() == "png":
            image = load_image(f)
            file_names.append(file[4:-4])

            _,uciqe = non_ref_based(image)
            overall_uciqe.append(uciqe)

    print ("UCIQE  : Mean: {:014.12f} std: {:.12f}".format(np.mean(overall_uciqe), np.std(overall_uciqe)))

# python evaluate_nr.py --output-path /home/aditya/llve/StableLLVE/outputs/PHiSHNet_01_UWVideo