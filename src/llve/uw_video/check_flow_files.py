# check if the next file is available to find backward flow

import os

codes = []

for f in os.listdir('/home/aditya/llve/StableLLVE/uw_video_2/images'):
    codes.append(int(f.split('_')[-1][:-4]))

for f in os.listdir('/home/aditya/llve/StableLLVE/uw_video_2/images'):
    if (int(f.split('_')[-1][:-4])) + 1 not in codes:
        print(f)
