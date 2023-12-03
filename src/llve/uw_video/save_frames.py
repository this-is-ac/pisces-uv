import os
from PIL import Image

from tqdm import tqdm

savedir = '/home/aditya/llve/StableLLVE/uw_video_2/images'

os.makedirs(savedir, exist_ok=True)

# files = [x.split('.')[0] for x in sorted(os.listdir('/home/aditya/llve/StableLLVE/uw_video/orig_images/pool_flipper_001_A'))]

# maxconsec = 0
# consec = 1
# index = 0
# for i in range(len(files)-1):
#     if int(files[i+1]) == int(files[i]) + 1:
#         consec += 1
#     else:
#         consec = 1

#     if consec > maxconsec:
#         maxconsec = consec
#         index = i

# print(maxconsec, index, files[index])

for f in tqdm(sorted(os.listdir('/home/aditya/llve/StableLLVE/uw_video/orig_images/pool_flipper_001_A'))[-401:]):
    img = Image.open(os.path.join('/home/aditya/llve/StableLLVE/uw_video/orig_images/pool_flipper_001_A', f)).resize((512, 512)).convert("RGB")
    img.save(os.path.join(savedir, 'pool_filpper_001_A' + '_' + f[:-3] + 'png'))