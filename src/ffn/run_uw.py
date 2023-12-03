import numpy as np
import cv2
import torch
import torch.nn.functional as F
from models.FastFlowNet_v2 import FastFlowNet
from flow_vis import flow_to_color

from tqdm import tqdm
import os

div_flow = 20.0
div_size = 64

def centralize(img1, img2):
    b, c, h, w = img1.shape
    rgb_mean = torch.cat([img1, img2], dim=2).view(b, c, -1).mean(2).view(b, c, 1, 1)
    return img1 - rgb_mean, img2 - rgb_mean, rgb_mean

model = FastFlowNet().cuda().eval()
model.load_state_dict(torch.load('./checkpoints/fastflownet_ft_mix.pth'))

img_files = sorted(os.listdir('/home/aditya/llve/StableLLVE/uw_video_2/images'))

txt_file = open('/home/aditya/llve/StableLLVE/uw_video_2/splits/all_list.txt','w')
for item in img_files[:-1]:
	txt_file.write(item+"\n")
txt_file.close()

for img_index in tqdm(range(len(img_files)-1)):
    img1_path = os.path.join('/home/aditya/llve/StableLLVE/uw_video_2/images', img_files[img_index+1])
    img2_path = os.path.join('/home/aditya/llve/StableLLVE/uw_video_2/images', img_files[img_index])

    # Orig Resolution

    img1 = torch.from_numpy(cv2.imread(img1_path)).float().permute(2, 0, 1).unsqueeze(0)/255.0
    img2 = torch.from_numpy(cv2.imread(img2_path)).float().permute(2, 0, 1).unsqueeze(0)/255.0
    img1, img2, _ = centralize(img1, img2)

    height, width = img1.shape[-2:]
    orig_size = (int(height), int(width))

    if height % div_size != 0 or width % div_size != 0:
        input_size = (
            int(div_size * np.ceil(height / div_size)), 
            int(div_size * np.ceil(width / div_size))
        )
        img1 = F.interpolate(img1, size=input_size, mode='bilinear', align_corners=False)
        img2 = F.interpolate(img2, size=input_size, mode='bilinear', align_corners=False)
    else:
        input_size = orig_size

    input_t = torch.cat([img1, img2], 1).cuda()

    output = model(input_t).data

    flow = div_flow * F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)

    if input_size != orig_size:
        scale_h = orig_size[0] / input_size[0]
        scale_w = orig_size[1] / input_size[1]
        flow = F.interpolate(flow, size=orig_size, mode='bilinear', align_corners=False)
        flow[:, 0, :, :] *= scale_w
        flow[:, 1, :, :] *= scale_h

    flow = flow[0].cpu().permute(1, 2, 0).numpy()

    np.save('/home/aditya/llve/StableLLVE/uw_video_2/flow/{}.npy'.format(img_files[img_index].split('.')[0]), flow)

    # Low Res

    img1 = torch.from_numpy(cv2.resize(cv2.imread(img1_path), (256,256))).float().permute(2, 0, 1).unsqueeze(0)/255.0
    img2 = torch.from_numpy(cv2.resize(cv2.imread(img2_path), (256,256))).float().permute(2, 0, 1).unsqueeze(0)/255.0
    img1, img2, _ = centralize(img1, img2)

    height, width = img1.shape[-2:]
    orig_size = (int(height), int(width))

    if height % div_size != 0 or width % div_size != 0:
        input_size = (
            int(div_size * np.ceil(height / div_size)), 
            int(div_size * np.ceil(width / div_size))
        )
        img1 = F.interpolate(img1, size=input_size, mode='bilinear', align_corners=False)
        img2 = F.interpolate(img2, size=input_size, mode='bilinear', align_corners=False)
    else:
        input_size = orig_size

    input_t = torch.cat([img1, img2], 1).cuda()

    output = model(input_t).data

    flow = div_flow * F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)

    if input_size != orig_size:
        scale_h = orig_size[0] / input_size[0]
        scale_w = orig_size[1] / input_size[1]
        flow = F.interpolate(flow, size=orig_size, mode='bilinear', align_corners=False)
        flow[:, 0, :, :] *= scale_w
        flow[:, 1, :, :] *= scale_h

    flow = flow[0].cpu().permute(1, 2, 0).numpy()

    np.save('/home/aditya/llve/StableLLVE/uw_video_2/flow_low/{}.npy'.format(img_files[img_index].split('.')[0]), flow)