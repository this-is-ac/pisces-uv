import os
import sys
import cv2
import numpy as np
import skimage.exposure
import torch
from torchvision import transforms

from model_depthv2 import HDRPointwiseNN_depthv2
from utils import load_image, resize, load_params
import matplotlib.pyplot as plt

from tqdm import tqdm
import glob
# from torchsummary import summary
import time

def test(ckpt, params):

    if not params["legacy"]:
        model = HDRPointwiseNN_depthv2(params=params)
        model.load_state_dict(torch.load(ckpt))
    else:
        state_dict, _ = load_params(torch.load(ckpt))
        model = HDRPointwiseNN_depthv2(params=params)
        model.load_state_dict(state_dict)

    device = torch.device("cuda")

    os.makedirs(params["output"], exist_ok=True)

    device = torch.device("cuda")
    tensor = transforms.Compose([transforms.ToTensor(),])

    if os.path.isdir(params['input']):
        test_files = glob.glob(params['input']+'/*')
    else:
        test_files = [params['input']]

    t_100 = 0
    times = []
    for img_path in tqdm(test_files):
        img_name = img_path.split('/')[-1]

        low = tensor(resize(load_image(img_path),params['net_input_size'],strict=True).astype(np.float32)).repeat(1,1,1,1)/255
        full = tensor(load_image(img_path).astype(np.float32)).repeat(1,1,1,1)/255

        low = low.to(device)
        full = full.to(device)
        with torch.no_grad():
            model.eval()
            model.to(device)
            
            start = time.time()
            res, _ = model(low, full)
            times.append(time.time() - start)

            res = torch.clip(res, min=full, max=torch.ones(full.shape).to(device))
            img =  torch.div(full, torch.add(res, 0.001))

            res = (res.cpu().detach().numpy()).transpose(0,2,3,1)[0]
            res = skimage.exposure.rescale_intensity(res, out_range=(0.0,255.0)).astype(np.uint8)
            res = (res*255.0).astype(np.uint8)

            img = (img.cpu().detach().numpy()).transpose(0,2,3,1)[0]
            img = skimage.exposure.rescale_intensity(img, out_range=(0.0,255.0)).astype(np.uint8)
            cv2.imwrite(os.path.join(params['output'],f'out_{img_name}'), img[...,::-1])

    print("done")
    print(sum(times)/len(times))
    print(sum(times[10:])/len(times[10:]))
    print(sum(times[1:])/len(times[1:]))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='HDRNet Inference')
    parser.add_argument('--checkpoint', type=str, default='/home/manogna/aditya/ViTMAE/hdrnet/checkpoints/hdr_depth_v2/ckpt_99_1399.pth',  help='model state path')
    parser.add_argument('--input', type=str, default='/home/aditya/llve/StableLLVE/data/train', help='image path')
    parser.add_argument('--output', type=str, default='/home/manogna/aditya/ViTMAE/hdrnet/datasets/real/test_outputs/hdr_depthv2' , help='output image path')
    parser.add_argument('--legacy', default=False, type=bool, help='old phishnet')

    parser.add_argument('--bs', default=12, type=int, help='[train] batch size(default: 1)')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate (default: 1e-4)')
    parser.add_argument('--gpu', default='0', type=str, help='GPU id to use (default: 0)')
    parser.add_argument('--log', default=None, type=str, help='folder to log')
    parser.add_argument('--weight', default=20, type=float, help='weight of consistency loss')

    parser.add_argument('--resume', type=bool, default=False, help='Continue training from latest checkpoint')
    parser.add_argument('--ckpt-path', type=str, default='', help='Model checkpoint path')
    parser.add_argument('--test-image', type=str,default='/home/aditya/llve/StableLLVE/data/train/10_img_.png', dest="test_image", help='Test image path')
    parser.add_argument('--test-out', type=str, default='/home/aditya/llve/StableLLVE/results/test', dest="test_out", help='Output test image path')

    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--ckpt-interval', type=int, default=100)

    parser.add_argument('--luma-bins', type=int, default=8)
    parser.add_argument('--channel-multiplier', default=1, type=int)
    parser.add_argument('--spatial-bin', type=int, default=16)
    parser.add_argument('--guide-complexity', type=int, default=16)
    parser.add_argument('--batch-norm', action='store_true', help='If set use batch norm')
    parser.add_argument('--net-input-size', type=int, default=256, help='Size of low-res input')
    parser.add_argument('--net-output-size', type=int, default=512, help='Size of full-res input/output')


    args = vars(parser.parse_args())

    test(args['checkpoint'], args)

# python test_PhISHVideo.py --input /home/aditya/llve/StableLLVE/uw_video_2/train --checkpoint /home/aditya/llve/StableLLVE/checkpoints/PhISHNet.pth --output /home/aditya/llve/StableLLVE/outputs/newvid_PhishNet/ --legacy True
# python test_PhISHVideo.py --input /home/aditya/llve/StableLLVE/uw_video_2/train --checkpoint /home/aditya/llve/StableLLVE/logs/PHiSHVideo_01_K10/checkpoint-49.pth --output /home/aditya/llve/StableLLVE/outputs/newvid_PhishVideo/