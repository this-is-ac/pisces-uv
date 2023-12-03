import argparse
import os, socket
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from warp import WarpingLayerBWFlow

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import cv2
import random

import skimage.exposure
from torchvision import transforms
import glob

from utils import load_image, resize
from metrics import psnr
from dataset import HDRDataset_depth
from model_depthv2 import HDRPointwiseNN_depthv2, L2LOSS

def save_checkpoint(state, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth')
    torch.save(state, checkpoint_filename)

def test(ckpt, params = {}, epoch={}):
    model = HDRPointwiseNN_depthv2(params=params)
    model.load_state_dict(torch.load(ckpt))
    device = torch.device("cuda")

    model.eval()
    model.to(device)

    tensor = transforms.Compose([
        transforms.ToTensor(),])

    if os.path.isdir(params['test_image']):
        test_files = glob.glob(params['test_image']+'/*')
    else:
        test_files = [params['test_image']]

    for img_path in test_files:
        img_name = img_path.split('/')[-1]
        print(f'Testing image: {img_name}')
        low = tensor(resize(load_image(img_path),params['net_input_size'],strict=True).astype(np.float32)).repeat(1,1,1,1)/255
        full = tensor(load_image(img_path).astype(np.float32)).repeat(1,1,1,1)/255

        low = low.to(device)
        full = full.to(device)
        with torch.no_grad():
            res, _ = model(low, full)
            res = torch.clip(res, min=full, max=torch.ones(full.shape).to(device))
            img =  torch.div(full, torch.add(res, 0.001))
            res = (res.cpu().detach().numpy()).transpose(0,2,3,1)[0]
            res = skimage.exposure.rescale_intensity(res, out_range=(0.0,255.0)).astype(np.uint8)

            img = (img.cpu().detach().numpy()).transpose(0,2,3,1)[0]
            img = skimage.exposure.rescale_intensity(img, out_range=(0.0,255.0)).astype(np.uint8)
            cv2.imwrite(os.path.join(params['test_out'],f'out_e{epoch.zfill(3)}_{img_name}'), img[...,::-1])
    return

# Parse arguments
parser = argparse.ArgumentParser(description='TC-HDRNet')
parser.add_argument('--data-path', default='./data', type=str, help='path to the dataset')
parser.add_argument('--epochs', default=1, type=int, help='n of epochs (default: 50)')
parser.add_argument('--bs', default=12, type=int, help='[train] batch size(default: 1)')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate (default: 1e-4)')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use (default: 0)')
parser.add_argument('--checkpoint', default=None, type=str, help='path to checkpoint')
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

args = parser.parse_args()
params = vars(parser.parse_args())

print('PARAMS:')
print(params)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

device = torch.device("cuda")

torch.manual_seed(ord('c')+137)
random.seed(ord('c')+137)
np.random.seed(ord('c')+137)

train_set = HDRDataset_depth(params['data_path'], params=params, split='all')
train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True)

model = HDRPointwiseNN_depthv2(params=params)

if params['resume']:
    print('Loading previous state:', params['ckpt_path'])
    model.load_state_dict(torch.load(params['ckpt_path']))

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-7)

criterion = torch.nn.MSELoss()
warp = WarpingLayerBWFlow().cuda()

if args.log==None:
    log_dir = os.path.join(os.path.abspath(os.getcwd()), 'logs', datetime.now().strftime('%b%d_%H-%M-%S_') + socket.gethostname())
else:
    log_dir = os.path.join(os.path.abspath(os.getcwd()), 'logs', args.log)

os.makedirs(log_dir)
logger = SummaryWriter(log_dir)

with open(os.path.join(log_dir, "config.txt"), "a") as f:
    print(args, file=f)

count = 0
for e in range(params['epochs']):
    model.train()

    for i, (low, full, target, depth, flow_low, flow_full) in enumerate(train_loader):

        low, full, target, depth, flow_low, flow_full = low.to(device), full.to(device), target.to(device), depth.to(device), flow_low.to(device), flow_full.to(device)
        optimizer.zero_grad()

        illum, z_params = model(low, full)
        res = torch.clip(illum, min=full, max=torch.ones(full.shape).to(device))
        pred =  torch.div(full, torch.add(res, 0.001))

        loss = criterion(pred, target)

        low_t = warp(low, flow_low)
        full_t = warp(full, flow_full)

        illum_t, z_params_t = model(low_t, full_t)
        res_t = torch.clip(illum_t, min=full_t, max=torch.ones(full_t.shape).to(device))
        input_t_pred =  torch.div(full_t, torch.add(res_t, 0.001))

        pred_t = warp(pred, flow_full)
        
        loss_t = criterion(input_t_pred, pred_t) * args.weight
        total_loss = loss + loss_t

        total_loss.backward()

        if (count+1) % params['log_interval'] == 0:
            _psnr = psnr(res,target).item()
            tloss = total_loss.item()
            print('Epoch: {0} [{1}/{2}]\t loss={tloss:.5f} psnr={psnr:.4f} l1={Loss1:.5f} l2={Loss2:.5f} '.format(e, (count + 1)%len(train_loader), len(train_loader), tloss=tloss, psnr=_psnr, Loss1=loss.item(), Loss2=loss_t.item()))
        
        optimizer.step()
        
        logger.add_scalar('Train/Loss', loss.item(), count)
        logger.add_scalar('Train/Loss_t', loss_t.item(), count)

        count += 1

    save_checkpoint(model.state_dict(), e, log_dir)

    model.eval().cpu()
    test(os.path.join(log_dir, 'checkpoint-' + str(e) + '.pth'), params=params, epoch=str(e))
    model.to(device).train()
    
    print()

logger.close()

# no self-consistency
# python train_PhiSHNet.py --weight 0 --bs 28 --log PHiSHNet --epochs 50 --test-out /home/aditya/llve/StableLLVE/results/PHiSHNet
# python train_PhiSHNet.py --weight 0 --bs 16 --log PHiSHNet_ckpt --resume True --ckpt-path /home/aditya/llve/StableLLVE/logs/short_test/checkpoint-0.pth

# with self-consistency
# python train_PhiSHNet.py --weight 20 --bs 28 --log TC_PHiSHNet --epochs 50 --test-out /home/aditya/llve/StableLLVE/results/TC_PHiSHNet