import os
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from load_raw import preprocess_depthmap, load_depthmap
from torchvision.transforms import functional as F
import numpy as np
from load_raw import estimate_far, load_image

def load_image(image_path, max_side = 1024):
    image_file = Image.open(image_path)
    image_file.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return np.float64(image_file) / 255.0

def preprocess_depthmap(image, depths):
    far = estimate_far(image)
    ratio = far

    return depths * ratio

def load_depthmap(depthmap_path, size):
    depth_file = Image.open(depthmap_path)
    depths = depth_file.resize(size, Image.Resampling.LANCZOS)
    depths = np.float64(depths)/np.max(depths)
    return depths

class HDRDataset_depth(Dataset):
    def __init__(self, data_dir='./data', params=None, suffix='', aug=False, split='train'):
        self.data_dir = data_dir
        self.suffix = suffix
        self.aug = aug

        self.in_files = open(f'{data_dir}/splits/{split}_list.txt', 'r').read().splitlines()

        ls = params['net_input_size']
        fs = params['net_output_size']
        self.ls, self.fs = ls, fs
        self.low = transforms.Compose([
            transforms.Resize((ls,ls), Image.BICUBIC),
            transforms.ToTensor()
        ])
        self.correction = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0),
        ])
        self.out = transforms.Compose([
            transforms.Resize((fs,fs), Image.BICUBIC),
            transforms.ToTensor()
        ])
        self.full = transforms.Compose([
            transforms.Resize((fs,fs), Image.BICUBIC),
            transforms.ToTensor()
        ])

        self.depth_trans = transforms.Compose([
            transforms.Resize((fs,fs), Image.BICUBIC),
            # transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, idx):
        fname = self.in_files[idx]
        imagein = Image.open(os.path.join(self.data_dir, 'train'+self.suffix, fname)).convert('RGB')
        imageout = Image.open(os.path.join(self.data_dir, 'gt'+self.suffix, fname)).convert('RGB')

        if self.aug:
            imagein = self.correction(imagein)
        imagein_low = self.low(imagein)
        imagein_full = self.full(imagein)
        imageout = self.out(imageout)

        im = load_image(os.path.join(self.data_dir, 'train'+self.suffix, fname), max_side=self.fs )
        depth = preprocess_depthmap(im, load_depthmap(os.path.join(self.data_dir, 'depthmaps'+self.suffix, fname), (self.fs, self.fs) ))

        flow_low = np.load(os.path.join(os.path.join(self.data_dir, 'flow_low'+self.suffix, fname[:-3] + 'npy'))).astype(np.float32).transpose([2,0,1])
        flow_full = np.load(os.path.join(os.path.join(self.data_dir, 'flow'+self.suffix, fname[:-3] + 'npy'))).astype(np.float32).transpose([2,0,1])

        depth = torch.Tensor(depth)

        return imagein_low, imagein_full, imageout, depth, flow_low, flow_full

    def list_files(self, in_path):
        files = []
        for (dirpath, dirnames, filenames) in os.walk(in_path):
            files.extend(filenames)
            break
        files = sorted([os.path.join(in_path, x) for x in files])
        return files

class UWDataset_depth(Dataset):
    def __init__(self, data_dir='./uw_video', params=None, suffix='', aug=False, split='train'):
        self.data_dir = data_dir
        self.suffix = suffix
        self.aug = aug

        self.in_files = open(f'{data_dir}/splits/{split}_list.txt', 'r').read().splitlines()

        ls = params['net_input_size']
        fs = params['net_output_size']
        self.ls, self.fs = ls, fs
        self.low = transforms.Compose([
            transforms.Resize((ls,ls), Image.BICUBIC),
            transforms.ToTensor()
        ])
        self.correction = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0),
        ])
        self.out = transforms.Compose([
            transforms.Resize((fs,fs), Image.BICUBIC),
            transforms.ToTensor()
        ])
        self.full = transforms.Compose([
            transforms.Resize((fs,fs), Image.BICUBIC),
            transforms.ToTensor()
        ])

        self.depth_trans = transforms.Compose([
            transforms.Resize((fs,fs), Image.BICUBIC),
            # transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, idx):
        fname = self.in_files[idx]
        imagein = Image.open(os.path.join(self.data_dir, 'train'+self.suffix, fname)).convert('RGB')
        # imageout = Image.open(os.path.join(self.data_dir, 'gt'+self.suffix, fname)).convert('RGB')

        if self.aug:
            imagein = self.correction(imagein)
        imagein_low = self.low(imagein)
        imagein_full = self.full(imagein)
        # imageout = self.out(imageout)

        im = load_image(os.path.join(self.data_dir, 'train'+self.suffix, fname), max_side=self.fs )
        depth = preprocess_depthmap(im, load_depthmap(os.path.join(self.data_dir, 'depthmaps'+self.suffix, fname), (self.fs, self.fs) ))

        flow_low = np.load(os.path.join(os.path.join(self.data_dir, 'flow_low'+self.suffix, fname[:-3] + 'npy'))).astype(np.float32).transpose([2,0,1])
        flow_full = np.load(os.path.join(os.path.join(self.data_dir, 'flow'+self.suffix, fname[:-3] + 'npy'))).astype(np.float32).transpose([2,0,1])

        depth = torch.Tensor(depth)

        return imagein_low, imagein_full, depth, flow_low, flow_full

    def list_files(self, in_path):
        files = []
        for (dirpath, dirnames, filenames) in os.walk(in_path):
            files.extend(filenames)
            break
        files = sorted([os.path.join(in_path, x) for x in files])
        return files