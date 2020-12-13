# -*- coding: utf-8 -*-

# import libraries
import numpy as np
import os
import torch.utils.data as data
from PIL import Image
import random
import torch


"""# Load Dataset"""

class CoordsDataset(data.Dataset):

    # def __init__(self, root='./', load_set='train', transform=None, with_object=False, isSTB=True):
    def __init__(self, root='./', load_set='train'):
        self.root = root#os.path.expanduser(root)
        self.load_set = load_set  # 'train','val','test', 'pre_train'

        self.images = np.load(os.path.join(root, 'images-%s.npy'%self.load_set))
        self.points2d = np.load(os.path.join(root, 'points2d-%s.npy'%self.load_set))
        self.points3d = np.load(os.path.join(root, 'points3d-%s.npy'%self.load_set))
        self.scale = np.load(os.path.join(root, 'scale-%s.npy'%self.load_set))
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, points2D, points3D).
        """
        
        image = Image.open(self.images[index])
        im_h = image.size[0]
        scale_label = 1.0 / self.scale[index]
        
        point2d = self.points2d[index]/im_h*256
        point2d = point2d + np.random.normal(0, 10, point2d.shape) # add gaussion noise (mean=0, σ=10)
        point2d = np.clip(point2d, 0, 256) # clip the 2d coords in [0,256]
        point3d = self.points3d[index]

        return point2d, point3d, scale_label

    def __len__(self):
        return len(self.images)

class GenerateHeatmap():
    def __init__(self, output_res, num_parts):
        self.output_res = output_res
        self.num_parts = num_parts
        sigma = self.output_res/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, keypoints):
        hms = np.zeros(shape = (self.num_parts, self.output_res, self.output_res), dtype = np.float32)
        sigma = self.sigma
        for idx, pt in enumerate(keypoints):
            if pt[0] > 0: 
                x, y = int(pt[0]), int(pt[1])
                if x<0 or y<0 or x>=self.output_res or y>=self.output_res:
                    continue
                ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
                br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

                c,d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                a,b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                cc,dd = max(0, ul[0]), min(br[0], self.output_res)
                aa,bb = max(0, ul[1]), min(br[1], self.output_res)
                hms[idx, aa:bb,cc:dd] = np.maximum(hms[idx, aa:bb,cc:dd], self.g[a:b,c:d])
        return hms