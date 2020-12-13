# Copyright (c) Liuhao Ge. All Rights Reserved.
r"""
Networks for heatmap estimation from RGB images using Hourglass Network
"Stacked Hourglass Networks for Human Pose Estimation", Alejandro Newell, Kaiyu Yang, Jia Deng, ECCV 2016
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn



class Hourglass(nn.Module):
    def __init__(self, n, nModules, nFeats):
        super(Hourglass, self).__init__()
        self.n = n
        self.nModules = nModules
        self.nFeats = nFeats

        _up1_, _low1_, _low2_, _low3_ = [], [], [], []
        for j in range(self.nModules):
            _up1_.append(Residual(self.nFeats, self.nFeats))
        self.low1 = nn.MaxPool2d(kernel_size=2, stride=2)
        for j in range(self.nModules):
            _low1_.append(Residual(self.nFeats, self.nFeats))

        if self.n > 1:
            self.low2 = Hourglass(n - 1, self.nModules, self.nFeats)
        else:
            for j in range(self.nModules):
                _low2_.append(Residual(self.nFeats, self.nFeats))
            self.low2_ = nn.ModuleList(_low2_)

        for j in range(self.nModules):
            _low3_.append(Residual(self.nFeats, self.nFeats))

        self.up1_ = nn.ModuleList(_up1_)
        self.low1_ = nn.ModuleList(_low1_)
        self.low3_ = nn.ModuleList(_low3_)

        self.up2 = nn.Upsample(scale_factor=2)

    def forward(self, x):
        '''
            the input x is 64*64*256 feature maps
        '''
        up1 = x
        for j in range(self.nModules):
            up1 = self.up1_[j](up1)
        # print('up1:',up1.shape)

        low1 = self.low1(x)
        for j in range(self.nModules):
            low1 = self.low1_[j](low1)
        # print('low1:',low1.shape)
        if self.n > 1:
            low2 = self.low2(low1)
            # print('n>1 low2:',low2.shape)
        else:
            low2 = low1
            for j in range(self.nModules):
                low2 = self.low2_[j](low2)
                # print('n=1 low2:',low2.shape)

        # print('low2:',low2.shape)
        low3 = low2
        for j in range(self.nModules):
            low3 = self.low3_[j](low3)
            # print('low3:',low2.shape)
        up2 = self.up2(low3)
        
        return up1 + up2

class Net_HM_HG(nn.Module):
    def __init__(self, num_joints, num_stages=2, num_modules=2, num_feats=256):
        super(Net_HM_HG, self).__init__()

        self.numOutput = num_joints
        self.nStack = num_stages

        self.nModules = num_modules
        self.nFeats = num_feats

        self.conv1_ = nn.Conv2d(3, 64, bias=True, kernel_size=7, stride=2, padding=3) #256*256->128*128
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.r1 = Residual(64, 128)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) #128*128->64*64
        self.r4 = Residual(128, 128)
        self.r5 = Residual(128, self.nFeats)

        _hourglass, _Residual, _lin_, _tmpOut, _ll_, _tmpOut_, _reg_ = [], [], [], [], [], [], []
        for i in range(self.nStack):
            _hourglass.append(Hourglass(4, self.nModules, self.nFeats))
            # _hourglass.append(Hourglass(3, self.nModules, self.nFeats))
            for j in range(self.nModules):
                _Residual.append(Residual(self.nFeats, self.nFeats))
            lin = nn.Sequential(nn.Conv2d(self.nFeats, self.nFeats, bias=True, kernel_size=1, stride=1),
                                nn.BatchNorm2d(self.nFeats), self.relu)
            _lin_.append(lin)
            _tmpOut.append(nn.Conv2d(self.nFeats, self.numOutput, bias=True, kernel_size=1, stride=1))
            if i < self.nStack - 1:
                _ll_.append(nn.Conv2d(self.nFeats, self.nFeats, bias=True, kernel_size=1, stride=1))
                _tmpOut_.append(nn.Conv2d(self.numOutput, self.nFeats, bias=True, kernel_size=1, stride=1))

        self.hourglass = nn.ModuleList(_hourglass)
        self.Residual = nn.ModuleList(_Residual)
        self.lin_ = nn.ModuleList(_lin_)
        self.tmpOut = nn.ModuleList(_tmpOut)
        self.ll_ = nn.ModuleList(_ll_)
        self.tmpOut_ = nn.ModuleList(_tmpOut_)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.heatmapLoss = HeatmapLoss()

    def forward(self, x):
        '''
        input:
            the input x is 256*256*3 RGB image
        output:
            out: n_stage*K*64*64 heatmaps of all hourglass stage
            encoding: n_stage*256*64*64 feature maps of all hourglass stage
        '''
        x = self.conv1_(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.r1(x)
        x = self.maxpool(x)
        x = self.r4(x)
        x = self.r5(x)

        out = []
        encoding = []

        for i in range(self.nStack):
            hg = self.hourglass[i](x)
            ll = hg
            for j in range(self.nModules):
                ll = self.Residual[i * self.nModules + j](ll) # 经过HG module之后经过Res module
            ll = self.lin_[i](ll) # 然后经过1x1 conv + BN + ReLU
            tmpOut = self.tmpOut[i](ll) # 经1x1conv得 K*64*64(num_joints heatmap)
            out.append(tmpOut)
            if i < self.nStack - 1:
                ll_ = self.ll_[i](ll) #上边分支，ll经1x1 conv
                tmpOut_ = self.tmpOut_[i](tmpOut) # 下面分支，经1x1 conv
                x = x + ll_ + tmpOut_ # 上下分支和Res融合(三个进行点加)，送给下一个阶段
                # encoding.append(x)
                x_avg_pool = self.avgpool(x)
                encoding.append(x_avg_pool.squeeze())
            else:
                # encoding.append(ll)
                x_avg_pool = self.avgpool(ll)
                encoding.append(x_avg_pool.squeeze())

        # coord_2d_ini = self.soft_argmax_2d(out[-1])
        # coord_2d_ini = coord_2d_ini * (256/64)
        return out, torch.cat(encoding, dim=-1)
        # return out, encoding #所有阶段的中间heatmap输出；所有阶段的featumaps (num_stage, B, #joint, 64,64) (num_stage ,B, 256, 1, 1)
        # return coord_2d_ini, out, torch.cat(encoding, dim=-1) #(#stage, B, #joint, 64, 64); (B, 256*#stage)

    def calc_loss(self, combined_hm_preds, heatmaps):
        combined_loss = []
        for i in range(self.nStack):
            combined_loss.append(self.heatmapLoss(combined_hm_preds[0][:,i], heatmaps))
        combined_loss = torch.stack(combined_loss, dim=1)
        return combined_loss

    # def soft_argmax_2d(self, features, device=torch.device('cuda:2')):
    def soft_argmax_2d(self, features, device=torch.device('cuda')):
        """
        Arguments: voxel patch in shape (batch_size, channel, H, W)
        Return: x,y coordinates in shape (batch_size, channel, 2)
        """
        assert features.dim()==4
        # alpha is here to make the largest element really big, so it
        # would become very close to 1 after softmax
        alpha = 1000.0 
        N,C,H,W = features.shape
        soft_max = nn.functional.softmax(features.contiguous().view(N,C,-1)*alpha,dim=2)
        soft_max = soft_max.contiguous().view(features.shape)
        indices_kernel = torch.arange(start=0,end=H*W).unsqueeze(0).to(device)
        indices_kernel = indices_kernel.contiguous().view((H,W))
        conv = soft_max*indices_kernel.float()
        indices = conv.sum(2).sum(2)
        # y = indices%W
        # x = (indices/W).floor()%H
        # coords = torch.stack([x,y],dim=2)
        y = indices%W + 1
        x = (indices/W).floor()%H + 1
        coords = torch.stack([y,x],dim=2)
        return coords



class HeatmapLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        l = ((pred - gt)**2)
        l = l.mean(dim=3).mean(dim=2).mean(dim=1)
        return l ## l of dim bsize

class Residual(nn.Module):
    def __init__(self, numIn, numOut):
        super(Residual, self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        self.bn = nn.BatchNorm2d(self.numIn)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.numIn, self.numOut // 2, bias=True, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.numOut // 2)
        self.conv2 = nn.Conv2d(self.numOut // 2, self.numOut // 2, bias=True, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.numOut // 2)
        self.conv3 = nn.Conv2d(self.numOut // 2, self.numOut, bias=True, kernel_size=1)

        if self.numIn != self.numOut:
            self.conv4 = nn.Conv2d(self.numIn, self.numOut, bias=True, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.numIn != self.numOut:
            residual = self.conv4(x)

        return out + residual

if __name__ == "__main__":
    # import cv2
    import torch
    from PIL import Image
    from torchvision import transforms
    img = Image.open('../test-image/test_HG.png').convert('RGB')
    transform = transforms.Compose([transforms.CenterCrop(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    net = Net_HM_HG(10)
    out, encoding = net(img)
    print(len(out),out[0].shape, len(encoding),encoding[0].shape)
