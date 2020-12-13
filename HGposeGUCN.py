# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

"""# Import Libraries"""

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from utils.HGposeGUCN_option import parse_args_function
from utils.dataset import Dataset
from utils.metric import *
from tqdm import tqdm
from models.hourglass import HeatmapLoss
from models.HGposeGUCNnet import HGposeGUCN

args = parse_args_function()

"""# Load Dataset"""

root = args.input_file

transform = transforms.Compose([transforms.Resize((256, 256)),
                                transforms.ToTensor()])

if args.train:
    trainset = Dataset(root=root, load_set='train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16, drop_last=True)
    
    print('Train files loaded')

if args.val:
    valset = Dataset(root=root, load_set='val', transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=True)
    
    print('Validation files loaded')

if args.test:
    testset = Dataset(root=root, load_set='test', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=True)
    
    print('Test files loaded')

"""# Model"""

use_cuda = False
if args.gpu:
    use_cuda = True

# model = select_model(args.model_def)
model = HGposeGUCN(chk_poseHG=args.pretrained_posehg, chk_gucn=args.pretrained_gucn)

if use_cuda and torch.cuda.is_available():
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=args.gpu_number)

"""# Load Snapshot"""

if args.pretrained_model != '':
    model.load_state_dict(torch.load(args.pretrained_model))

start = 0

"""# Optimizer"""

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_step_gamma)
scheduler.last_epoch = start
lambda_3d = 5
lambda_2d = 0.001
lambda_hm = 1

"""# pre train"""
def calc_loss(combined_hm_preds, heatmaps):
    heatmapLoss = HeatmapLoss()
    combined_loss = []
    for i in range(2):
        # combined_loss.append(heatmapLoss(combined_hm_preds[0][:,i], heatmaps))
        combined_loss.append(heatmapLoss(combined_hm_preds[i], heatmaps))
    combined_loss = torch.stack(combined_loss, dim=1)
    return combined_loss 

if args.train:
    print('Begin pre_training the network...')
    for epoch in range(start, args.num_epoch):  # loop over the dataset multiple times
        e_dis3d = []
        with tqdm(total=trainloader.__len__(), 
                bar_format='{n_fmt}/{total_fmt} |{bar}| {rate_fmt} {postfix}', ncols=120) as process_bar: 
            train_loss = 0.0

            for i, tr_data in enumerate(trainloader):
                # get the inputs
                imgs, hm, labels2d, labels3d, scale_label = tr_data
                scale_label = np.array(scale_label)
                # wrap them in Variable
                imgs = Variable(imgs)
                hm = Variable(hm)
                labels2d = Variable(labels2d)
                labels3d = Variable(labels3d)
                if use_cuda and torch.cuda.is_available():
                    imgs = imgs.float().cuda(device=args.gpu_number[0])
                    hm = hm.float().cuda(device=args.gpu_number[0])
                    labels2d = labels2d.float().cuda(device=args.gpu_number[0])
                    labels3d = labels3d.float().cuda(device=args.gpu_number[0])
                
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                out_hms, outputs2d, outputs3d = model(imgs)
                loss_hm = calc_loss(out_hms, hm)
                loss_hm = loss_hm.mean()
                loss_2d = criterion(outputs2d, labels2d)
                loss_3d = criterion(outputs3d, labels3d)
                
                loss = (lambda_hm * loss_hm) + (lambda_2d * loss_2d) + (lambda_3d * loss_3d)

                train_loss += loss.data
                loss.backward()
                optimizer.step()

                process_bar.set_postfix_str('loss=%.5f; l_hm=%.5f, l_2d=%.5f, l_3d=%.5f' % 
                                            (loss.data, loss_hm.data, loss_2d.data, loss_3d.data))
                process_bar.update()

                # calculate the distance between label3d and output3d
                b_labels3d = labels3d.reshape(-1, 21, 3) #[B, 21, 3]
                b_out3d = outputs3d.reshape(-1, 21, 3)
                b_dis3d = torch.norm(labels3d - outputs3d, dim = -1)
                b_dis3d = b_dis3d.cpu().detach().numpy() #* 1000 [B, 21]
                b_dis3d = b_dis3d / np.repeat(scale_label.reshape(-1, 1), 21, axis=-1)

                e_dis3d.append(b_dis3d)

        # calculate auc for this epoch
        e_dis3d = np.r_[e_dis3d] 
        auc_trian = calc_auc(e_dis3d.reshape(-1), 20, 50)

        print('%d epoch training done, train loss=%.5f, auc20-50=%.5f' % (epoch+1, train_loss / (i+1), auc_trian))
        # losses.append((train_loss / (i+1)).cpu().numpy())
        if (epoch+1) % args.snapshot_epoch == 0:
            torch.save(model.state_dict(), args.output_file+str(epoch+1)+'.pkl')
            # np.save(args.output_file+str(epoch+1)+'-losses.npy', np.array(losses))

        # Decay Learning Rate
        scheduler.step()
    print('HourGlass Net pre_training done')

"""# Test"""

if args.test:
    print('Begin testing the network...')
    
    running_loss = 0.0
    l_2d = 0.0
    l_3d = 0.0
    l_hm = 0.0
    e_dis3d = []
    for i, ts_data in enumerate(testloader):
        # get the inputs
        imgs, hm, labels2d, labels3d, scale_label = ts_data
        scale_label = np.array(scale_label)
        
        # wrap them in Variable
        imgs = Variable(imgs)
        hm = Variable(hm)
        labels2d = Variable(labels2d)
        labels3d = Variable(labels3d)
        if use_cuda and torch.cuda.is_available():
            imgs = imgs.float().cuda(device=args.gpu_number[0])
            hm = hm.float().cuda(device=args.gpu_number[0])
            labels2d = labels2d.float().cuda(device=args.gpu_number[0])
            labels3d = labels3d.float().cuda(device=args.gpu_number[0])

        out_hms, outputs2d, outputs3d = model(imgs)

        loss_hm = calc_loss(out_hms, hm)
        loss_hm = loss_hm.mean()
        loss_2d = criterion(outputs2d, labels2d)
        loss_3d = criterion(outputs3d, labels3d)

        loss = (lambda_hm * loss_hm) + (lambda_2d * loss_2d) + (lambda_3d * loss_3d)

        # calculate the distance between label3d and output3d
        b_labels3d = labels3d.reshape(-1, 21, 3) #[B, 21, 3]
        b_out3d = outputs3d.reshape(-1, 21, 3)
        b_dis3d = torch.norm(labels3d - outputs3d, dim = -1)
        b_dis3d = b_dis3d.cpu().detach().numpy() #* 1000 [B, 21]
        b_dis3d = b_dis3d / np.repeat(scale_label.reshape(-1, 1), 21, axis=-1)

        e_dis3d.append(b_dis3d)
        
        running_loss += loss.data
        l_2d += loss_2d.data
        l_3d += loss_3d.data
        l_hm += loss_hm.data

    e_dis3d = np.r_[e_dis3d] 
    print(e_dis3d.shape)
    # e_dis3d = e_dis3d.reshape(-1, 21)
    auc_test = calc_auc(e_dis3d.reshape(-1), 20, 50)
    print('test auc: %f' % (auc_test))
    print('avg test loss: %.5f; l_hm=%.5f, l_2d=%.5f, l_3d=%.5f' % 
        (running_loss / (i+1), l_hm/(i+1), l_2d/(i+1), l_3d/(i+1)))
