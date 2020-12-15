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
from utils.model import select_model
from utils.options import parse_args_function
from utils.dataset import Dataset
from utils.metric import *
from tqdm import tqdm
from models.hourglass import HeatmapLoss

args = parse_args_function()

"""# Load Dataset"""

root = args.input_file

#mean = np.array([120.46480086, 107.89070987, 103.00262132])
#std = np.array([5.9113948 , 5.22646725, 5.47829601])

transform = transforms.Compose([transforms.Resize((256, 256)),
                                transforms.ToTensor()])

if args.train or args.pre_train:
    trainset = Dataset(root=root, load_set='train', transform=transform, with_object=True)
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

model = select_model(args.model_def)

if use_cuda and torch.cuda.is_available():
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=args.gpu_number)

"""# Load Snapshot"""

if args.pretrained_model != '':
    model.load_state_dict(torch.load(args.pretrained_model))
    losses = np.load(args.pretrained_model[:-4] + '-losses.npy').tolist()
    start = len(losses)
else:
    losses = []
    start = 0

"""# Optimizer"""

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_step_gamma)
scheduler.last_epoch = start
lambda_2d = 0.0003
lambda_res = 0.001
lambda_2dhm = 0.0003
lambda_hm = 0.1

"""# pre train"""
def calc_loss(combined_hm_preds, heatmaps):
    heatmapLoss = HeatmapLoss()
    combined_loss = []
    for i in range(2):
        # combined_loss.append(heatmapLoss(combined_hm_preds[0][:,i], heatmaps))
        combined_loss.append(heatmapLoss(combined_hm_preds[i], heatmaps))
    combined_loss = torch.stack(combined_loss, dim=1)
    return combined_loss 

if args.pre_train:
    print('Begin pre_training the poseHG network...')
    for epoch in range(start, args.num_iterations):  # loop over the dataset multiple times
    
        with tqdm(total=trainloader.__len__(), 
                bar_format='{n_fmt}/{total_fmt} |{bar}| {rate_fmt} {postfix}', ncols=120) as process_bar: 
            train_loss = 0.0

            for i, tr_data in enumerate(trainloader):
                # get the inputs
                inputs, hm, labels2d, labels3d, _ = tr_data
                # wrap them in Variable
                inputs = Variable(inputs)
                labels2d = Variable(labels2d)
                labels3d = Variable(labels3d)
                hm = Variable(hm)
                if use_cuda and torch.cuda.is_available():
                    inputs = inputs.float().cuda(device=args.gpu_number[0])
                    hm = hm.float().cuda(device=args.gpu_number[0])
                    labels2d = labels2d.float().cuda(device=args.gpu_number[0])
                    labels3d = labels3d.float().cuda(device=args.gpu_number[0])
                
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                pred_hms, out_2d_hm, out_2d_res = model(inputs)
                poseHG2d = out_2d_hm + out_2d_res
                
                loss_hm = calc_loss(pred_hms, hm)
                loss_hm = loss_hm.mean() * 1000
                loss_2dhm = criterion(out_2d_hm, labels2d)
                loss_2dres = criterion(out_2d_res, labels2d-out_2d_hm)
                loss2d = criterion(poseHG2d, labels2d)

                # loss = (lambda_hm * loss_hm) + (lambda_2d * loss2d_pose) + (lambda_2d * loss2d_hm)
                # loss = (lambda_2dhm * loss_2dhm) + (lambda_res * loss_2dres) + (lambda_2d * loss2d)
                loss = (lambda_hm * loss_hm) + (lambda_res * loss_2dres)
                train_loss += loss.data
                loss.backward()
                optimizer.step()

                process_bar.set_postfix_str('loss=%.5f, 2d=%.2f, hm=%.5f, 2dhm=%.2f, res=%.2f' % 
                                            (loss.data, loss2d.data, loss_hm.data, loss_2dhm.data, loss_2dres.data))

                process_bar.update()

        print('%d epoch training done, train loss=%.5f' % (epoch+1, train_loss / (i+1)))
        losses.append((train_loss / (i+1)).cpu().numpy())
        if (epoch+1) % args.snapshot_epoch == 0:
            torch.save(model.state_dict(), args.output_file+str(epoch+1)+'.pkl')
            np.save(args.output_file+str(epoch+1)+'-losses.npy', np.array(losses))

        # Decay Learning Rate
        scheduler.step()
    print('PoseHG Net pre_training done')

"""# Test"""

if args.test:
    print('Begin testing the network...')
    
    running_loss = 0.0
    l_2dhm = 0.0
    l_2dres = 0.0
    l_hm = 0.0
    l_2d = 0.0
    e_dis2d = []
    for i, ts_data in enumerate(testloader):
        # get the inputs
        inputs, hm, labels2d, labels3d, scale_label = ts_data
        
        # wrap them in Variable
        inputs = Variable(inputs)
        labels2d = Variable(labels2d)
        hm = Variable(hm)

        if use_cuda and torch.cuda.is_available():
            inputs = inputs.float().cuda(device=args.gpu_number[0])
            labels2d = labels2d.float().cuda(device=args.gpu_number[0])
            hm = hm.float().cuda(device=args.gpu_number[0])

        pred_hms, out_2d_hm, out_2d_res = model(inputs)
        poseHG2d = out_2d_hm + out_2d_res

        loss_hm = calc_loss(pred_hms, hm)
        loss_hm = loss_hm.mean() * 1000
        loss_2dhm = criterion(out_2d_hm, labels2d)
        loss_2dres = criterion(out_2d_res, labels2d-out_2d_hm)
        loss2d = criterion(poseHG2d, labels2d)

        # calculate the distance between label3d and output3d
        b_labels2d = labels2d.reshape(-1, 21, 2) #[B, 21, 2]
        # b_out2d = out_2d_pose.reshape(-1, 21, 2)
        b_dis2d = torch.norm(labels2d - poseHG2d, dim = -1)
        b_dis2d = b_dis2d.cpu().detach().numpy() #* 1000 [B, 21]

        e_dis2d.append(b_dis2d)

        loss = (lambda_2dhm * loss_2dhm) + (lambda_res * loss_2dres) + (lambda_2d * loss2d)
        running_loss += loss.data
        l_2dhm += loss_2dhm.mean().data
        l_2dres += loss_2dres.mean().data
        l_hm += loss_hm.data
        l_2d += loss2d.mean().data
    e_dis2d = np.r_[e_dis2d] 
    print(e_dis2d.shape, 'samples in total, avg 2d distance: ', np.mean(e_dis2d))
    
    # print('test loss: %.5f, 2d= l_hm=%.5f, 2dhm=%.5f, res=%.5f' % 
    #         (running_loss / (i+1), l_2d / (i+1), l_hm / (i+1), l_2dhm / (i+1), l_2dres / (i+1)))
    print('test heatmap loss: %.5f' % (l_hm / (i+1)))
