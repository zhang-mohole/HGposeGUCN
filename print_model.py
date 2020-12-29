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
from models.graphunet import GraphUHandNet
from models.poseHG import Net_Pose_HG
from torchsummary import summary

model1 = Net_Pose_HG(21)
model1 = model1.cuda()
summary(model1, (3, 256, 256))

gucn = GraphUHandNet(in_features=2, out_features=3)
gucn = gucn.cuda()
summary(gucn, (21, 2))