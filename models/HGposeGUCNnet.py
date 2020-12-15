# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from models.graphunet import GraphUHandNet
from models.poseHG import Net_Pose_HG


class HGposeGUCN(nn.Module):

    def __init__(self, chk_poseHG='', chk_gucn=''):
        super(HGposeGUCN, self).__init__()
        self.poseHG = Net_Pose_HG(21)
        self.gucn = GraphUHandNet(in_features=2, out_features=3)
        """# Load Snapshot"""
        if chk_poseHG != '':
            pre_dict = torch.load(chk_poseHG)
            new_dict = {}
            for k,v in pre_dict.items():
                name = k[7:]
                new_dict[name] = v
            self.poseHG.load_state_dict(new_dict)

        if chk_gucn != '':
            pre_dict = torch.load(chk_gucn)
            new_dict = {}
            for k,v in pre_dict.items():
                name = k[7:]
                new_dict[name] = v
            self.gucn.load_state_dict(new_dict)

    def forward(self, x):
        hms, pose_2d_hm, coord_2d_res = self.poseHG(x)
        pose2d = pose_2d_hm + coord_2d_res
        pose3d = self.gucn(pose2d)
        return hms, pose_2d_hm, coord_2d_res, pose2d, pose3d
