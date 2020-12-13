# -*- coding: utf-8 -*-
from models.graphunet import GraphUNet, GraphNet, GraphUHandNet
from models.resnet import resnet10, resnet18, resnet50, resnet101
from models.hopenet import HopeNet
from models.hourglass import Net_HM_HG
from models.poseHG import Net_Pose_HG

def select_model(model_def):
    if model_def.lower() == 'hopenet':
        model = HopeNet()
        print('HopeNet is loaded')
    elif model_def.lower() == 'hourglass':
        model = Net_HM_HG(21)
        print('HourGlass Net is loaded')
    elif model_def.lower() == 'posehg':
        model = Net_Pose_HG(21)
        print('PoseHG Net is loaded')
    elif model_def.lower() == 'graphuhand':
        model = GraphUHandNet()
        print('GraphUHand Net is loaded')
    elif model_def.lower() == 'resnet10':
        # model = resnet10(pretrained=False, num_classes=29*2)
        model = resnet10(pretrained=False, num_classes=21*2)
        print('ResNet10 is loaded')
    elif model_def.lower() == 'resnet18':
        # model = resnet18(pretrained=False, num_classes=29*2)
        model = resnet18(pretrained=False, num_classes=21*2)
        print('ResNet18 is loaded')
    elif model_def.lower() == 'resnet50':
        # model = resnet50(pretrained=False, num_classes=29*2)
        model = resnet50(pretrained=False, num_classes=21*2)
        print('ResNet50 is loaded')
    elif model_def.lower() == 'resnet101':
        # model = resnet101(pretrained=False, num_classes=29*2)
        model = resnet101(pretrained=False, num_classes=21*2)
        print('ResNet101 is loaded')
    elif model_def.lower() == 'graphunet':
        model = GraphUNet(in_features=2, out_features=3)
        print('GraphUNet is loaded')
    elif model_def.lower() == 'graphnet':
        model = GraphNet(in_features=2, out_features=3)
        print('GraphNet is loaded')
    else:
        raise NameError('Undefined model')
    return model
