from __future__ import absolute_import

import numpy as np
import torch


def AUC(dist):
    x = np.array(sorted(list(dist)))
    y = np.array([1. * (i + 1)/ len(x)  for i in range(len(x))])
    # print(x)
    # print(y)
    return (x, y)

def calc_auc(dist, limx = -1, limy = 1e99):
    """ Given x and y values it calculates the approx. integral and normalizes it: area under curve"""
    x, y = AUC(dist)
    # x = x * 1000 #meter to million meter
    # print (x.min(), x.max())
    l, r = 0, 0
    for i in range(len(x)):
        if limx > x[i]:
            l = i + 1
        if limy > x[i]:
            r = i + 1
    # print("l: ",l, x[l], "r: ", r, x[r - 1])
    if x[-1] < limx:
        return 1.0
    if l==r:
        return 0.
    tx = x[l:r]
    ty = y[l:r]    
    integral = np.trapz(ty, tx) + max(0, (limy - x[-1]))
    # norm = np.trapz(np.ones_like(ty), tx)
    norm = 30.
    print('intergral:{}; norm:{}'.format(integral, norm))
    # norm = x[-1] - x[]
    # if limy < 1e98 and limy > x[-1]:
    #     add = limy - x[-1]
    #     integral += add
    #     norm += add
    # print(integral , norm)
    return integral / norm

def combine(x , y):
    assert type(x) == type(y), 'combine two different type items {} and {}'.format(type(x), type(y))
    if isinstance(x, dict):
        assert x.keys() == y.keys()
        return {kx: combine(x[kx],y[kx]) for kx in x}
    if isinstance(x, list):
        assert len(x) == len(y), 'lists size does not match'
        return [combine(a,b) for a, b in zip(x, y)]
    if isinstance(x, torch.Tensor):
        return torch.cat([x,y], 0)
    if isinstance(x, np.ndarray):
        return np.concatenate([x,y], 0)
    raise Exception("Unrecognized type {}".format(type(x)))