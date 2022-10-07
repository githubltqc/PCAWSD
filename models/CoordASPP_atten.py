#!/usr/bin/env python
# encoding: utf-8
"""
@project:ENL-FCN-master(复件)
@author:jane
@file: CoordASPP_atten.py
@time: 2021/11/17 下午10:18
@license: (C) Copyright 2020-2025, Node Supply Chain Manager Corporation Limited
@desc: 
@note:
@url:
"""
import torch
import torch.nn as nn

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class ASPPModule(nn.Module):
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
        super(ASPPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center

class SPCS(nn.Module):
    """
    与 CoordAsppAtt2相比去除了hmax和wmax,psp中去除了global均值池化
    """
    def __init__(self, inp, oup, reduction=32):
        super(SPCS, self).__init__()
        self.aspp_sizes = ( 3, 6, 8)
        self.pool_havg = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_wavg = nn.AdaptiveAvgPool2d((1, None))
        self.pool_aspp = ASPPModule(self.aspp_sizes)
        self.mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, self.mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(self.mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(self.mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(self.mip, oup, kernel_size=1, stride=1, padding=0)

        self.conv_aspp2 = nn.Conv2d(self.mip, oup, kernel_size=self.aspp_sizes[0])
        self.conv_aspp3 = nn.Conv2d(self.mip, oup, kernel_size=self.aspp_sizes[1])
        self.conv_aspp4 = nn.Conv2d(self.mip, oup, kernel_size=self.aspp_sizes[2])
        self.fc_bn = nn.BatchNorm2d(oup)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_havg(x)
        x_w = self.pool_wavg(x).permute(0, 1, 3, 2)
        x_aspp = self.pool_aspp(x).unsqueeze(-1)

        y = torch.cat([x_h, x_w, x_aspp], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w, x_aspp = torch.split(y, [h, w, 109], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        x_aspp2, x_aspp3, x_aspp4 = torch.split(x_aspp, [ 9, 36, 64], dim=2)
        x_aspp2 = x_aspp2.reshape(-1, self.mip, self.aspp_sizes[0], self.aspp_sizes[0])
        x_aspp3 = x_aspp3.reshape(-1, self.mip, self.aspp_sizes[1], self.aspp_sizes[1])
        x_aspp4 = x_aspp4.reshape(-1, self.mip, self.aspp_sizes[2], self.aspp_sizes[2])
        a_aspp2 = self.conv_aspp2(x_aspp2)
        a_aspp3 = self.conv_aspp3(x_aspp3)
        a_aspp4 = self.conv_aspp4(x_aspp4)
        a_aspp = ( a_aspp2 + a_aspp3 + a_aspp4)
        a_aspp = a_aspp.sigmoid()
        out1 = identity * a_w * a_h
        out2 = identity * a_aspp
        out = out1+out2
        out = out + identity
        return out

class SPCS_NoSPP(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(SPCS_NoSPP, self).__init__()
        self.pool_havg = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_wavg = nn.AdaptiveAvgPool2d((1, None))
        self.mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, self.mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(self.mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(self.mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(self.mip, oup, kernel_size=1, stride=1, padding=0)

        self.fc_bn = nn.BatchNorm2d(oup)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_havg(x)
        x_w = self.pool_wavg(x) .permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out= identity * a_w * a_h
        out = out + identity
        return out