#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 09:04:18 2020

@author: carpc
"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.cluster import KMeans

def get_inplanes():
    return [64, 128, 256, 512]


def K_means_clustering(x,k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(x)
    return kmeans.labels_

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
     


        out += residual
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=2,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 num_labeled_classes=4,
                 num_unlabeled_classes=3):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 3, 3),
                               stride=(conv1_t_stride, 1, 1),
                               padding=(conv1_t_size // 1, 1, 1),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            elif shortcut_type == 'B':
                #downsample = nn.AvgPool3d(2)
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    )

                    
                #nn.BatchNorm3d(planes * block.expansion)

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes,downsample=nn.Sequential()))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x


def generate_model(model_depth, n_labelled_classes=6, n_unlabelled_classes=6, **kwargs):
    
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), num_labeled_classes=n_labelled_classes,
                       num_unlabeled_classes=n_unlabelled_classes, **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), num_labeled_classes=n_labelled_classes,
                       num_unlabeled_classes=n_unlabelled_classes, **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), num_labeled_classes=n_labelled_classes,
                       num_unlabeled_classes=n_unlabelled_classes, **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), num_labeled_classes=n_labelled_classes, 
                       num_unlabeled_classes=n_unlabelled_classes, **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), num_labeled_classes=n_labelled_classes, 
                       num_unlabeled_classes=n_unlabelled_classes, **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), num_labeled_classes=n_labelled_classes, 
                       num_unlabeled_classes=n_unlabelled_classes, **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), num_labeled_classes=n_labelled_classes,
                       num_unlabeled_classes=n_unlabelled_classes,**kwargs)

    return model


class Decoder(nn.Module):
    def __init__(self, in_channels,image_size=80)->None:
        super(Decoder, self).__init__()

        self.conv1 = nn.ConvTranspose3d(in_channels, 20, kernel_size=(2,2,2), stride = (1,2,2))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose3d(20, 40, kernel_size=(2,2,2), stride = (1,2,2))
        self.conv3 = nn.ConvTranspose3d(40, image_size, kernel_size=(2,4,4), stride = (1,2,2))
        self.conv4 = nn.ConvTranspose3d(image_size, image_size, kernel_size=(1,5,5), stride = (1,2,2))
        self.conv5 = nn.ConvTranspose3d(image_size, 1, kernel_size=(1,6,6), stride = (1,3,3))

        self.fn = nn.Sigmoid()
        
    def forward(self,z):
        x = z.view((z.size(0),int(z.size(1)/4),1,2,2))
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)

        x = self.fn(x)
        return x
    
class AE(nn.Module):
    def __init__(self,num_labelled,num_unlabelled,image_size,model_depth)->None:
        super(AE,self).__init__()
        self.encoder = generate_model(model_depth=model_depth)
        if model_depth < 50:
            latent_dim = 512
        else:
            latent_dim = 2048
        self.decoder =  Decoder(in_channels=int(latent_dim/4),image_size=image_size)
        self.num_unlabelled = num_unlabelled

        
    def forward(self,x):
        z = self.encoder(x)
        xhat = self.decoder(z)

        with torch.no_grad():
            labels = K_means_clustering(z.view(z.size(0),-1).cpu().numpy(), self.num_unlabelled)

        return labels,z.view(z.size(0),-1),xhat
         


if __name__ == '__main__':
    from torch.nn.parameter import Parameter
    device = torch.device('cuda')
    encoder = generate_model(model_depth=18)
    input_tensor = Variable(torch.randn(2,1,80,80,4))
    x = encoder(input_tensor) #b * f * w * h * d
    print(x.size())
    decoder = Decoder(in_channels=512,image_size=80)
    recon = decoder(x)
    print(recon.size())
    loss = F.mse_loss(input_tensor, recon)
    
    ae = AE(num_labelled=6,num_unlabelled=6,image_size=80,model_depth=18)
    _,_,_ = ae(input_tensor)
    
    
    

    