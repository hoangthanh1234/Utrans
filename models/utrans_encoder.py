import torch
import torch.nn as nn
import torch.nn.functional as F
from option import Option
import argparse
import os
from blocks import Block

import torch.nn.functional as F
import copy
import timm
from timm.models.layers import trunc_normal_

from model_utils import get_grid_size_1d, get_grid_size_2d

class Utrans_encoder(nn.Module):
    def __init__(self,
                 in_channels=5,
                 base_channels=32,
                 img_size=(16, 32),
                 patch_stride=(2, 8),
                 embed_dim=384,
                 flatten=True,
                 hidden_dim=None):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 8 * base_channels

        self.base_channels = base_channels
        self.dropout_ratio = 0.2

        # Build encoder         

        self.node_1 = ResContextBlock(in_channels,base_channels)
        self.node_2 = ResContextBlock(base_channels,base_channels*2)
        self.node_3 = ResContextBlock(base_channels*2,base_channels*4)
        self.node_4 = ResContextBlock(base_channels*4,base_channels*8)


        assert patch_stride[0] % 2 == 0
        assert patch_stride[1] % 2 == 0
        kernel_size = (patch_stride[0] + 1, patch_stride[1] + 1) #(3,9)
        
        padding = (patch_stride[0] // 2, patch_stride[1] // 2) #(1,4)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        

        self.proj_block = nn.Sequential(
             nn.AvgPool2d(kernel_size=kernel_size, stride=patch_stride, padding=padding),
             nn.Conv2d(hidden_dim, embed_dim, kernel_size=1))

        self.patch_stride = patch_stride
        self.patch_size = patch_stride
        self.grid_size = (img_size[0] // patch_stride[0], img_size[1] // patch_stride[1])       
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

    def get_grid_size(self, H, W):
        return get_grid_size_2d(H, W, self.patch_size, self.patch_stride)

    def forward(self, x):
        B, C, H, W = x.shape  # B, in_channels, image_size[0], image_size[1]        

        shortcut1, node_1, node_1_pool = self.node_1(x) #[1,32, 128, 256]
        re_pool = self.pool(node_1_pool)    
        shortcut2, node_2, node_2_pool = self.node_2(re_pool) #[1, 64, 64, 128]
        shortcut3, node_3,node_3_pool = self.node_3(node_2_pool) #[1, 128, 32, 64]
        shortcut4, node_4, node_4_pool = self.node_4(node_3_pool) #[1, 256, 16, 32]
        x_proj = self.proj_block(node_4_pool)
        x_latten = x_proj.flatten(2).transpose(1, 2)  # BCHW -> BNC [1,384,8,4] -> [1,32,384]        
        
       
        results = {
            "shortcut1":shortcut1,
            "shortcut2":shortcut2,
            "shortcut3":shortcut3,
            "shortcut4":shortcut4,
            "node1":node_1,
            "node2":node_2,
            "node3":node_3,
            "node4":node_4,
            "node_1_pool":node_1_pool,
            "node_2_pool":node_2_pool,
            "node_3_pool":node_3_pool,
            "node_4_pool":node_4_pool,
            "x_proj":x_proj,
            "x_latten":x_latten
         }
        
        return results 

class ResContextBlock(nn.Module):

    def __init__(self, in_filters, out_filters,dropout_rate=0.1,stride=1):
        super(ResContextBlock, self).__init__()
       
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3), padding=1)        
        self.act2 = nn.LeakyReLU()              
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()       
        self.bn2 = nn.BatchNorm2d(out_filters)
       
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x,n_1_node=None,n_2_node=None):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut) #[1,32,512,1024]
      
        resA = self.conv2(shortcut)
        resA = self.act2(resA)        
        resA = self.bn1(resA)

        resA1 = self.conv3(resA)
        resA1 = self.act3(resA1)
        resA1 = self.bn2(resA1) #[1,32,512,1024]

        output  = resA1 + shortcut #[1,32,512,1024]                
        output_pool = self.dropout(self.pool(output)) #[1, 32, 128, 256]  

        return shortcut, output, output_pool