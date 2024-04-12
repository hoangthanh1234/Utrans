import torch
import torch.nn as nn
import torch.nn.functional as F
from option import Option
import argparse
import os
from .blocks import Block

import torch.nn.functional as F
import copy
import timm
from timm.models.layers import trunc_normal_

from .model_utils import get_grid_size_1d, get_grid_size_2d

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
        
        self.node_1 = Encoder_node(in_channels,base_channels,)#32
        self.node_2 = Encoder_node(base_channels,base_channels*2)#64
        self.node_3 = Encoder_node(base_channels*2,base_channels*4)#128
        self.node_4 = Encoder_node(base_channels*4,base_channels*8)#256


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
        
        resize_in = [128,288,608,1216] #the inputsize for next block in encoder

        self.resize_channel_1 = nn.Sequential(
            nn.Conv2d(resize_in[0],base_channels, kernel_size=(1, 1), stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(base_channels)
        )

        self.resize_channel_2 = nn.Sequential(
            nn.Conv2d(resize_in[1],base_channels*2, kernel_size=(1, 1), stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(base_channels*2)
        )

        self.resize_channel_3 = nn.Sequential(
            nn.Conv2d(resize_in[2],base_channels*4, kernel_size=(1, 1), stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(base_channels*4)
        )

        self.resize_channel_4 = nn.Sequential(
            nn.Conv2d(resize_in[3],base_channels*8, kernel_size=(1, 1), stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(base_channels*8)
        )      
       

    def get_grid_size(self, H, W):
        return get_grid_size_2d(H, W, self.patch_size, self.patch_stride)

    def forward(self, x):
        B, C, H, W = x.shape  # B, in_channels, image_size[0], image_size[1]   

        shortcut1, node_1 = self.node_1(x,previous=None,Node=1)       
        reChannel_1 = self.resize_channel_1(node_1)         
        
        shortcut2, node_2 = self.node_2(reChannel_1,previous=None,Node =2)
        reChannel_2 = self.resize_channel_2(node_2) 

        shortcut3, node_3 = self.node_3(reChannel_2,previous=reChannel_1,Node=3) 
        reChannel_3 = self.resize_channel_3(node_3)

        shortcut4, node_4 = self.node_4(reChannel_3,previous=reChannel_2,Node=4)
        reChannel_4 = self.resize_channel_4(node_4) 

        x_proj = self.proj_block(reChannel_4)   

        x_latten = x_proj.flatten(2).transpose(1, 2)  # BCHW -> BNC [1,384,8,4] -> [1,32,384]        
              
        
        results = {
            "skip1":shortcut1,
            "skip2":shortcut2,
            "skip3":shortcut3,
            "skip4":shortcut4,
            "en_note1":reChannel_1,
            "en_note2":reChannel_2,
            "en_note3":reChannel_3,
            "en_note4":reChannel_4,            
            "x_proj":x_proj,
            "x_latten":x_latten
         }
       
       
        #print(f"shortcut1 {shortcut1.shape},shortcut2: {shortcut2.shape}, shortcut3: {shortcut3.shape}, shortcut4: {shortcut4.shape}, reChannel_1: {reChannel_1.shape}, reChannel_2: {reChannel_2.shape},reChannel_3: {reChannel_3.shape}, reChannel_4: {reChannel_4.shape}, x_proj: {x_proj.shape}, x_latten: {x_latten.shape}")
        
        return results 

class Encoder_node(nn.Module):

    def __init__(self, in_filters, out_filters,dropout_rate=0.1,stride=1,previous=None):
        super(Encoder_node, self).__init__()   

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)      
       
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), padding=1)        
        self.act2 = nn.LeakyReLU()              
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()       
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3), dilation=2, padding=2)
        self.act4 = nn.LeakyReLU()       
        self.bn4 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act5 = nn.LeakyReLU()       
        self.bn5 = nn.BatchNorm2d(out_filters)

        self.conv6 = nn.Conv2d(out_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act6 = nn.LeakyReLU()
        self.bn6 = nn.BatchNorm2d(out_filters)

       
        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)
                  

    def forward(self, x, previous,Node):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut) 
        shortcut = self.bn1(shortcut)

        shortcut = self.conv2(shortcut)
        shortcut = self.act2(shortcut)
        shortcut = self.bn2(shortcut)
        shortcut_pool = self.pool(shortcut)
        shortcut_pool = self.dropout1(shortcut_pool)

        resA1 = self.conv3(shortcut_pool)
        resA1 = self.act3(resA1)        
        resA1 = self.bn3(resA1) 

        resA2 = self.conv4(resA1)
        resA2 = self.act4(resA2)
        resA2 = self.bn4(resA2)

        resA3 = self.conv5(resA2)
        resA3 = self.act5(resA3)
        resA3 = self.bn5(resA3)

        resA3_plus = shortcut_pool + resA3

        resA3_plus = self.dropout2(resA3_plus)

        resA4 = self.conv6(resA3_plus)
        resA4 = self.act6(resA4)
        resA4 = self.bn6(resA4)       
        

        if  previous != None:        #cat is concatenation
            output = torch.cat((resA1,resA2,resA3,resA4,self.pool(x),self.pool(self.pool(previous))),dim=1)
        else:
            if Node == 2:
                output = torch.cat((resA1, resA2, resA3, resA4, self.pool(x)),dim=1)
            else:
                output = torch.cat((resA1, resA2, resA3, resA4),dim=1)
               
        output = self.dropout3(output)          

        return shortcut_pool, output, 