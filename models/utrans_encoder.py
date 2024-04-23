import torch
import torch.nn as nn
import torch.nn.functional as F
from option import Option
import argparse
import os
import sys  

import torch.nn.functional as F
import copy

from .swin_transformer import Swin_SCA
from .model_utils import ConvBNReLuK1, ConvBNReLuK3, ConvBNReLuK3D2, ConvBNReLuK7, ConvBNReLuK7D2
from .MRCIAM import MRCIAM
from .SCA import sa_layer

class Encoder_node(nn.Module):

    def __init__(self, in_filters, out_filters,concat_channel,dropout_rate=0.2,stride=1,previous=None):
        super(Encoder_node, self).__init__()  

        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)  
        self.EncodeB1 = ConvBNReLuK1(in_filters,out_filters)
        self.EncodeB2 = ConvBNReLuK3(out_filters,out_filters)
        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.EncodeB3 = ConvBNReLuK3(out_filters, out_filters)  
        self.EncodeB4 = ConvBNReLuK3(out_filters, out_filters)  
        self.EncodeB5 = ConvBNReLuK3D2(out_filters, out_filters)  
        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.EncodeB6 = ConvBNReLuK3(out_filters, out_filters)  
        self.EncodeB7 = ConvBNReLuK3(out_filters, out_filters)  
        self.EncodeB8 = ConvBNReLuK3D2(out_filters, out_filters)

        self.EncodeB9 = ConvBNReLuK1(concat_channel,out_filters)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)  
         
                  
    def forward(self, x, previous,Node):
        x1 = self.EncodeB1(x)   
        x2 = self.EncodeB2(x1)        

        x3 = self.EncodeB3(x2)
        x4 = self.EncodeB4(x3)
        x5 = self.EncodeB5(x4)
        x5 = self.dropout1(x5)

        x_plus = x2+x3+x4+x5+x5      
        
        x6 = self.EncodeB6(x_plus)
        x7 = self.EncodeB7(x6)
        x8 = self.EncodeB8(x7) 
        x8 = self.dropout2(x8)  

        if  previous != None:       #cat is concatenation
            output = torch.cat((x8,x1,x,self.pool(previous)),dim=1)
        else:
            if Node == 2:                
                output = torch.cat((x8, x1,x),dim=1)                
            else:
                output = torch.cat((x8, x1),dim=1)      

        output = self.dropout3(self.EncodeB9(output))          

        return x_plus, self.pool(output),

class Utrans_encoder(nn.Module):
    def __init__(self,in_channels=None,base_channels=None, window_swin_size=None, shift_size=None, im_size=None):
        super().__init__()

        self.im_size = im_size
        H, W = self.im_size[0], self.im_size[1]
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.window_swin_size = window_swin_size
        self.shift_size = shift_size

        #process input via MRCIAM blocks
        self.MRCIAM_xyz = MRCIAM(3, base_channels)
        self.MRCIAM_1d = MRCIAM(1, base_channels)
        self.MRCIAM_1r = MRCIAM(1, base_channels)
        self.sca = sa_layer(base_channels*3)
        self. reduce_channel = ConvBNReLuK1(base_channels*3, base_channels)        

        #Main block of model
        concat_channels = [64, 160, 352, 704]
        self.node_1 = Encoder_node(base_channels, base_channels,concat_channels[0])  # 32        

        self.node_2 = Encoder_node(base_channels, base_channels * 2,concat_channels[1])  # 64       

        self.node_3 = Encoder_node(base_channels * 2, base_channels * 4, concat_channels[2])  # 128       

        self.node_4 = Encoder_node(base_channels * 4, base_channels * 8,concat_channels[3])  # 256             

        self.bridge = Swin_SCA(dim=256, input_resolution=(H//16,W//16), window_size=self.window_swin_size, shift_size=self.shift_size)

    def forward(self, x):
        B, C, H, W = x.shape  # B, in_channels, image_size[0], image_size[1]
        
        x1 = x[:, 0:3,:,:] #xyz
        x2 = x[:, 3:4,:,:] #i_intensive
        x3 = x[:, 4:5,:,:] #d_depth
        
        MRCIAM_xyz = self.MRCIAM_xyz(x1)
        MRCIAM_i = self.MRCIAM_1r(x2)
        MRCIAM_r = self.MRCIAM_1d(x3)
        MRCIAM_plus = torch.cat((MRCIAM_xyz, MRCIAM_i, MRCIAM_r), dim=1)
        SCA = self.sca(MRCIAM_plus) #Change position between SCA and reduce changel compare to TranRV
        reduce_channel = self.reduce_channel(SCA)


        x_plus1, node_1 = self.node_1(reduce_channel, previous=None, Node=1)
        x_plus2, node_2 = self.node_2(node_1, previous=None, Node=2)
        x_plus3, node_3 = self.node_3(node_2, previous=node_1, Node=3)
        x_plus4, node_4 = self.node_4(node_3, previous=node_2, Node=4)       
        
        bridge = self.bridge(node_4)

        results = {
            "x_plus1": x_plus1,
            "x_plus2": x_plus2,
            "x_plus3": x_plus3,
            "x_plus4": x_plus4,
            "en_note1": node_1,
            "en_note2": node_2,
            "en_note3": node_3,
            "en_note4": node_4            
        }

        #print(f"x_plus1 {x_plus1.shape},x_plus2: {x_plus2.shape}, x_plus3: {x_plus3.shape}, x_plus4: {x_plus4.shape}, node_1: {node_1.shape}, node_2: {node_2.shape},node_3: {node_3.shape}, node_4: {node_4.shape}")
        return bridge, results

if __name__ == '__main__':
   
    image = torch.rand(4,5,64,512)
    model1 = Utrans_encoder(5,32)
    
    result = model1(image)
    print(result.shape)
    