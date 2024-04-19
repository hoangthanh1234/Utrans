import torch
import torch.nn as nn
import torch.nn.functional as F
from option import Option
import argparse
import os
import sys  

import torch.nn.functional as F
import copy
import timm
from timm.models.layers import trunc_normal_
from swin_transformer import Small_swin
from model_utils import ConvBNReLuK1, ConvBNReLuK3, ConvBNReLuK3D2, ConvBNReLuK7, ConvBNReLuK7D2

class Encoder_node(nn.Module):

    def __init__(self, in_filters, out_filters,concat_channel,dropout_rate=0.1,stride=1,previous=None):
        super(Encoder_node, self).__init__()  

        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)  

        self.EncodeB1 = ConvBNReLuK1(in_filters,out_filters)
        self.EncodeB2 = ConvBNReLuK3(out_filters,out_filters)

        self.EncodeB3 = ConvBNReLuK3(out_filters, out_filters)  
        self.EncodeB4 = ConvBNReLuK3(out_filters, out_filters)  
        self.EncodeB5 = ConvBNReLuK3D2(out_filters, out_filters)  

        self.EncodeB6 = ConvBNReLuK3(out_filters, out_filters)  
        self.EncodeB7 = ConvBNReLuK3(out_filters, out_filters)  
        self.EncodeB8 = ConvBNReLuK3D2(out_filters, out_filters)

        self.EncodeB9 = ConvBNReLuK1(concat_channel,out_filters)  
       
               
        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)
                  
    def forward(self, x, previous,Node):
        x1 = self.EncodeB1(x)   
        x2 = self.EncodeB2(x1)
        x3 = self.EncodeB3(x2)
        x4 = self.EncodeB4(x3)
        x5 = self.EncodeB5(x4)

        x_plus = x2+x3+x4+x5+x5      
        
        x6 = self.EncodeB6(x_plus)
        x7 = self.EncodeB7(x6)
        x8 = self.EncodeB8(x7)   

        if  previous != None:        #cat is concatenation
            output = torch.cat((x8,x1,x,self.pool(previous)),dim=1)
        else:
            if Node == 2:                
                output = torch.cat((x8, x1,x),dim=1)                
            else:
                output = torch.cat((x8, x1),dim=1)      

        output = self.dropout3(self.EncodeB9(self.pool(output)))          

        return x_plus, output,

class Utrans_encoder(nn.Module):
    def __init__(self,in_channels=None,base_channels=None):
        super(Utrans_encoder,self).__init__()
        concat_channels = [64, 160, 352, 704]
        self.node_1 = Encoder_node(in_channels, base_channels,concat_channels[0])  # 32        

        self.node_2 = Encoder_node(base_channels, base_channels * 2,concat_channels[1])  # 64       

        self.node_3 = Encoder_node(base_channels * 2, base_channels * 4, concat_channels[2])  # 128       

        self.node_4 = Encoder_node(base_channels * 4, base_channels * 8,concat_channels[3])  # 256             

        self.bridge = Small_swin(in_channels=256, hidden_dimension=128, window_size=4)

    def forward(self, x):
        B, C, H, W = x.shape  # B, in_channels, image_size[0], image_size[1]
        x_plus1, node_1 = self.node_1(x, previous=None, Node=1)
        x_plus2, node_2 = self.node_2(node_1, previous=None, Node=2)
        x_plus3, node_3 = self.node_3(node_2, previous=node_1, Node=3)
        x_plus4, node_4 = self.node_4(node_3, previous=node_2, Node=4)
       
        print("Node4", node_4.shape)
        bridge = self.bridge(node_4)

        results = {
            "x_plus1": x_plus1,
            "x_plus2": x_plus2,
            "x_plus3": x_plus3,
            "x_plus4": x_plus4,
            "en_note1": node_1,
            "en_note2": node_2,
            "en_note3": node_3,
            "en_note4": node_4,
            "bridge": bridge
        }

        #print(f"x_plus1 {x_plus1.shape},x_plus2: {x_plus2.shape}, x_plus3: {x_plus3.shape}, x_plus4: {x_plus4.shape}, node_1: {node_1.shape}, node_2: {node_2.shape},node_3: {node_3.shape}, node_4: {node_4.shape}")
        return bridge

if __name__ == '__main__':
    print("In module products sys.path[0], __package__ ==", sys.path[0], __package__)
    image = torch.rand(4,5,64,512)
    model1 = Utrans_encoder(5,32)
    print(model1)
    result = model1(image)
    print(result.shape)