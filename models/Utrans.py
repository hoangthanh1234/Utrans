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

from .model_utils import get_grid_size_1d, get_grid_size_2d, padding, unpadding
from .utrans_encoder import Utrans_encoder
from .utrans_kpconv import Utrans_KPConv, KPClassifier
from .utrans_decoder import DecoderUpConv


def create_utrans(model_cfg):   
    model_cfg = model_cfg.copy()

    encoder =Utrans_encoder(
        in_channels=model_cfg['in_channels'], 
        base_channels=model_cfg['base_channels'], 
        window_swin_size=model_cfg['window_swin_size'], 
        shift_size= model_cfg['shift_size'], 
        im_size=model_cfg['im_size']
        )

    decoder = DecoderUpConv(
        dropout_rate=model_cfg['dropout_rate'],
        window_swin_size=model_cfg['window_swin_size'], 
        shift_size= model_cfg['shift_size'], 
        im_size=model_cfg['im_size']
        )

    kpclassifier = KPClassifier(
        in_channels=model_cfg['out_channels'],
        out_channels=model_cfg['out_channels'],
        num_classes=model_cfg['n_cls']
        )

    model = Utrans_KPConv(
        encoder, 
        decoder, 
        kpclassifier, 
        n_cls=model_cfg['n_cls']
        )

    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Utrans_RV(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels, n_cls, im_size, window_swin_size, shift_size, dropout_rate):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.n_cls = n_cls
        self.im_size = im_size       
        self. window_swin_size = window_swin_size
        self.shift_size = shift_size
        self.dropout_rate = dropout_rate 

        net_kwargs = {
            'in_channels': in_channels,
            'base_channels': base_channels,
            'n_cls': n_cls,
            'im_size': im_size,
            'window_swin_size': window_swin_size,
            'shift_size': shift_size,
            'dropout_rate': dropout_rate,
            'out_channels': out_channels
        }

        self.utrans = create_utrans(net_kwargs)

    def counter_model_parameters(self):
        stats = {}
        stats['total_num_parameters'] = count_parameters(self.utrans)
        stats['decoder_num_parameters'] = count_parameters(self.utrans.decoder)
        stats['encoder'] = count_parameters(self.utrans.encoder)
        stats['kpclassifier'] = count_parameters(self.utrans.kpclassifier)
        return stats
    
    def forward(self, *args):
        return self.utrans(*args)
        
if __name__ == '__main__':
    image = torch.rand(4,256,64,512)    
    model = Utrans_RV(in_channel=5, base_channel=32, n_cls=17, im_size=(64,512),window_size=(4,16), dropout_rate=0.1)
    x = count_parameters(model)   
    result = model(image) 
   