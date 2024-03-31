# Copyright 2023 - Valeo Comfort and Driving Assistance
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from .model_utils import get_grid_size_1d, get_grid_size_2d, init_weights


class UpConvBlock(nn.Module):
    def __init__(
        self,
        in_filters,
        out_filters,
        dropout_rate,
        scale_factor=(2, 8),
        drop_out=False,
        skip_filters=0):
        super(UpConvBlock, self).__init__()

        self.in_filters = in_filters
        self.out_filters = out_filters
        self.skip_filters = skip_filters

        # scale_factor has to be a tuple or a list with two elements
        if isinstance(scale_factor, int):
            scale_factor = (scale_factor, scale_factor)
        assert isinstance(scale_factor, (list, tuple))
        assert len(scale_factor) == 2
        self.scale_factor = scale_factor

        if self.scale_factor[0] != self.scale_factor[1]:            
            upsample_layers = [
                nn.Conv2d(in_filters, out_filters * self.scale_factor[0] * self.scale_factor[1], kernel_size=(1, 1)),
                Rearrange('b (c s0 s1) h w -> b c (h s0) (w s1)', s0=self.scale_factor[0], s1=self.scale_factor[1]),]
        else:           
            upsample_layers = [
                nn.Conv2d(in_filters, out_filters * self.scale_factor[0] * self.scale_factor[1], kernel_size=(1, 1)),
                nn.PixelShuffle(self.scale_factor[0]),]

        if drop_out:
            upsample_layers.append(nn.Dropout2d(p=dropout_rate))
        self.conv_upsample = nn.Sequential(*upsample_layers)

        self.conv1 = nn.Sequential(
            nn.Conv2d(out_filters + skip_filters, out_filters, (3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_filters)
        )

        num_filters = out_filters
        output_layers = [
            nn.Conv2d(num_filters, out_filters, kernel_size=(1, 1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_filters),
        ]
        if drop_out:
            output_layers.append(nn.Dropout2d(p=dropout_rate))
            
        self.conv_output = nn.Sequential(*output_layers)

    def forward(self, x, skip=None):
        x_up = self.conv_upsample(x) # increase spatial size by a scale factor. B, 2*base_channels, image_size[0], image_size[1]       
        if self.skip_filters > 0:
            assert skip is not None
            assert skip.shape[1] == self.skip_filters
            x_up = torch.cat((x_up, skip), dim=1)

        x_up_out = self.conv_output(self.conv1(x_up))        
        return x_up_out


class DecoderUpConv(nn.Module):
    def __init__(
        self,
        n_cls,
        patch_size,
        d_encoder,
        d_decoder,
        scale_factor=(2, 8),
        patch_stride=None,
        dropout_rate=0.2,
        drop_out=False,
        skip_filters=0):
        super().__init__()

        self.d_encoder = d_encoder
        self.d_decoder = d_decoder
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.n_cls = n_cls

        self.up_conv_block = UpConvBlock(
            d_encoder, d_decoder,
            dropout_rate=dropout_rate,
            scale_factor=scale_factor,
            drop_out=drop_out,
            skip_filters=skip_filters)

        self.head = nn.Conv2d(d_decoder, n_cls, kernel_size=(1, 1))
        self.apply(init_weights)

        self.block_1 = Decoder_block(256,256)
        self.block_2 = Decoder_block(256,128)
        self.block_3 = Decoder_block(128,128)
        self.block_4 = Decoder_block(128,256)

        resize_in = [512,768,768,832] # the output from block1 to block 4. then resize the change to reduce the weigh of model
        self.resize_channel_1 = nn.Conv2d(resize_in[0],256, kernel_size=(1, 1), stride=1)
        self.resize_channel_2 = nn.Conv2d(resize_in[1],128, kernel_size=(1, 1), stride=1)
        self.resize_channel_3 = nn.Conv2d(resize_in[2],128, kernel_size=(1, 1), stride=1)
        self.resize_channel_4 = nn.Conv2d(resize_in[3],256, kernel_size=(1, 1), stride=1)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size, skip=None, return_features=False, encoder_infor=None):
        H, W = im_size
        GS_H, GS_W = get_grid_size_2d(16, 48, self.patch_size, self.patch_stride) # importance to resize;       
        x = rearrange(x, 'b (h w) c -> b c h w', h=GS_H) # B, d_model, image_size[0]/patch_stride[0], image_size[1]/patch_stride[1]
        #Up conv to C=256
        x = self.up_conv_block(x, skip)
        
        #get the information from encoder 
        skip1 = encoder_infor['skip1']
        skip2 = encoder_infor['skip2']
        skip3 = encoder_infor['skip3']
        skip4 = encoder_infor['skip4']

        en_block1 = encoder_infor['en_block1']
        en_block2 = encoder_infor['en_block2']
        en_block3 = encoder_infor['en_block3']
        en_block4 = encoder_infor['en_block4']
        
        #implement Encoder Blocks
        block1 = self.block_1(x, previous = None, Node =1, encode_1=None, encode_2=None)
        resize_channel_1 = self.resize_channel_1(block1)     

        block2 = self.block_2(resize_channel_1, previous = None, Node =2, encode_1=skip3, encode_2=en_block3)
        resize_channel_2 = self.resize_channel_2(block2)
       
        block3 = self.block_3(resize_channel_2, previous = resize_channel_1, Node =3, encode_1=skip2, encode_2=en_block2)
        resize_channel_3 = self.resize_channel_3(block3)

        block4 = self.block_4(resize_channel_3, previous = resize_channel_2, Node =4, encode_1=skip1, encode_2=en_block1)
        resize_channel_4 = self.resize_channel_4(block4)

        
        if return_features:            
            return resize_channel_4 #return feature this case
        else:           
            return self.head(resize_channel_4)



class Decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1,stride=1,previous=None):
        super(Decoder_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1)        
        self.act2 = nn.LeakyReLU()              
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()       
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(p=dropout_rate)        

    def forward(self, x, previous, Node, encode_1, encode_2):       
       
        if Node != 1:
            upsample = F.interpolate(x, scale_factor=2, mode='bilinear')
            
        else:
            upsample = x
        
        shortcut = self.conv1(upsample)
        shortcut = self.act1(shortcut)    

        resA = self.conv2(shortcut)
        resA = self.act2(resA)
        resA = self.bn1(resA)
       
        resA1 = self.conv3(resA)
        resA1 = self.act3(resA1)
        resA1 = self.bn2(resA1)
                
        if Node == 1:
            output = torch.cat((shortcut,resA1), dim=1)            
        elif Node == 2:                 
            output = torch.cat((shortcut,resA1,upsample,encode_1, encode_2), dim=1)                       
        else:
            previous_up = F.interpolate(previous, scale_factor=4, mode='bilinear')       
            output = torch.cat((shortcut,resA1,upsample,previous_up, encode_1, encode_1),dim=1)
       
        return output
       

