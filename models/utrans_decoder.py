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
from .model_utils import ConvBNReLuK1, ConvBNReLuK3, ConvBNReLuK3D2, ConvBNReLuK7, ConvBNReLuK7D2
from .swin_transformer import Swin_SCA
from .SCA import sa_layer

class DecoderUpConv(nn.Module):
    def __init__(
        self,      
        dropout_rate,
        window_swin_size,
        shift_size,
        im_size
        ):
        super().__init__()

        self.window_swin_size = window_swin_size
        self.shift_size =shift_size
        self.im_size = im_size
        self.dropout_rate = dropout_rate

        H, W = im_size[0], im_size[1]       

        self.node1 = Decoder_node(in_channel=256, out_channel=128)
        self.rechanel_Node1 = ConvBNReLuK1(256,128)
        self.SCA_Node1 = Swin_SCA(dim=128, input_resolution=(H//8, W//8), window_size=self.window_swin_size, shift_size=self.shift_size)
       
        self.node2 = Decoder_node(in_channel=128, out_channel=256)
        self.rechanel_Node2 = ConvBNReLuK1(704,256)
        self.SCA_Node2 = Swin_SCA(dim=256, input_resolution=(H//4, W//4), window_size=self.window_swin_size, shift_size=self.shift_size)
       
        self.node3 = Decoder_node(in_channel=256, out_channel=128)
        self.rechanel_Node3 = ConvBNReLuK1(464,128)
        self.SCA_Node3 = Swin_SCA(dim=128, input_resolution=(H//2, W//2), window_size=self.window_swin_size, shift_size=self.shift_size)
       
        self.node4 = Decoder_node(in_channel=128, out_channel=256)
        self.rechanel_Node4 = ConvBNReLuK1(624,256)
        self.SCA_Node4 = Swin_SCA(dim=256, input_resolution=(H, W), window_size=self.window_swin_size, shift_size=self.shift_size)


    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size, return_features=False, encoder_infor=None):
        H, W = im_size
       
        skip1 = encoder_infor['x_plus1']
        skip2 = encoder_infor['x_plus2']
        skip3 = encoder_infor['x_plus3']
        skip4 = encoder_infor['x_plus3']

        en_block1 = encoder_infor['en_note1']
        en_block2 = encoder_infor['en_note2']
        en_block3 = encoder_infor['en_note3']
        en_block4 = encoder_infor['en_note4']       
        
        #implement Encoder Blocks
        note1 = self.node1(x, previous = None, Node =1, encode_1=None, encode_2=None)   
        resize_channel_1 = self.rechanel_Node1(note1)           
        resize_channel_1 = self.SCA_Node1(resize_channel_1)        
          
        
        note2 = self.node2(resize_channel_1, previous = None, Node =2, encode_1=skip3, encode_2=en_block3)                  
        resize_channel_2 = self.rechanel_Node2(note2)         
        resize_channel_2 = self.SCA_Node2(resize_channel_2)       
       
        note3 = self.node3(resize_channel_2, previous = note1, Node =3, encode_1=skip2, encode_2=en_block2)
        resize_channel_3 = self.rechanel_Node3(note3)        
        resize_channel_3 = self.SCA_Node3(resize_channel_3)      

        note4 = self.node4(resize_channel_3, previous = resize_channel_2, Node =4, encode_1=skip1, encode_2=en_block1)
        resize_channel_4 = self.rechanel_Node4(note4)        
        resize_channel_4 = self.SCA_Node4(resize_channel_4)        

        return resize_channel_4 
        

class PixelShuffleUp(nn.Module):
  """
  A module for upsampling using PixelShuffle.
  """
  def __init__(self, upscale_factor):
    super(PixelShuffleUp, self).__init__()
    if upscale_factor % 2 != 0:
      raise ValueError("Upscale factor must be a multiple of 2 for PixelShuffle.")
    self.upscale_factor = upscale_factor

  def forward(self, x):
    """
    Performs upsampling on the input tensor using PixelShuffle.

    Args:
      x: Input tensor (torch.Tensor) of shape (batch_size, channels, height, width).

    Returns:
      Upsampled tensor (torch.Tensor) with the spatial resolution increased by the upscale factor.
    """
    batch_size, channels, in_height, in_width = x.shape
    out_channels = channels // (self.upscale_factor * self.upscale_factor)
    out_height = in_height * self.upscale_factor
    out_width = in_width * self.upscale_factor
    return x.view(batch_size, out_channels, self.upscale_factor, self.upscale_factor, in_height, in_width) \
           .permute(0, 1, 3, 5, 2, 4).contiguous().view(batch_size, out_channels, out_height, out_width)


class Decoder_node(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate=0.2,stride=1,previous=None):
        super(Decoder_node, self).__init__()        
        
        self.DecoderB1 = ConvBNReLuK1(in_channel//4, out_channel)       
        self.DecoderB2 = ConvBNReLuK3(out_channel, out_channel)

        self.DecoderB3 = ConvBNReLuK3(out_channel, out_channel)
        self.DecoderB4 = ConvBNReLuK3(out_channel, out_channel)
        self.DecoderB5 = ConvBNReLuK3D2(out_channel, out_channel)
        self.dropout1 = nn.Dropout2d(p=dropout_rate) 

        self.DecoderB6 = ConvBNReLuK3(out_channel, out_channel)
        self.DecoderB7 = ConvBNReLuK3(out_channel, out_channel)
        self.DecoderB8 = ConvBNReLuK3D2(out_channel, out_channel)
        self.dropout2 = nn.Dropout2d(p=dropout_rate) 
        
        self.dropout3 = nn.Dropout2d(p=dropout_rate)  

        self.upsample_2 = PixelShuffleUp(2)
        self.upsample_4 = PixelShuffleUp(4)

    def forward(self, x, previous, Node, encode_1, encode_2):  
        
        upsample = self.upsample_2(x)       
        x1 = self.DecoderB1(upsample)          
        x2 = self.DecoderB2(x1)

        x3 = self.DecoderB3(x2)
        x4 = self.DecoderB4(x3)
        x5 = self.DecoderB5(x4)
        x5 = self.dropout1(x5)

        x_plus = x2+x3+x4+x5+x5      
        
        x6 = self.DecoderB6(x_plus)
        x7 = self.DecoderB7(x6)
        x8 = self.DecoderB8(x7) 
        x8 = self.dropout2(x8)
        
                
        if Node == 1:
            output = torch.cat((x8, x1), dim=1)            
        elif Node == 2:             
            output = torch.cat((x8, x1, upsample,encode_1, self.upsample_2(encode_2)), dim=1)                       
        else:            
            previous_up = self.upsample_4(previous)      
            output = torch.cat((x8, x1, upsample, previous_up, encode_1, encode_1),dim=1)

        return self.dropout3(output)


if __name__ == '__main__':
    image = torch.rand(4,256,4,32)    
    model = DecoderUpConv()
    result = model(image,(16,32))
   