import torch
from torch import nn, einsum
import numpy as np
from timm.models.swin_transformer import SwinTransformerBlock

if __name__ == '__main__':
    #image = torch.rand(1,256,4,32)   
    #image = torch.rand(1,256,4,32)
    #x = torch.randn(1, 4*32, 96)
    #result = window_partition(image, (4,16))  #partition a image 
    #result2 = window_reverse(result,(4,16),4,32) #reverse partition to image
    #model = SwinTransformerBlock(dim=96, input_resolution=(4, 32))
    #t_1 = SwinTransformerBlock(dim=256, input_resolution=(4, 32))
    #a = t_1(image)
    #print(result.shape)
    #print(result2.shape)
   
    x = torch.randn(1, 56,56, 96)
    print(x)
    print("==========================")
    t_1 = SwinTransformerBlock(dim=96, input_resolution=(56, 56))
    t_2 = SwinTransformerBlock(dim=96, input_resolution=(56, 56), shift_size=3)
    t_1(x).shape, t_2(t_1(x)).shape


class Utrans_Swin(nn.Module):
    def __init__(self, dim, input_resolution, window_size, shift_size):
        super(Utrans_Swin).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        self.block1 = SwinTransformerBlock(dim, input_resolution, window_size)
        self.block2 = SwinTransformerBlock(dim, input_resolution, window_size, shift_size)
    def forward(self, x):
        x = self.block2(self.block1(x))
        return x


if __name__ == '__main__':
    image = torch.rand(4,4,32,256)  #(B, H, W, C)
    window_size=(4,16)
    input_resolution = (4,32)
    shift_size=3 
    dim = 256  
    model = Utrans_Swin(dim, input_resolution, window_size, shift_size)
    result = model(image)
   
    print(result.shape)
  