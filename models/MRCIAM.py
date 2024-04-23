import torch
from torch import nn, einsum
from .model_utils import ConvBNReLuK1, ConvBNReLuK3, ConvBNReLuK3D2, ConvBNReLuK7, ConvBNReLuK7D2, ConvBNReLuK5
from .SCA import sa_layer

class MRCIAM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.layer1_1 = ConvBNReLuK3(in_channel, out_channel)
        self.layer1_2 = ConvBNReLuK5(in_channel, out_channel)
        self.layer1_3 = ConvBNReLuK7(in_channel, out_channel)
        self.layer1_4 = ConvBNReLuK1(in_channel, out_channel)
        self.layer1_5 = ConvBNReLuK3(out_channel*3, out_channel)

        self.layer2_1 = ConvBNReLuK3(out_channel, out_channel)
        self.layer2_2 = ConvBNReLuK3D2(out_channel, out_channel)
        self.layer2_3 = ConvBNReLuK1(out_channel, out_channel)

        self.layer3_1 = ConvBNReLuK3(out_channel, out_channel)
        self.layer3_2 = ConvBNReLuK5(out_channel, out_channel)
        self.layer3_3 = ConvBNReLuK7(out_channel, out_channel)
        self.layer3_4 = ConvBNReLuK1(out_channel, out_channel)
        self.layer3_5 = ConvBNReLuK3(out_channel*3, out_channel)
        self.sca = sa_layer(out_channel)
    def forward(self, x):
        shortcut = x
        L1_x1 = self.layer1_1(x)
        L1_x2 = self.layer1_2(x)
        L1_x3 = self.layer1_3(x)
        L1_x4 = self.layer1_4(x)
        
        L1_x1x2x3Cat = torch.cat((L1_x1,L1_x2,L1_x3), dim=1)        
        L1_x5 = self.layer1_5(L1_x1x2x3Cat)   
        L1 = L1_x4 + L1_x5
        
        L2_x1 = self.layer2_1(L1)
        L2_x2 = self.layer2_2(L2_x1)
        L2_x3 = self.layer2_3(L1)
        L2 = L2_x2 + L2_x3

        L3_x1 = self.layer3_1(L2)
        L3_x2 = self.layer3_2(L2)
        L3_x3 = self.layer3_3(L2)
        L3_x4 = self.layer3_4(L2)

        L3_x1x2x3Cat = torch.cat((L3_x1, L3_x2, L3_x3), dim=1)
        L3_x5 = self.layer3_5(L3_x1x2x3Cat)
        L3 = L3_x4 + L3_x5
        
        L3 = self.sca(L3)

        return L3


if __name__ == '__main__':
    image = torch.rand(4,3,32,512)  #(B, C, H, W)
    in_channel = 3
    out_channel = 32
    test = MRCIAM(in_channel, out_channel) 
    result = test(image)
    print(result.shape)