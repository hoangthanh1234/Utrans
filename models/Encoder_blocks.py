import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import get_grid_size_1d, get_grid_size_2d

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

        return output, output_pool



class Encoder(nn.Module):
    def __init__(self,
                 in_channels=5,
                 base_channels=32,
                 img_size=(512, 1024),
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
        print(" self.grid_size: ",  self.grid_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

    def get_grid_size(self, H, W):
        return get_grid_size_2d(H, W, self.patch_size, self.patch_stride)

    def forward(self, x):
        B, C, H, W = x.shape  # B, in_channels, image_size[0], image_size[1]

        node_1, node_1_pool = self.node_1(x) #[1,32, 128, 256]
        
        re_pool = self.pool(node_1_pool)     

        node_2, node_2_pool = self.node_2(re_pool) #[1, 64, 64, 128]

        #re_pool2 = self.pool(node_2_pool) 
       
        node_3,node_3_pool = self.node_3(node_2_pool) #[1, 128, 32, 64]
      
        node_4, node_4_pool = self.node_4(node_3_pool) #[1, 256, 16, 32]
       
        print("node_4_pool: ", node_4_pool.shape)
        x_proj = self.proj_block(node_4_pool)

        print("x_proj", x_proj.shape)
        
        x_latten = x_proj.flatten(2).transpose(1, 2)  # BCHW -> BNC [1,384,8,4] -> [1,32,384]        
        
        print(x_latten.shape)

        return x_proj, x_latten

#class Decode(nn.Module):


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        n_layers,
        d_model,
        d_ff,
        n_heads,
        n_cls,
        dropout=0.1,
        drop_path_rate=0.0,
        channels=3,
        ls_init_values=None,
        patch_stride=None,
        conv_stem='none',
        stem_base_channels=32,
        stem_hidden_dim=None,
    ):
        super().__init__()

        self.conv_stem = conv_stem

        # in this case self.conv_stem = 'ConvStem'
        assert patch_stride == patch_size # patch_size = patch_stride if a convolutional stem is used       
        self.patch_embed = ConvStem(
            in_channels=channels,
            base_channels=stem_base_channels,
            img_size=image_size,
            patch_stride=patch_stride,
            embed_dim=d_model,
            flatten=True,
            hidden_dim=stem_hidden_dim)

        self.patch_size = patch_size
        self.PS_H, self.PS_W = patch_size
        self.patch_stride = patch_stride
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.n_cls = n_cls
        self.image_size = image_size

        # cls and pos tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.patch_embed.num_patches + 1, d_model))

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]

        self.blocks = nn.ModuleList(
                [Block(d_model, n_heads, d_ff, dropout, dpr[i], init_values=ls_init_values) for i in range(n_layers)]
            )

        self.norm = nn.LayerNorm(d_model)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_grid_size(self, H, W):
        return self.patch_embed.get_grid_size(H, W)

    def forward(self, im, return_features=False):
        B, _, H, W = im.shape
        x, skip = self.patch_embed(im) # x.shape = [16, 576, 384]

        cls_tokens = self.cls_token.expand(B, -1, -1) #[8,1,384]  
        x = torch.cat((cls_tokens, x), dim=1) # x.shape = [16, 577, 384] or [8,769,384]
        pos_embed = self.pos_embed #[1,769,384]
        num_extra_tokens = 1        
        x = x + pos_embed
        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x) #[8,769,384]        
        return x, skip  # x.shape = [16, 577, 384] | skip.shape [8,256,32,384]


def create_vit(model_cfg):
    model_cfg = model_cfg.copy()
    model_cfg.pop('backbone')
    mlp_expansion_ratio = 4
    model_cfg['d_ff'] = mlp_expansion_ratio * model_cfg['d_model']

    new_patch_size = model_cfg.pop('new_patch_size')
    new_patch_stride = model_cfg.pop('new_patch_stride')

    if (new_patch_size is not None):
        if new_patch_stride is None:
            new_patch_stride = new_patch_size
        model_cfg['patch_size'] = new_patch_size
        model_cfg['patch_stride'] = new_patch_stride

    model = VisionTransformer(**model_cfg)

    return model



if __name__ == '__main__':

    model = Encoder(in_channels=5, base_channels=32)
    
    image = torch.randn(1,5, 512, 1024)

    x_proj, x_latten = model(image)

    
        