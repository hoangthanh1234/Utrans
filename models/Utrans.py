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
from utrans_encoder import Utrans_encoder
from utrans_kpconv import Utrans_KPConv, KPClassifier
from utrans_decode import DecoderLinear, DecoderUpConv

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

        self.patch_embed = Utrans_encoder(5,32) ############################ADD code to VIT
        self.patch_size = patch_size
        self.PS_H, self.PS_W = patch_size
        self.patch_stride = patch_stride
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.n_cls = 17
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

        
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_grid_size(self, H, W):
        return self.patch_embed.get_grid_size(H, W)

    def forward(self, im, return_features=False):
        B, _, H, W = im.shape
        encoder_information = self.patch_embed(im) # encoder_information is a dictionary as follow  results = {"shortcut1","shortcut2","shortcut3","shortcut4","node1","node2":node_2,"node3":node_3,"node4","node_1_pool","node_2_pool","node_3_pool","node_4_pool","x_proj","x_latten"}
        x = encoder_information['x_latten'] 
        cls_tokens = self.cls_token.expand(B, -1, -1) #[8,1,384]  
        x = torch.cat((cls_tokens, x), dim=1) #[8,33,384]        
        pos_embed = self.pos_embed #[1,769,384]       
        num_extra_tokens = 1        
        x = x + pos_embed
        x = self.dropout(x)       
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x) #[8,32,384]               
        return x, results

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

def create_decoder(encoder, decoder_cfg):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop('name')
    decoder_cfg['d_encoder'] = encoder.d_model
    decoder_cfg['patch_size'] = encoder.patch_size

    if name == 'up_conv':
        decoder_cfg['patch_stride'] = encoder.patch_stride
        decoder = DecoderUpConv(**decoder_cfg)
    else:
        raise ValueError(f'Unknown decoder: {name}')
    return decoder

def create_utrans(model_cfg, use_kpconv=False):
    
    model_cfg = model_cfg.copy()

    decoder_cfg = model_cfg.pop('decoder')
    decoder_cfg['n_cls'] = model_cfg['n_cls']

    encoder = create_vit(model_cfg)
    decoder = create_decoder(encoder, decoder_cfg)
    

    kpclassifier = KPClassifier(
        in_channels=decoder_cfg['d_decoder'] ,
        out_channels=decoder_cfg['d_decoder'],
        num_classes=model_cfg['n_cls'])
        
    model = Utrans_KPConv(encoder, decoder, kpclassifier, n_cls=model_cfg['n_cls'])

    print(model)
    return model

class Utrans(nn.Module):
    def __init__(
        self,
        in_channels=5,
        n_cls=17,
        backbone='vit_small_patch16_384',
        image_size=(512, 1024),
        pretrained_path=None,
        new_patch_size=None,
        new_patch_stride=None,
        reuse_pos_emb=False,
        reuse_patch_emb=False,
        conv_stem='none',
        stem_base_channels=32,
        stem_hidden_dim=None,
        skip_filters=0,
        decoder='up_conv',
        up_conv_d_decoder=64,
        up_conv_scale_factor=(2, 8),
        use_kpconv=False,
        ):
        super(Utrans, self).__init__()

        self.n_cls = n_cls

        if backbone == 'vit_small_patch16_384':
            n_heads = 6
            n_layers = 12
            patch_size = 16
            dropout = 0.0
            drop_path_rate = 0.1
            d_model = 384        
        else:
            raise NameError('Not known ViT backbone.')

        decoder_cfg = {
                'n_cls': n_cls, 'name': 'up_conv',
                'd_decoder': up_conv_d_decoder, # hidden dim of the decoder
                'scale_factor': up_conv_scale_factor, # scaling factor in the PixelShuffle layer
                'skip_filters': skip_filters,} 
      
        # ViT encoder and stem config
        net_kwargs = {
            'backbone': backbone,
            'd_model': d_model, # dim of features    
            'decoder': decoder_cfg,        
            'drop_path_rate': drop_path_rate,
            'dropout': dropout,
            'channels': in_channels, # nb of channels for the 3D point projections
            'image_size': image_size,
            'n_cls': n_cls,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'patch_size': patch_size, # old patch size for the ViT encoder
            'new_patch_size': new_patch_size, # new patch size for the ViT encoder
            'new_patch_stride': new_patch_stride, # new patch stride for the ViT encoder
            'conv_stem': conv_stem,
            'stem_base_channels': stem_base_channels,
            'stem_hidden_dim': stem_hidden_dim,
        }

        # Create Utrans model
        self.utrans = create_utrans(net_kwargs, use_kpconv)
        
        old_state_dict = self.utrans.state_dict() #have value
        # for k in old_state_dict.keys():
        #     print("key: ", k)
         # Loading pre-trained weights in the ViT encoder
        if pretrained_path is not None:
            pretrained_state_dict = torch.load(pretrained_path, map_location='cpu')
            pretrained_state_dict = pretrained_state_dict['model']
              
            del pretrained_state_dict['encoder.pos_embed'] # remove positional embeddings
            del pretrained_state_dict['encoder.patch_embed.proj.weight'] # remove patch embedding layers
            del pretrained_state_dict['encoder.patch_embed.proj.bias'] # remove patch embedding layers

            # for k in pretrained_state_dict.keys():
            #     print("key: ", k)
            # Delete the pre-trained weights of the decoder
            decoder_keys = []
            for key in pretrained_state_dict.keys():
                if 'decoder' in key:
                    decoder_keys.append(key)
            for decoder_key in decoder_keys:
                del pretrained_state_dict[decoder_key]

            msg = self.utrans.load_state_dict(pretrained_state_dict, strict=False) #don't show comment load to rangeVit
            #print(f'{msg}') #print the removed layers
        
    def forward(self, *args):
        return self.utrans(*args)

if __name__ == '__main__':

    # model = Encoder(in_channels=5, base_channels=32)
    # print(model)
    # image = torch.randn(4,5, 1024, 2048)
    # x_proj, x_latten = model(image)

    parser = argparse.ArgumentParser(description='Experiment Options')
    parser.add_argument('config_path', type=str, metavar='config_path', help='path of config file, type: string')
    parser.add_argument('--data_root', type=str, required=True,help='path to the data, type: string')
    parser.add_argument('--save_path', type=str, required=True, help='path to save the file, type: string')
    parser.add_argument('--id', type=str,
                        help='name to identify the run')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of threads used for data loading, type: int')
    parser.add_argument('--pretrained_model', type=str,
                        help='path of pre-trained model to initialize the ViT encoder backbone, type: string')
    parser.add_argument('--checkpoint', type=str,
                        help='path of checkpoint model for resuming training or evaluation, type: string')
    parser.add_argument('--window_stride', type=int,
                        help='sliding window stride during validation, type: int')
    parser.add_argument('--mini', action='store_true', help='use mini version of the dataset, type: bool')
    parser.add_argument('--val_only', action='store_true', help='run inference only')
    parser.add_argument('--test_split', action='store_true', help='run inference on the test split')
    parser.add_argument('--save_eval_results', action='store_true', help='save the predictions')
    parser.add_argument('--log_frequency', type=int, default=100, help='logging frequency')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    args = parser.parse_args()
    settings = Option(args.config_path, args)

    model = Utrans(
        in_channels=settings.in_channels,
        n_cls=settings.n_classes,
        backbone=settings.vit_backbone,
        image_size=settings.image_size,
        pretrained_path=args.pretrained_model,
        new_patch_size=settings.patch_size,
        new_patch_stride=settings.patch_stride,
        reuse_pos_emb=settings.reuse_pos_emb,
        reuse_patch_emb=settings.reuse_patch_emb,
        conv_stem=settings.conv_stem,
        stem_base_channels=settings.stem_base_channels,
        stem_hidden_dim=settings.D_h,
        skip_filters=settings.skip_filters,
        decoder=settings.decoder,
        up_conv_d_decoder=settings.D_h,
        up_conv_scale_factor=settings.patch_stride,
        use_kpconv=settings.use_kpconv) 

    image = torch.randn(8,5, 512, 1024)
    print(model)
    model(image,2,3,4,4,5)


    
        