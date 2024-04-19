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
from .utrans_encoder import Utrans_encoder
from .utrans_kpconv import Utrans_KPConv, KPClassifier
from .utrans_decoder import DecoderUpConv
from .swin_transformer import Small_swin

class VisionTransformer(nn.Module):
    def __init__(
        self,
        in_channels,
        encode_hidden_dim,
        bridge_in_channel,
        bridge_hidden,
        window_size    
    ):
        super().__init__()
        

        self.patch_embed = Utrans_encoder(in_channels, encode_hidden_dim) ############################ADD code to VIT
        self.bridge = Small_swin(in_channels=bridge_in_channel,hidden_dimension=bridge_hidden,window_size=window_size)          

    def forward(self, im, return_features=False):
        B, _, H, W = im.shape
        encoder_information = self.patch_embed(im) # encoder_information is a dictionary as follow  results = {"shortcut1","shortcut2","shortcut3","shortcut4","node1","node2":node_2,"node3":node_3,"node4","node_1_pool","node_2_pool","node_3_pool","node_4_pool","x_proj","x_latten"}               
        return encoder_information

def create_encoder(encoder_cfg):
    encoder_cfg = encoder_cfg.copy()
    print("**encoder_cfg: ", encoder_cfg)
    model = VisionTransformer(**encoder_cfg)
    return model

def create_decoder(encoder, decoder_cfg):
    decoder_cfg = decoder_cfg.copy()       
    decoder = DecoderUpConv(**decoder_cfg)    
    return decoder

def create_utrans(model_cfg=None, use_kpconv=False, encoder_cfg=None):
    
    model_cfg = model_cfg.copy()

    encoder =create_encoder(encoder_cfg)

    decoder_cfg = model_cfg.pop('decoder')   
    decoder = create_decoder(encoder, decoder_cfg)
    

    kpclassifier = KPClassifier(
        in_channels=256 ,
        out_channels=256,
        num_classes=17)


    model = Utrans_KPConv(encoder, decoder, kpclassifier, n_cls=17)

  
    return model

class Utrans(nn.Module):
    def __init__(
        self,
        in_channels=5,
        n_cls=17,
        backbone='vit_small_patch16_384',
        image_size=(256, 512),
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
        encode_hidden_dim=32,
        bridge_in_channel=256,
        bridge_hidden=128,
        window_size=4 
        ):
        super(Utrans, self).__init__()

        self.n_cls = n_cls

        if backbone == 'vit_small_patch16_384':
            n_heads = 6
            n_layers = 12
            patch_size = 16
            dropout = 0.1
            drop_path_rate = 0.1
                  
        else:
            raise NameError('Not known ViT backbone.')

        encoder_cfg = {
            'in_channels':in_channels,
            'encode_hidden_dim':encode_hidden_dim,
            'bridge_in_channel':bridge_in_channel,
            'bridge_hidden':bridge_hidden,
            'window_size':window_size
        }

        decoder_cfg = {               
                'dropout_rate': 0.2
                } 

        # ViT encoder and stem config
        net_kwargs = {
            'backbone': backbone,               
            'decoder': decoder_cfg,        
            'drop_path_rate': drop_path_rate,
            'dropout': dropout,
            'channels': in_channels, # nb of channels for the 3D point projections
            'image_size': image_size,            
            'n_heads': n_heads,
            'n_layers': n_layers,
            'patch_size': patch_size, # old patch size for the ViT encoder
            'new_patch_size': new_patch_size, # new patch size for the ViT encoder
            'new_patch_stride': new_patch_stride, # new patch stride for the ViT encoder
            'conv_stem': conv_stem,
            'stem_base_channels': stem_base_channels,
            'stem_hidden_dim': stem_hidden_dim,
            'encode_hidden_dim':encode_hidden_dim,
            'bridge_in_channel':bridge_in_channel,
            'bridge_hidden':bridge_hidden,
            'window_size':window_size
            
        }

        # Create Utrans model
        self.utrans = create_utrans(net_kwargs, use_kpconv, encoder_cfg)
        
        old_state_dict = self.utrans.state_dict() #have value        
         # Loading pre-trained weights in the ViT encoder
        if pretrained_path is not None:
            pretrained_state_dict = torch.load(pretrained_path, map_location='cpu')
            pretrained_state_dict = pretrained_state_dict['model']
              
            del pretrained_state_dict['encoder.pos_embed'] # remove positional embeddings
            del pretrained_state_dict['encoder.patch_embed.proj.weight'] # remove patch embedding layers
            del pretrained_state_dict['encoder.patch_embed.proj.bias'] # remove patch embedding layers

                       
            # Delete the pre-trained weights of the decoder
            decoder_keys = []
            for key in pretrained_state_dict.keys():
                if 'decoder' in key:
                    decoder_keys.append(key)
            for decoder_key in decoder_keys:
                del pretrained_state_dict[decoder_key]

            msg = self.utrans.load_state_dict(pretrained_state_dict, strict=False) #don't show comment load to Utrans
            #print(f'{msg}') #print the removed layers
    def counter_model_parameters(self):
        stats = {}
        stats['total_num_parameters'] = count_parameters(self.utrans)
        stats['decoder_num_parameters'] = count_parameters(self.utrans.decoder)
        stats['patch_num_parameters'] = count_parameters(self.utrans.encoder.patch_embed)
        stats['encoder_num_parameters'] = count_parameters(self.utrans.encoder) - stats['patch_num_parameters']
        return stats
        
    def forward(self, *args):
        return self.utrans(*args)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

        