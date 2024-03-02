import copy
import re
import warnings
from typing import Dict, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType
from ..layers import SinePositionalEncoding
from ..layers import DetrTransformerEncoderLayer
from mmengine.model import ModuleList

# from ..layers.transformer.video_grounding_dino_layers import (
#     VideoGroundingDinoTransformerEncoder,
#     VideoGroundingDinoTransformerDecoder,
# )
from ..layers import DetrTransformerEncoderLayer
from ..layers.transformer.video_grounding_dino_layers import (
    VideoGroundingDinoTransformerEncoder,
    VideoGroundingDinoTransformerDecoder,
)

from .video_grounding_dino import VideoGroundingDINO


@MODELS.register_module()
class VideoTubeGroundingDINO(VideoGroundingDINO):

    def __init__(self, stride=3, fast_layer_cfg=None, *args, **kwargs) -> None:
        self.stride = stride
        self.fast_layer_cfg = fast_layer_cfg
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        self.encoder = VideoGroundingDinoTransformerEncoder(**self.encoder)
        self.decoder = VideoGroundingDinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, (
            f'embed_dims should be exactly 2 times of num_feats. ' f'Found {self.embed_dims} and {num_feats}.'
        )

        self.level_embed = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # text modules
        self.language_model = MODELS.build(self.language_model_cfg)
        self.text_feat_map = nn.Linear(
            self.language_model.language_backbone.body.language_dim, self.embed_dims, bias=True
        )
        if self.fast_layer_cfg is not None:
            self.fast_encoder = DetrTransformerEncoderLayer(**self.fast_layer_cfg)
            self.fast_img_residual = nn.Linear(256, 256)
            self.fast_text_residual = nn.Linear(256, 256)

    def forward_encoder(
        self,
        feat: Tensor,
        feat_mask: Tensor,
        feat_pos: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        text_dict: Dict,
    ) -> Dict:
        text_token_mask = text_dict['text_token_mask']

        if self.use_time_embed:
            time_embed = self.time_embed(feat_pos.shape[0]).to(feat_pos.device)
        else:
            time_embed = None
        if self.fast_layer_cfg is not None:
            feat_fast = feat
            memory_fast = self.fast_encoder(
                query=feat_fast.transpose(0, 1),
                query_pos=time_embed.repeat(1, feat_fast.shape[1], 1).transpose(0, 1),
                key_padding_mask=feat_mask,
            ).transpose(0, 1)

        memory, memory_text = self.encoder(
            query=feat[:: self.stride],
            query_pos=feat_pos[:: self.stride],
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios[:: self.stride],
            # for text encoder
            memory_text=text_dict['embedded'][:: self.stride],
            text_attention_mask=~text_token_mask[:: self.stride],
            position_ids=text_dict['position_ids'][:: self.stride],
            text_self_attention_masks=text_dict['masks'][:: self.stride],
            time_embed=time_embed[:: self.stride],
        )

        if self.stride:  # temporal replication
            device = memory.device
            t, n_tokens, f = feat.shape
            t, len, f = text_dict['embedded'].shape
            b=1
            durations = [t]
            pad_img_memory = torch.zeros(t, n_tokens, f).to(device)
            pad_text_memory = torch.zeros(t, len, f).to(device)
            cur_clip = 0
            n_clips = math.ceil(t / self.stride)
            for i_dur, dur in enumerate(durations):  # 将stride_t的特征复制为t个时间步
                for i_clip in range(n_clips):
                    clip_dur = min(self.stride, t - i_clip * self.stride)
                    pad_img_memory[i_clip * self.stride : i_clip * self.stride + clip_dur,] = (
                        memory[cur_clip].unsqueeze(0).repeat(clip_dur, 1, 1)
                    )
                    pad_text_memory[i_clip * self.stride : i_clip * self.stride + clip_dur,] = (
                        memory_text[cur_clip].unsqueeze(0).repeat(clip_dur, 1, 1)
                    )
                    cur_clip += 1

            # aggregate slow and fast features
            img_memory2 = pad_img_memory + memory_fast
            img_memory2 = self.fast_img_residual(img_memory2)
            memory = pad_img_memory + img_memory2
            txt_memory2 = pad_text_memory + text_dict['embedded']
            txt_memory2 = self.fast_text_residual(txt_memory2)
            memory_text = pad_text_memory + txt_memory2

        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask,
        )
        return encoder_outputs_dict
