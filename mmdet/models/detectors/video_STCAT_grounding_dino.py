# Copyright (c) OpenMMLab. All rights reserved.
import copy
import re
import warnings
from typing import Dict, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType
from ..layers import SinePositionalEncoding
from ..layers.transformer.video_grounding_dino_layers import (
    VideoGroundingDinoTransformerEncoder,
    VideoGroundingDinoTransformerDecoder,
)
from ..layers.transformer.video_STCAT_layer import VideoSTCATDinoTransformerDecoder

from .video_grounding_dino import VideoGroundingDINO


@MODELS.register_module()
class VideoSTCATGroundingDINO(VideoGroundingDINO):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # print(self)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        self.encoder = VideoGroundingDinoTransformerEncoder(**self.encoder)
        self.decoder = VideoSTCATDinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        self.time_query = nn.Embedding(1, self.embed_dims)
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

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes
        )

        enc_outputs_class = self.bbox_head.cls_branches[self.decoder.num_layers](
            output_memory, memory_text, text_token_mask
        )
        cls_out_features = self.bbox_head.cls_branches[self.decoder.num_layers].max_text_len
        enc_outputs_coord_unact = (
            self.bbox_head.reg_branches[self.decoder.num_layers](output_memory) + output_proposals
        )

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]

        topk_score = torch.gather(
            enc_outputs_class, 1, topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features)
        )
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 4)
        )
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        if self.bbox_head.use_enc_sted:
            enc_outputs_sted = self.bbox_head.sted_branch[0](output_memory)
            topk_sted = torch.gather(enc_outputs_sted, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 2))

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)

        if self.use_time_embed:
            time_query = self.time_query.weight[None, :, :]
            time_query = time_query.repeat(1, bs, 1).transpose(0, 1).to(query.device)

        # if self.training and self.use_dn:
        #     dn_label_query, dn_bbox_query, dn_mask, dn_meta = self.dn_query_generator(batch_data_samples)
        #     query = torch.cat([dn_label_query, query], dim=1)
        #     reference_points = torch.cat([dn_bbox_query, topk_coords_unact], dim=1)
        # else:
        #     reference_points = topk_coords_unact
        #     dn_mask, dn_meta = None, None
        reference_points = topk_coords_unact
        dn_mask, dn_meta = None, None

        reference_points = reference_points.sigmoid()

        '''video'''
        t = query.shape[0]
        b = 1
        # if durations is not None:
        #     t = max(durations)  # all_frame
        #     b = len(durations)
        #     bs_oq = bs if (not self.stride) else b * t
        assert bs == b * t
        # add temporal encoding to init time queries
        if self.use_time_embed:
            time_embed = (
                self.time_embed(bs).repeat(b, 1, 1).to(query.device)
            )  # n_queries * t, b, 256
            decoder_inputs_dict = dict(
                query=query,
                time_query=time_query,
                memory=memory,
                reference_points=reference_points,
                dn_mask=dn_mask,
                memory_text=memory_text,
                text_attention_mask=~text_token_mask,
                time_embed=time_embed,
            )
        else:
            decoder_inputs_dict = dict(
                query=query,
                memory=memory,
                reference_points=reference_points,
                dn_mask=dn_mask,
                memory_text=memory_text,
                text_attention_mask=~text_token_mask,
            )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        if self.bbox_head.use_enc_sted:
            head_inputs_dict = (
                dict(
                    enc_outputs_class=topk_score,
                    enc_outputs_coord=topk_coords,
                    dn_meta=dn_meta,
                    enc_outputs_sted=topk_sted,
                )
                if self.training
                else dict()
            )
        else:
            head_inputs_dict = (
                dict(enc_outputs_class=topk_score, enc_outputs_coord=topk_coords, dn_meta=dn_meta)
                if self.training
                else dict()
            )
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(
        self,
        query: Tensor,
        memory: Tensor,
        memory_mask: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        dn_mask: Optional[Tensor] = None,
        time_query: Optional[Tensor] = None,
        time_embed: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict:
        """Forward with Transformer decoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries_total, dim), where `num_queries_total` is the
                sum of `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries_total, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            dn_mask (Tensor, optional): The attention mask to prevent
                information leakage from different denoising groups and
                matching parts, will be used as `self_attn_mask` of the
                `self.decoder`, has shape (num_queries_total,
                num_queries_total).
                It is `None` when `self.training` is `False`.
            time_embed (Tensor, optional): The temporal encoding of the batch. Default: None, has shape[t,1, embed_dims]

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output and `references` including
            the initial and intermediate reference_points.
        """
        # if self.use_time_embed:
        #     inter_states, references, output_time, weights = self.decoder(
        #         query=query,
        #         time_query=time_query,
        #         value=memory,
        #         key_padding_mask=memory_mask,
        #         self_attn_mask=dn_mask,
        #         reference_points=reference_points,
        #         spatial_shapes=spatial_shapes,
        #         level_start_index=level_start_index,
        #         valid_ratios=valid_ratios,
        #         reg_branches=self.bbox_head.reg_branches,
        #         time_embed=time_embed,
        #         **kwargs,
        #     )
        #     decoder_outputs_dict = dict(
        #         hidden_states=inter_states, references=list(references), output_time=output_time
        #     )
        # else:
        #     inter_states, references = self.decoder(
        #         query=query,
        #         time_query=time_query,
        #         value=memory,
        #         key_padding_mask=memory_mask,
        #         self_attn_mask=dn_mask,
        #         reference_points=reference_points,
        #         spatial_shapes=spatial_shapes,
        #         level_start_index=level_start_index,
        #         valid_ratios=valid_ratios,
        #         reg_branches=self.bbox_head.reg_branches,
        #         time_embed=time_embed,
        #         **kwargs,
        #     )
        #     decoder_outputs_dict = dict(hidden_states=inter_states, references=list(references))
        # return decoder_outputs_dict
        inter_states, references, output_time, weights = self.decoder(
                query=query,
                time_query=time_query,
                value=memory,
                key_padding_mask=memory_mask,
                self_attn_mask=dn_mask,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reg_branches=self.bbox_head.reg_branches,
                time_embed=time_embed,
                **kwargs,
            )
        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=list(references), output_time=output_time,weights=weights
        )
        return decoder_outputs_dict

        # if len(query) == self.num_queries:
        #     # NOTE: This is to make sure label_embeding can be involved to
        #     # produce loss even if there is no denoising query (no ground truth
        #     # target in this GPU), otherwise, this will raise runtime error in
        #     # distributed training.
        #     inter_states[0] += self.dn_query_generator.label_embedding.weight[0, 0] * 0.0
