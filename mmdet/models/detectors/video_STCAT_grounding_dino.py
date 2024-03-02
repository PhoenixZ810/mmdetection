# Copyright (c) OpenMMLab. All rights reserved.
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
from mmengine.model import ModuleList
# from ..layers.transformer.video_grounding_dino_layers import (
#     VideoGroundingDinoTransformerEncoder,
#     VideoGroundingDinoTransformerDecoder,
# )
from ..layers import DetrTransformerEncoderLayer
from ..layers.transformer.video_STCAT_layer import VideoSTCATDinoTransformerDecoder, VideoSTCATGroundingDinoTransformerEncoder

from .video_grounding_dino import VideoGroundingDINO


@MODELS.register_module()
class VideoSTCATGroundingDINO(VideoGroundingDINO):
    def __init__(
        self,
        frame_layer_cfg: Optional[ConfigType] = None,
        use_time_query: bool = True,
        *args,
        **kwargs,
    ) -> None:
        self.frame_layer_cfg = frame_layer_cfg
        self.use_time_query = use_time_query
        super().__init__(*args, **kwargs)
        self.video_cls = nn.Embedding(1, 256)  # the video level global cls token
        # print(self)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        self.encoder = VideoSTCATGroundingDinoTransformerEncoder(**self.encoder)
        self.decoder = VideoSTCATDinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        if self.use_time_query:
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
        # self.spatial_temporal_layer = SpatialTemporalEncoder(
        #     encoder_layer=TransformerEncoderLayer(), num_layers=6
        # )
        self.template_generator = TemplateGenerator()
        if self.frame_layer_cfg is not None:
            self.video_layers = DetrTransformerEncoderLayer(**self.frame_layer_cfg)
            # self.video_layers = ModuleList(
            #     [DetrTransformerEncoderLayer(**self.frame_layer_cfg) for _ in range(3)]
            # )

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
            # time_embed = self.time_embed(feat_pos.shape[0]).to(feat_pos.device)
            time_embed = self.time_embed
        else:
            time_embed = None
        # memory, memory_text, frames_src, video_src = self.encoder(
        memory, memory_text = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text encoder
            memory_text=text_dict['embedded'],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'],
            time_embed=time_embed,
        )
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            memory_pos=feat_pos,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask,
            # frames_cls=frames_src,
            # video_cls=video_src
        )
        return encoder_outputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        memory_pos=None,
        # frames_cls=None,
        # video_cls=None,
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
        )  # [t, l, 4]

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
        )  # [t,1,4]
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        # if self.bbox_head.use_enc_sted:
        #     # enc_outputs_sted = self.bbox_head.sted_branch[0](frames_cls.unsqueeze(1))
        #     # topk_sted = enc_outputs_sted
        #     enc_outputs_sted = self.bbox_head.sted_branch[0](output_memory)
        #     topk_sted = torch.gather(enc_outputs_sted, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 2))

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)

        if self.use_time_query:
            time_query = self.time_query.weight[None, :, :]
            time_query = time_query.repeat(1, bs, 1).transpose(0, 1).to(query.device)

        vis_durations = batch_data_samples[0].durations
        b = len(vis_durations)
        t = max(vis_durations)
        frames_cls = torch.gather(output_memory, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 256))
        # The position embedding, token mask, in temporal layer
        video_src = self.video_cls.weight.unsqueeze(0).repeat(1, 1, 1)  # b x 1 x d_model
        temp_pos = self.time_embed(t + 1).repeat(1, b, 1)  # (T + 1) x b x d_model
        temp_mask = torch.ones(b, t + 1).bool().to(query.device)
        temp_mask[:, 0] = False  # the mask for the video cls token
        for i_dur, dur in enumerate(vis_durations):
            temp_mask[i_dur, 1 : 1 + dur] = False
        frames_src = torch.zeros(b, t + 1, 256).to(query.device)  # b x seq_len x C
        for i_dur, dur in enumerate(vis_durations):
            frames_src[i_dur, 0:1, :] = video_src[i_dur]  # pad the video cls token
            frames_src[i_dur, 1 : 1 + dur, :] = frames_cls[:,0,:]  # [1,t+1,256]

        if self.frame_layer_cfg is not None:
            frames_src = self.video_layers(
                query=frames_src,
                query_pos=temp_pos.transpose(0, 1),
                key_padding_mask=temp_mask,
            )
            # for layer in self.video_layers:
            #     frames_src = layer(
            #         query=frames_src,
            #         query_pos=temp_pos.transpose(0, 1),
            #         key_padding_mask=temp_mask,
            #     )
        frames_src_list = []
        for i_dur, dur in enumerate(vis_durations):
            video_src[i_dur] = frames_src[i_dur, 0:1]  # video_src[1,1,256]
            frames_src_list.append(frames_src[i_dur, 1 : 1 + dur])  # LenxC

        frames_cls = torch.cat(frames_src_list, dim=0)
        video_cls = video_src.squeeze(0)

        if self.bbox_head.use_enc_sted:
            enc_outputs_sted = self.bbox_head.sted_branches[-1](frames_cls.unsqueeze(1))
            topk_sted = enc_outputs_sted

        # img_memory, frames_cls, videos_cls = self.spatial_temporal_layer(
        #     memory,
        #     src_key_padding_mask=memory_mask,
        #     pos=memory_pos,
        #     durations=vis_durations,
        #     time_embed=self.time_embed,
        # )  # frame_cls[t,d_model], video_cls[b,d_model]
        temp_query, temp_frames_query = self.template_generator(frames_cls, video_cls, vis_durations)
        if not self.use_time_query:
            time_query = video_cls.unsqueeze(0).repeat(t, 1, 1)  # [t, b, 256]
        temp_query = torch.split(temp_query, vis_durations, dim=0)
        # tgt = torch.zeros(t, b, self.d_model).to(query.device)
        # time_tgt = torch.zeros(t, b, self.d_model).to(query.device)

        # The position embedding of query
        t, _, d_model = query.shape
        b = 1
        query_temporal_embed = torch.zeros(b, t, d_model).to(query.device)
        query_mask = torch.ones(b, t).bool().to(query.device)
        query_mask[:, 0] = False  # avoid empty masks

        for i_dur, dur in enumerate(vis_durations):
            query_mask[i_dur, :dur] = False
            query_temporal_embed[i_dur, :dur, :] = temp_query[i_dur]

        query_temporal_embed = query_temporal_embed.permute(1, 0, 2)  # [n_frames, bs, d_model]

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
            time_embed = self.time_embed(bs).repeat(b, 1, 1).to(query.device)  # n_queries * t, b, 256
            decoder_inputs_dict = dict(
                query=query,
                time_query=time_query,
                memory=memory,
                reference_points=reference_points,
                dn_mask=dn_mask,
                memory_text=memory_text,
                text_attention_mask=~text_token_mask,
                time_embed=time_embed,
                time_query_pos=query_temporal_embed,
                time_query_frame_pos=temp_frames_query,
                memory_pos = memory_pos
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
        time_query_pos: Optional[Tensor] = None,
        time_query_frame_pos: Optional[Tensor] = None,
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
            time_query_pos=time_query_pos,
            time_query_frame_pos=time_query_frame_pos,
            **kwargs,
        )
        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=list(references), output_time=output_time, weights=weights
        )
        return decoder_outputs_dict

        # if len(query) == self.num_queries:
        #     # NOTE: This is to make sure label_embeding can be involved to
        #     # produce loss even if there is no denoising query (no ground truth
        #     # target in this GPU), otherwise, this will raise runtime error in
        #     # distributed training.
        #     inter_states[0] += self.dn_query_generator.label_embedding.weight[0, 0] * 0.0


class SpatialTemporalEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers=6, norm=None, return_weights=False, d_model=256):
        super().__init__()
        self.spatial_layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.temporal_layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.d_model = d_model

        # The position embedding of local frame tokens
        self.local_pos_embed = nn.Embedding(1, d_model)  # the learned pos embed for frame cls token

        # The learnd local and global embedding
        self.frame_cls = nn.Embedding(1, d_model)  # the frame level local cls token
        self.video_cls = nn.Embedding(1, d_model)  # the video level global cls token

        self.num_layers = num_layers
        self.norm = norm
        self.return_weights = return_weights

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        durations=None,
        time_embed=None,
    ):
        output = src
        b = len(durations)
        t = max(durations)
        n_frames = sum(durations)
        device = output.device

        # The position embedding, token mask, src feature for local frame token, in spatial layer
        frame_src = self.frame_cls.weight.unsqueeze(1).repeat(1, n_frames, 1)  # 1 x n_frames X d_model
        frame_pos = self.local_pos_embed.weight.unsqueeze(1).repeat(1, n_frames, 1)  # 1 x n_frames X d_model
        frame_mask = torch.zeros((n_frames, 1)).bool().to(device)
        output = output.transpose(0, 1)
        output = torch.cat([frame_src, output], dim=0)  # local_frames + fused_features
        if src_key_padding_mask is not None:
            src_key_padding_mask = torch.cat([frame_mask, src_key_padding_mask], dim=1)
        pos = pos.transpose(0, 1)
        pos = torch.cat([frame_pos, pos], dim=0)

        # The position embedding, token mask, in temporal layer
        video_src = self.video_cls.weight.unsqueeze(0).repeat(b, 1, 1)  # b x 1 x d_model
        temp_pos = time_embed(t + 1).repeat(1, b, 1)  # (T + 1) x b x d_model
        temp_mask = torch.ones(b, t + 1).bool().to(device)
        temp_mask[:, 0] = False  # the mask for the video cls token
        for i_dur, dur in enumerate(durations):
            temp_mask[i_dur, 1 : 1 + dur] = False

        for i_layer, layer in enumerate(self.spatial_layers):
            # spatial interaction on each single frame
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

            frames_src = torch.zeros(b, t + 1, self.d_model).to(device)  # b x seq_len x C
            frames_src_list = torch.split(output[0, :, :], durations)  # [(n_frames, C)]

            for i_dur, dur in enumerate(durations):
                frames_src[i_dur, 0:1, :] = video_src[i_dur]  # pad the video cls token
                frames_src[i_dur, 1 : 1 + dur, :] = frames_src_list[i_dur]

            frames_src = frames_src.permute(1, 0, 2)  # permute BxLenxC to LenxBxC, [t+1,1, 256]

            # temporal interaction between all video frames
            frames_src = self.temporal_layers[i_layer](
                frames_src, src_mask=None, src_key_padding_mask=temp_mask, pos=temp_pos
            )

            frames_src = frames_src.permute(1, 0, 2)  # permute LenxBxC to BxLenxC
            # dispatch the temporal context to each single frame token
            frames_src_list = []
            for i_dur, dur in enumerate(durations):
                video_src[i_dur] = frames_src[i_dur, 0:1]
                frames_src_list.append(frames_src[i_dur, 1 : 1 + dur])  # LenxC

            frames_src = torch.cat(frames_src_list, dim=0)
            output[0, :, :] = frames_src

        if self.norm is not None:
            output = self.norm(output)

        frame_src = output[0, :, :]  # t,256
        output = output[1:, :, :]
        video_src = video_src.squeeze(1)  # b x 1 x d_model => b x d_model, 1,256

        return output, frame_src, video_src


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TemplateGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 256
        self.pos_query_dim = 4
        self.content_proj = nn.Linear(self.d_model, self.d_model)
        self.time_pos_proj = nn.Linear(self.d_model, self.d_model)
        # self.gamma_proj = nn.Linear(self.d_model, self.d_model)
        # self.beta_proj = nn.Linear(self.d_model, self.d_model)
        # self.anchor_proj = nn.Linear(self.d_model, self.pos_query_dim)

    def forward(
        self, frames_cls=None, videos_cls=None, durations=None,  # [b, d_model]  # [b, d_model]
    ):
        # b = len(durations)
        # frames_cls_list = torch.split(frames_cls, durations, dim=0)
        # content_query = self.content_proj(videos_cls)

        # pos_query = []
        # temp_query = []
        # for i_b in range(b):
        #     frames_cls = frames_cls_list[i_b]
        #     # video_cls = videos_cls[i_b]
        #     # gamma_vec = torch.tanh(self.gamma_proj(video_cls))
        #     # beta_vec = torch.tanh(self.beta_proj(video_cls))
        #     # pos_query.append(self.anchor_proj(gamma_vec * frames_cls + beta_vec))
        #     temp_query.append(content_query[i_b].unsqueeze(0).repeat(frames_cls.shape[0], 1))

        # # pos_query = torch.cat(pos_query, dim=0)
        # temp_query = torch.cat(temp_query, dim=0)

        # # return pos_query, temp_query
        # return temp_query
        b = len(durations)
        frames_cls_list = torch.split(frames_cls, durations, dim=0)
        content_query = self.content_proj(videos_cls)

        pos_query = []
        temp_query = []
        for i_b in range(b):
            frames_cls = frames_cls_list[i_b]
            frame_query = self.time_pos_proj(frames_cls).unsqueeze(1)
            # video_cls = videos_cls[i_b]
            # gamma_vec = torch.tanh(self.gamma_proj(video_cls))
            # beta_vec = torch.tanh(self.beta_proj(video_cls))
            # pos_query.append(self.anchor_proj(gamma_vec * frames_cls + beta_vec))
            temp_query.append(content_query[i_b].unsqueeze(0).repeat(frames_cls.shape[0], 1))

        # pos_query = torch.cat(pos_query, dim=0)
        temp_query = torch.cat(temp_query, dim=0)

        # return pos_query, temp_query
        return temp_query, frame_query
