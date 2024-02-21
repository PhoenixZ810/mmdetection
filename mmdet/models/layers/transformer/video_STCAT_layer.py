# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Optional, Tuple, Union
import copy
import torch.nn.functional as F

import torch
from mmengine.model import BaseModule
from torch import Tensor, nn

from mmcv.ops import MultiScaleDeformableAttention
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention

from .grounding_dino_layers import (
    GroundingDinoTransformerEncoder,
    GroundingDinoTransformerDecoder,
    GroundingDinoTransformerDecoderLayer,
)
from .video_grounding_dino_layers import VideoGroundingDinoTransformerEncoder,VideoGroundingDinoTransformerDecoderLayer
from mmengine.model import ModuleList
from .utils import MLP, coordinate_to_encoding, inverse_sigmoid

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None


class VideoSTCATGroundingDinoTransformerEncoder(VideoGroundingDinoTransformerEncoder):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.frame_cls = nn.Embedding(1, 256)  # the frame level local cls token
        self.video_cls = nn.Embedding(1, 256)  # the video level global cls token

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        if self.time_attn_layer_cfg is not None:
            self.time_attn_layers = ModuleList(
                [DetrTransformerEncoderLayer(**self.time_attn_layer_cfg) for _ in range(self.num_layers)]
            )
        self.layers = ModuleList(
            [DeformableDetrTransformerEncoderLayer(**self.layer_cfg) for _ in range(self.num_layers)]
        )
        self.text_layers = ModuleList(
            [DetrTransformerEncoderLayer(**self.text_layer_cfg) for _ in range(self.num_layers)]
        )
        self.fusion_layers = ModuleList(
            [SingleScaleBiAttentionBlock(**self.fusion_layer_cfg) for _ in range(self.num_layers)]
        )
        self.embed_dims = self.layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.'
                )
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
                self.fusion_layers[i] = checkpoint_wrapper(self.fusion_layers[i])

    def forward(
        self,
        query: Tensor,
        query_pos: Tensor,
        key_padding_mask: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        memory_text: Tensor = None,
        text_attention_mask: Tensor = None,
        pos_text: Tensor = None,
        text_self_attention_masks: Tensor = None,
        position_ids: Tensor = None,
        time_embed: Optional[Tensor] = None,
    ):
        """Forward function of Transformer encoder.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, has shape
                (bs, num_queries, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (bs, num_queries).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device
        )
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text, device=memory_text.device)
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .repeat(bs, 1, 1)
                )
                pos_text = get_text_sine_pos_embed(pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None], num_pos_feats=256, exchange_xy=False
                )
        '''global & local query generate'''
        durations = [query.shape[0]]
        b = len(durations)
        t = max(durations)
        n_frames = sum(durations)
        device = output.device
        # The position embedding, token mask, src feature for local frame token, in spatial layer
        frame_src = self.frame_cls.weight.unsqueeze(1).repeat(1, n_frames, 1)  # 1 x n_frames X d_model
        frame_pos = self.local_pos_embed.weight.unsqueeze(1).repeat(1, n_frames, 1)  # 1 x n_frames X d_model
        frame_mask = torch.zeros((n_frames, 1)).bool().to(device)

        # The position embedding, token mask, in temporal layer
        video_src = self.video_cls.weight.unsqueeze(0).repeat(b, 1, 1)  # b x 1 x d_model
        temp_pos = time_embed(t + 1).repeat(1, b, 1)  # (T + 1) x b x d_model
        temp_mask = torch.ones(b, t + 1).bool().to(device)
        temp_mask[:, 0] = False  # the mask for the video cls token
        for i_dur, dur in enumerate(durations):
            temp_mask[i_dur, 1 : 1 + dur] = False

        # main process
        for layer_id, layer in enumerate(self.layers):
            if self.fusion_layers:
                output, memory_text = self.fusion_layers[layer_id](
                    visual_feature=output,
                    lang_feature=memory_text,
                    attention_mask_v=key_padding_mask,
                    attention_mask_l=text_attention_mask,
                )
            if self.text_layers:
                text_num_heads = self.text_layers[layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1
                    ),  # note we use ~ for mask here
                    key_padding_mask=None,
                )

            # if layer_id == 0:
            output = output.transpose(0, 1)
            output = torch.cat([frame_src, output], dim=0)  # local_frames + fused_features
            if src_key_padding_mask is not None:
                src_key_padding_mask = torch.cat([frame_mask, src_key_padding_mask], dim=1)
            pos = pos.transpose(0, 1)
            pos = torch.cat([frame_pos, pos], dim=0)

            output = layer(
                query=output.transpose(0, 1),
                query_pos=pos.transpose(0, 1),
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=src_key_padding_mask,
            )

            frames_src = torch.zeros(b, t + 1, 256).to(device)  # b x seq_len x C
            frames_src_list = torch.split(output[:, 0, :], durations)  # [(n_frames, C)]

            for i_dur, dur in enumerate(durations):
                frames_src[i_dur, 0:1, :] = video_src[i_dur]  # pad the video cls token
                frames_src[i_dur, 1 : 1 + dur, :] = frames_src_list[i_dur]  # [1,t+1,256]

            # frames_src = frames_src.permute(1, 0, 2)  # permute BxLenxC to LenxBxC, [t+1,1,256]

            if self.time_attn_layer_cfg is not None:
                frames_src = self.time_attn_layers[layer_id](
                    query=frame_src,
                    query_pos=temp_pos.transpose(0, 1),
                    key_padding_mask=temp_mask,
                ).transpose(0, 1)

            # frames_src = frames_src.permute(1, 0, 2)  # permute LenxBxC to BxLenxC
            # dispatch the temporal context to each single frame token
            frames_src_list = []
            for i_dur, dur in enumerate(durations):
                video_src[i_dur] = frames_src[i_dur, 0:1]
                frames_src_list.append(frames_src[i_dur, 1 : 1 + dur])  # LenxC

            frames_src = torch.cat(frames_src_list, dim=0)
            output[0, :, :] = frames_src
            output = output.transpose(0, 1)

        return output, memory_text, frames_src, video_src


class VideoSTCATDinoTransformerDecoder(GroundingDinoTransformerDecoder):
    """Transformer decoder of VideoDINO."""

    def __init__(self, use_weight_loss=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_weight_loss = use_weight_loss

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList(
            [VideoGroundingDinoTransformerDecoderLayer(**self.layer_cfg) for _ in range(self.num_layers)]
        )
        self.temp_layer = ModuleList([TimeDecoderLayer() for _ in range(self.num_layers)])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in ' f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims, self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    def forward(
        self,
        query: Tensor,
        value: Tensor,
        key_padding_mask: Tensor,
        self_attn_mask: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        reg_branches: nn.ModuleList,
        time_query: Optional[Tensor] = None,
        time_embed: Optional[Tensor] = None,
        time_query_pos: Optional[Tensor] = None,
        memory_pos: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        intermediate = []
        intermediate_reference_points = [reference_points]
        intermediate_weights = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

            # video decode
            query_sine_embed = coordinate_to_encoding(reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)
            query,weights_spa = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                time_embed=time_embed,
                **kwargs,
            )
            if time_query is not None:
                time_query, weights = self.temp_layer[lid](
                    tgt=time_query,
                    memory=value,
                    pos = memory_pos,
                    query_pos=time_query_pos,
                    query_time_pos=time_embed,
                    self_attn_mask=self_attn_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    reference_points=reference_points_input,
                    **kwargs,
                )

            if reg_branches is not None:
                tmp = reg_branches[lid](query)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(query))
                intermediate_reference_points.append(new_reference_points)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.
            if self.use_weight_loss:
                intermediate_weights.append(weights)
        # if time_query is not None:
        #     if self.return_intermediate:
        #         return torch.stack(intermediate), torch.stack(intermediate_reference_points), time_query
        #     return query, reference_points, time_query
        # else:
        #     if self.return_intermediate:
        #         return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        #     return query, reference_points
        if self.return_intermediate:
            if self.use_weight_loss:
                return (
                    torch.stack(intermediate),
                    torch.stack(intermediate_reference_points),
                    time_query,
                    torch.stack(intermediate_weights),
                )
            else:
                return torch.stack(intermediate), torch.stack(intermediate_reference_points), time_query, None
        if self.use_weight_loss:
            return query, reference_points, time_query, weights.unsqueeze(0)
        else:
            return query, reference_points, time_query, []


class TimeDecoderLayer(BaseModule):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        no_tsa=False,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.cross_attn_image = MultiScaleDeformableAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_image = MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        query_time_pos: Optional[Tensor] = None,
        durations=None,
        **kwargs
    ):

        q = k = self.with_pos_embed(tgt, query_pos + query_time_pos)
        # q = k = self.with_pos_embed(tgt, query_time_pos)
        # Temporal Self attention
        tgt2, weights = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        t, b, c = tgt.shape
        durations = [t]
        # n_tokens, bs, f = memory.shape
        bs, n_tokens, f = memory.shape

        # extract the actual video length query
        clip_start = 0
        device = tgt.device
        tgt_cross = torch.zeros(1, bs, c).to(device)
        query_pos_cross = torch.zeros(1, bs, c).to(device)
        for i_b in range(b):
            tgt_clip = tgt[:, i_b, :]  # t x f
            query_pos_clip = query_pos[:, i_b, :]
            clip_length = durations[i_b]
            tgt_cross[0, clip_start : clip_start + clip_length] = tgt_clip[:clip_length]
            query_pos_cross[0, clip_start : clip_start + clip_length] = query_pos_clip[:clip_length]
            clip_start += clip_length

        assert clip_start == bs
        memory = memory.transpose(0, 1)  # txlxf -> lxtxf
        pos = pos.transpose(0, 1)  # txlxf -> lxtxf
        tgt2 = self.cross_attn_image(
            query=self.with_pos_embed(tgt_cross, query_pos_cross),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            **kwargs,
        )

        # reshape to the batched query
        clip_start = 0
        # tgt2_pad = torch.zeros(1, t * b, c).to(device)

        # for i_b in range(b):
        #     clip_length = durations[i_b]
        #     tgt2_pad[0, i_b * t : i_b * t + clip_length] = tgt2[0, clip_start : clip_start + clip_length]
        #     clip_start += clip_length

        # tgt2 = tgt2_pad
        tgt2 = tgt2.view(b, t, f).transpose(0, 1)  # 1x(b*t)xf -> bxtxf -> txbxf

        # tgt = tgt + self.dropout3(tgt2)
        # tgt = self.norm3(tgt)
        tgt = self.norm3(tgt2)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)

        return tgt, weights
        # return tgt

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
