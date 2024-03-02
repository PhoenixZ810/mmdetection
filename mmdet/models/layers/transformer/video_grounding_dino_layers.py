# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Optional, Tuple, Union

import torch
from mmengine.model import BaseModule
from torch import Tensor, nn

from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh

from mmdet.models.utils.vlfuse_helper import SingleScaleBiAttentionBlock
from mmdet.utils import ConfigType, OptConfigType
from .deformable_detr_layers import (
    DeformableDetrTransformerEncoderLayer,
)
from .detr_layers import DetrTransformerEncoderLayer
from mmengine.model import ModuleList
from .grounding_dino_layers import (
    GroundingDinoTransformerEncoder,
    GroundingDinoTransformerDecoder,
    GroundingDinoTransformerDecoderLayer,
)
from .utils import MLP, coordinate_to_encoding, inverse_sigmoid, get_text_sine_pos_embed
from .dino_layers import CdnQueryGenerator

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None


class VideoGroundingDinoTransformerEncoder(GroundingDinoTransformerEncoder):
    def __init__(self, time_attn_text_layer_cfg=None,time_attn_img_layer_cfg=None, **kwargs) -> None:
        self.time_attn_text_layer_cfg = time_attn_text_layer_cfg
        self.time_attn_img_layer_cfg = time_attn_img_layer_cfg
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        if self.time_attn_img_layer_cfg is not None:
            self.time_attn_img_layers = ModuleList(
                [DetrTransformerEncoderLayer(**self.time_attn_img_layer_cfg) for _ in range(self.num_layers)]
            )
        if self.time_attn_text_layer_cfg is not None:
            self.time_attn_text_layers = ModuleList(
                [DetrTransformerEncoderLayer(**self.time_attn_text_layer_cfg) for _ in range(self.num_layers)]
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
                if self.time_attn_text_layer_cfg is not None:
                    memory_text = self.time_attn_text_layers[layer_id](
                        query=memory_text.transpose(0, 1),
                        query_pos=time_embed.transpose(0, 1),
                        key_padding_mask=None,
                    ).transpose(0, 1)
            if self.time_attn_img_layer_cfg is not None:
                output = self.time_attn_img_layers[layer_id](
                    query=output.transpose(0, 1),
                    query_pos=time_embed.transpose(0, 1),
                    key_padding_mask=None,
                ).transpose(0, 1)
            output = layer(
                query=output,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask,
            )
        return output, memory_text


class VideoGroundingDinoTransformerDecoder(GroundingDinoTransformerDecoder):
    """Transformer decoder of VideoDINO."""

    def __init__(self, use_weight_loss=False, * args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_weight_loss = use_weight_loss

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList(
            [VideoGroundingDinoTransformerDecoderLayer(**self.layer_cfg) for _ in range(self.num_layers)]
        )
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
        time_embed: Optional[Tensor] = None,
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
            query, weights = layer(
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

        if self.return_intermediate:
            if self.use_weight_loss:
                return torch.stack(intermediate), torch.stack(intermediate_reference_points), torch.stack(intermediate_weights)
            else:
                return torch.stack(intermediate), torch.stack(intermediate_reference_points), None
        if self.use_weight_loss:
            return query, reference_points, weights.unsqueeze(0)
        else:
            return query, reference_points, None


class VideoGroundingDinoTransformerDecoderLayer(GroundingDinoTransformerDecoderLayer):
    def __init__(self, *args, use_self_attn=True, time_query_type='tq', time_attn_cfg=None, **kwargs):
        self.time_attn_cfg = time_attn_cfg
        self.use_self_attn = use_self_attn
        self.time_query_type = time_query_type
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        if self.use_self_attn:
            self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        if self.time_attn_cfg is not None:
            if 'batch_first' not in self.time_attn_cfg:
                self.time_attn_cfg['batch_first'] = True
            self.time_attn = MultiheadAttention(**self.time_attn_cfg)
        self.cross_attn_text = MultiheadAttention(**self.cross_attn_text_cfg)
        self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.embed_dims = self.cross_attn_cfg['embed_dims']
        self.ffn = FFN(**self.ffn_cfg)
        if self.time_attn_cfg is not None and self.use_self_attn:
            norms_list = [build_norm_layer(self.norm_cfg, self.embed_dims)[1] for _ in range(5)]
        else:
            norms_list = [build_norm_layer(self.norm_cfg, self.embed_dims)[1] for _ in range(4)]
        self.norms = ModuleList(norms_list)

    def forward(
        self,
        query: Tensor,
        key: Tensor = None,
        value: Tensor = None,
        query_pos: Tensor = None,
        key_pos: Tensor = None,
        self_attn_mask: Tensor = None,
        cross_attn_mask: Tensor = None,
        key_padding_mask: Tensor = None,
        memory_text: Tensor = None,
        text_attention_mask: Tensor = None,
        time_embed: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Implements decoder layer in Grounding DINO transformer.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.
            memory_text (Tensor): Memory text. It has shape (bs, len_text,
                text_embed_dims).
            text_attention_mask (Tensor): Text token mask. It has shape (bs,
                len_text).

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """

        if time_embed is not None and self.time_attn_cfg is not None:
            # temporal_self_attention
            if self.time_query_type == 'tq':
                query,weights = self.time_attn(
                    query=query.transpose(0, 1),
                    key=query.transpose(0, 1),
                    value=query.transpose(0, 1),
                    query_pos=query_pos.transpose(0, 1) + time_embed.transpose(0, 1),
                    key_pos=query_pos.transpose(0, 1) + time_embed.transpose(0, 1),
                    attn_mask=self_attn_mask,
                    return_weight=True,
                    **kwargs,
                )
                query = query.transpose(0, 1)
            elif self.time_query_type == 'q':
                query, weights = self.time_attn(
                    query=query.transpose(0, 1),
                    key=query.transpose(0, 1),
                    value=query.transpose(0, 1),
                    query_pos=query_pos.transpose(0, 1),
                    key_pos=query_pos.transpose(0, 1),
                    attn_mask=self_attn_mask,
                    return_weight=True,
                    **kwargs,
                )
                query = query.transpose(0, 1)
            elif self.time_query_type == 't':
                query, weights = self.time_attn(
                    query=query.transpose(0, 1),
                    key=query.transpose(0, 1),
                    value=query.transpose(0, 1),
                    query_pos=time_embed.transpose(0, 1),
                    key_pos=time_embed.transpose(0, 1),
                    attn_mask=self_attn_mask,
                    return_weight=True,
                    **kwargs,
                )
                query = query.transpose(0, 1)
            query = self.norms[0](query)

        if self.use_self_attn:
            # image self attention
            query = self.self_attn(
                query=query,
                key=query,
                value=query,
                query_pos=query_pos,
                key_pos=query_pos,
                attn_mask=self_attn_mask,
                **kwargs,
            )
            query = self.norms[1](query) if len(self.norms) == 5 else self.norms[0](query)

        # cross attention between query and text
        query = self.cross_attn_text(
            query=query,
            query_pos=query_pos,
            key=memory_text,
            value=memory_text,
            key_padding_mask=text_attention_mask,
        )
        query = self.norms[2](query) if len(self.norms) == 5 else self.norms[1](query)
        # cross attention between query and image
        query = self.cross_attn(
            query=query,
            key=key, # None
            value=value, # image memory
            query_pos=query_pos,
            key_pos=key_pos, # None
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs,
        )
        query = self.norms[3](query) if len(self.norms) == 5 else self.norms[2](query)
        query = self.ffn(query)
        query = self.norms[4](query) if len(self.norms) == 5 else self.norms[3](query)

        return query, weights
