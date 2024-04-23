# Copyright (c) OpenMMLab. All rights reserved.
import copy
import re
import warnings
from typing import Dict, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType
from ..layers import SinePositionalEncoding
from ..layers.transformer.video_grounding_dino_layers import (
    VideoGroundingDinoTransformerEncoder,
    VideoGroundingDinoTransformerDecoder,
)
from mmengine import MessageHub
from .dino import DINO
from .glip import create_positive_map, create_positive_map_label_to_token, run_ner
from .grounding_dino import GroundingDINO


@MODELS.register_module()
class VideoGroundingDINO(GroundingDINO):

    def __init__(
        self,
        train_cfg: Optional[ConfigType] = None,
        use_time_embed=True,
        use_dn=False,
        max_time_pos_frames=200,
        freeze_backbone=False,
        freeze_language_model=False,
        freeze_encoder=False,
        img_encoder_from_cache=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if train_cfg:
            if self.bbox_head.use_cls_loss:
                assert 'assigner' in train_cfg, 'assigner should be provided ' 'when train_cfg is set.'
                assigner = train_cfg['assigner']
                self.assigner = TASK_UTILS.build(assigner)
            else:
                assert 'assigner_gt' in train_cfg, 'assigner_gt should be provided ' 'when train_cfg is set.'
                assigner_gt = train_cfg['assigner_gt']
                self.assigner_gt = TASK_UTILS.build(assigner_gt)
                assert 'assigner_sted' in train_cfg, 'assigner_sted should be provided ' 'when train_cfg is set.'
                assigner_sted = train_cfg['assigner_sted']
                self.assigner_sted = TASK_UTILS.build(assigner_sted)

            if train_cfg.get('sampler', None) is not None:
                raise RuntimeError('DETR do not build sampler.')
        self.dn_query_generator = None

        self.use_dn = use_dn
        self.use_time_embed = use_time_embed
        self.max_frames = max_time_pos_frames
        self.freeze_backbone = freeze_backbone
        self.freeze_language_model = freeze_language_model
        self.freeze_encoder = freeze_encoder
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
        if self.freeze_language_model:
            self.language_model.requires_grad_(False)
        if self.freeze_encoder:
            self.encoder.requires_grad_(False)
        if self.use_time_embed:
            self.time_embed = TimeEmbeddingSine(max_len=self.max_frames, d_model=self.embed_dims)
        self.img_encoder_from_cache = img_encoder_from_cache
        # print(self)

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
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask,
        )
        return encoder_outputs_dict

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
        if self.bbox_head.use_cls_loss:
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
            if self.bbox_head.use_enc_sted and self.num_queries > 1:
                top1_indices = torch.topk(enc_outputs_class.max(-1)[0], k=1, dim=1)[1]

            topk_score = torch.gather(
                enc_outputs_class, 1, topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features)
            )
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 4)
            )
            topk_coords = topk_coords_unact.sigmoid()
            topk_coords_unact = topk_coords_unact.detach()

            if self.bbox_head.use_enc_sted:
                enc_outputs_sted = self.bbox_head.sted_branches[-1](output_memory)
                if self.num_queries == 1:
                    topk_sted = torch.gather(enc_outputs_sted, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 2))
                else:
                    topk_sted = torch.gather(enc_outputs_sted, 1, top1_indices.unsqueeze(-1).repeat(1, 1, 2))
        else:
            assert self.bbox_head.use_enc_sted==True
            enc_outputs_sted = self.bbox_head.sted_branches[-1](output_memory)
            enc_outputs_coord_unact = (
                self.bbox_head.reg_branches[self.decoder.num_layers](output_memory) + output_proposals
            )
            enc_outputs_sted_sum=enc_outputs_sted.sum(2)
            topk_indices = torch.topk(enc_outputs_sted_sum, k=self.num_queries, dim=1)[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 4)
            )
            topk_sted = torch.gather(enc_outputs_sted, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 2))
            topk_coords = topk_coords_unact.sigmoid()
            topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)

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
        if self.bbox_head.use_cls_loss:
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
        else:
            if self.bbox_head.use_enc_sted:
                head_inputs_dict = (
                    dict(
                        enc_outputs_class=None,
                        enc_outputs_coord=topk_coords,
                        dn_meta=dn_meta,
                        enc_outputs_sted=topk_sted,
                    )
                    if self.training
                    else dict()
                )
            else:
                head_inputs_dict = (
                    dict(enc_outputs_class=None, enc_outputs_coord=topk_coords, dn_meta=dn_meta)
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

        inter_states, references, weights = self.decoder(
            query=query,
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

        # if len(query) == self.num_queries:
        #     # NOTE: This is to make sure label_embeding can be involved to
        #     # produce loss even if there is no denoising query (no ground truth
        #     # target in this GPU), otherwise, this will raise runtime error in
        #     # distributed training.
        #     inter_states[0] += self.dn_query_generator.label_embedding.weight[0, 0] * 0.0

        decoder_outputs_dict = dict(hidden_states=inter_states, references=list(references), weights = weights)
        return decoder_outputs_dict

    def get_positive_map(self, tokenized, tokens_positive):
        positive_map = create_positive_map(
            tokenized,
            tokens_positive,
            max_num_entities=256,
        )
        positive_map_label_to_token = create_positive_map_label_to_token(positive_map, plus=1)
        return positive_map_label_to_token, positive_map

    def loss(
        self, batch_inputs: Tensor, batch_data_samples: SampleList, img_encoder_from_cache=False
    ) -> Union[dict, list]:
        text_prompts = [data_samples.text for data_samples in batch_data_samples]

        gt_labels = [data_samples.gt_instances.labels for data_samples in batch_data_samples]

        if 'tokens_positive' in batch_data_samples[0]:
            tokens_positive = [data_samples.tokens_positive for data_samples in batch_data_samples]
            positive_maps = []
            for token_positive, text_prompt, gt_label in zip(tokens_positive, text_prompts, gt_labels):
                tokenized = self.language_model.tokenizer(
                    [text_prompt],
                    padding='max_length' if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt',
                )
                new_tokens_positive = [token_positive[label.item()] for label in gt_label]
                _, positive_map = self.get_positive_map(tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
            new_text_prompts = text_prompts
        else:
            new_text_prompts = []
            positive_maps = []
            if len(set(text_prompts)) == 1:
                # All the text prompts are the same,
                # so there is no need to calculate them multiple times.
                tokenized, caption_string, tokens_positive, _ = self.get_tokens_and_prompts(
                    text_prompts[0], True
                )
                new_text_prompts = [caption_string] * len(batch_inputs)
                for gt_label in gt_labels:
                    new_tokens_positive = [tokens_positive[0] for label in gt_label]
                    _, positive_map = self.get_positive_map(tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
            else:
                for text_prompt, gt_label in zip(text_prompts, gt_labels):
                    tokenized, caption_string, tokens_positive, _ = self.get_tokens_and_prompts(
                        text_prompt, True
                    )
                    new_tokens_positive = [tokens_positive[label] for label in gt_label]
                    _, positive_map = self.get_positive_map(tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)

        text_dict = self.language_model(new_text_prompts)
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])

        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = text_token_mask.unsqueeze(0).repeat(
                len(positive_map), 1
            )
        if self.img_encoder_from_cache:
            message_hub = MessageHub.get_current_instance()
            epoch = message_hub.get_info('epoch')
        visual_features = self.extract_feat(batch_inputs)
        # torch.save(visual_features, f'data_cache/hcstvg_50_224/{epoch}_{batch_data_samples[0].video_id}.pt')
        # losses = {'loss':visual_features[0][0][0][0]*0}
        # return losses
        head_inputs_dict = self.forward_transformer(visual_features, text_dict, batch_data_samples)

        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples, use_dn=self.use_dn
        )
        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))

        if 'custom_entities' in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(set(text_prompts)) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities, enhanced_text_prompts[0], tokens_positives[0]
                )
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompt, custom_entities, enhanced_text_prompt, tokens_positive
                )
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives
                )
            ]
        token_positive_maps, text_prompts, _, entities = zip(*_positive_maps_and_prompts)

        # image feature extraction
        visual_feats = self.extract_feat(batch_inputs)

        if isinstance(text_prompts[0], list):
            # chunked text prompts, only bs=1 is supported
            assert len(batch_inputs) == 1
            count = 0
            results_list = []

            entities = [[item for lst in entities[0] for item in lst]]

            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]
                text_dict = self.language_model(text_prompts_once)
                # text feature map layer
                if self.text_feat_map is not None:
                    text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])

                batch_data_samples[0].token_positive_map = token_positive_maps_once

                head_inputs_dict = self.forward_transformer(
                    copy.deepcopy(visual_feats), text_dict, batch_data_samples
                )
                pred_instances = self.bbox_head.predict(
                    **head_inputs_dict, rescale=rescale, batch_data_samples=batch_data_samples
                )[0]

                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
            results_list = [results_list[0].cat(results_list)]
            is_rec_tasks = [False] * len(results_list)
        else:
            # extract text feats
            text_dict = self.language_model(list(text_prompts))
            # text feature map layer
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])

            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]

            head_inputs_dict = self.forward_transformer(visual_feats, text_dict, batch_data_samples)
            results_list = self.bbox_head.predict(
                **head_inputs_dict, rescale=False, batch_data_samples=batch_data_samples
            )

        for data_sample, pred_instances, entity, is_rec_task in zip(
            batch_data_samples, results_list, entities, is_rec_tasks
        ):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.'
                        )
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        return batch_data_samples


class TimeEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=200, d_model=512):
        super().__init__()
        self.time_embed = nn.Embedding(num_pos_feats, d_model)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.time_embed.weight)

    def forward(self, ln):
        return self.time_embed.weight[:ln].unsqueeze(1)


class TimeEmbeddingSine(nn.Module):
    """
    Same as below for temporal dimension
    """

    def __init__(self, max_len=200, d_model=512):
        super().__init__()
        self.max_len = max_len
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        te = torch.zeros(max_len, 1, d_model)
        te[:, 0, 0::2] = torch.sin(position * div_term)
        te[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("te", te)

    def forward(self, ln):
        pos_t = self.te[:ln]
        return pos_t
