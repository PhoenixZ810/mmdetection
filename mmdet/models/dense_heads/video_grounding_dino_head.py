import copy
import math
from typing import Dict, List, Optional, Tuple, Union
import sys

import torch
import torch.nn as nn
from mmcv.cnn import Linear
from mmengine.structures import InstanceData
from torch import Tensor
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.utils import InstanceList, reduce_mean, OptInstanceList
from ..layers import inverse_sigmoid
from .atss_vlfusion_head import convert_grounding_to_cls_scores
from .grounding_dino_head import GroundingDINOHead, ContrastiveEmbed
from ..layers import MLP
from .deformable_detr_head import DeformableDETRHead
from ..utils import multi_apply
from mmdet.models.losses import QualityFocalLoss

@MODELS.register_module()
class VideoGroundingHead(GroundingDINOHead):
    def __init__(
        self,
        use_sted=False,
        use_aux_time=False,
        use_enc_sted=False,
        sigma=1,
        sted_loss_weight=5.0,
        enc_sted_loss_weight=None,
        time_only=False,
        exclude_cls=False,
        exclude_box = False,
        **kwargs,
    ):
        self.use_sted = use_sted
        self.use_aux_time = use_aux_time
        self.use_enc_sted = use_enc_sted
        self.sigma = sigma
        self.sted_loss_weight = sted_loss_weight
        if enc_sted_loss_weight is not None:
            self.enc_sted_loss_weight = enc_sted_loss_weight
        else:
            self.enc_sted_loss_weight = sted_loss_weight
        self.time_only = time_only
        self.exclude_cls = exclude_cls
        self.exclude_box = exclude_box

        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        fc_cls = ContrastiveEmbed(**self.contrastive_cfg)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        # NOTE: due to the fc_cls is a contrastive embedding and don't
        # have any trainable parameters,we do not need to copy it.
        if self.share_pred_layer:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(self.num_pred_layer)])
        else:
            self.cls_branches = nn.ModuleList([copy.deepcopy(fc_cls) for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList([copy.deepcopy(reg_branch) for _ in range(self.num_pred_layer)])

        '''branch for start and end time prediction'''
        sted_branch = MLP(self.embed_dims, self.embed_dims, 2, 2)
        if self.use_sted and self.use_aux_time:
            if not self.use_enc_sted:
                self.sted_branches = nn.ModuleList(
                    [copy.deepcopy(sted_branch) for _ in range(self.num_pred_layer-1)]
                )
            else:
                self.sted_branches = nn.ModuleList(
                    [copy.deepcopy(sted_branch) for _ in range(self.num_pred_layer)]
                )
        elif self.use_sted and not self.use_aux_time:
            if self.use_enc_sted:
                self.sted_branches = nn.ModuleList([copy.deepcopy(sted_branch), copy.deepcopy(sted_branch)])
            else:
                self.sted_branches = nn.ModuleList([copy.deepcopy(sted_branch)])

    def forward(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        memory_text: Tensor,
        text_token_mask: Tensor,
        use_dn=False,
    ) -> Tuple[Tensor]:
        """Forward function.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries, dim).
            references (List[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.
            memory_text (Tensor): Memory text. It has shape (bs, len_text,
                text_embed_dims).
            text_token_mask (Tensor): Text token mask. It has shape (bs,
                len_text).

        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - all_layers_outputs_classes (Tensor): Outputs from the
              classification head, has shape (num_decoder_layers, bs,
              num_queries, cls_out_channels).
            - all_layers_outputs_coords (Tensor): Sigmoid outputs from the
              regression head with normalized coordinate format (cx, cy, w,
              h), has shape (num_decoder_layers, bs, num_queries, 4) with the
              last dimension arranged as (cx, cy, w, h).
        """
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []
        all_layers_outputs_sted = []

        # if self.use_sted:
        #     if self.use_enc_sted:
        #         outputs_sted = self.sted_branch[1](hidden_states[-1])
        #     else:
        #         outputs_sted = self.sted_branch(hidden_states[-1])
        # else:
        #     outputs_sted = None

        for layer_id in range(hidden_states.shape[0]):
            if self.use_sted and self.use_aux_time:
                output_sted = self.sted_branches[layer_id](hidden_states[layer_id])
                all_layers_outputs_sted.append(output_sted)
            elif self.use_sted and not self.use_aux_time:
                if layer_id == hidden_states.shape[0] - 1:
                    output_sted = self.sted_branches[0](hidden_states[layer_id])
                    all_layers_outputs_sted.append(output_sted)
            else:
                outputs_sted = None
            reference = inverse_sigmoid(references[layer_id])
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state, memory_text, text_token_mask)
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            if reference.shape[-1] == 4:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `True`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `True`.
                tmp_reg_preds += reference
            else:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `False`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `False`.
                assert reference.shape[-1] == 2
                tmp_reg_preds[..., :2] += reference
            outputs_coord = tmp_reg_preds.sigmoid()
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)

        if self.use_sted:
            outputs_sted = torch.stack(all_layers_outputs_sted)
        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)

        return all_layers_outputs_classes, all_layers_outputs_coords, outputs_sted
        # tup = ()
        # return (outputs_sted,)

    def loss(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        memory_text: Tensor,
        text_token_mask: Tensor,
        enc_outputs_class: Tensor,
        enc_outputs_coord: Tensor,
        batch_data_samples: SampleList,
        use_dn: bool,
        dn_meta: Dict[str, int],
        weights: Optional[Tensor] = None,
        enc_outputs_sted=None,
    ) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries_total,
                dim), where `num_queries_total` is the sum of
                `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries_total, 4) and each `inter_reference` has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            memory_text (Tensor): Memory text. It has shape (bs, len_text,
                text_embed_dims).
            enc_outputs_class (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_outputs_coord (Tensor): The proposal generate from the
                encode feature map, has shape (bs, num_feat_points, 4) with the
                last dimension arranged as (cx, cy, w, h).
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references, memory_text, text_token_mask)
        self.text_masks = text_token_mask

        durations = batch_data_samples[0].durations
        inter_idx = [batch_data_samples[duration - 1].inter_idx for duration in durations]

        # set time_mask of valid video frames for each batchsize(duration)
        if self.use_sted:
            device = hidden_states.device
            time_mask = torch.zeros(1, outs[2].shape[1]).bool().to(device)
            # time_mask = torch.zeros(1, outs[0].shape[0]).bool().to(device)
            for i_dur, duration in enumerate(durations):
                time_mask[i_dur, :duration] = True
        else:
            time_mask = None
        if weights is not None:
            weights = weights[-1]
        loss_inputs = outs + (
            weights,
            enc_outputs_class,
            enc_outputs_coord,
            enc_outputs_sted,
            batch_gt_instances,
            batch_img_metas,
            use_dn,
            dn_meta,
            time_mask,
            durations,
            inter_idx,
        )
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        outputs_sted,
        weights: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        enc_outputs_sted: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        use_dn: bool,
        dn_meta: Dict[str, int],
        time_mask=None,
        durations=None,
        inter_idx=None,
        batch_gt_instances_ignore: OptInstanceList = None,
    ) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels), where
                `num_queries_total` is the sum of `num_denoising_queries`
                and `num_matching_queries`.
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries_total, 4).
            outputs_sted(optional): Start and end time prediction outputs of final decoder layers, with shape (t, num_queries, 2).
            enc_cls_scores (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_bbox_preds (Tensor): The proposal generate from the encode
                feature map, has shape (bs, num_feat_points, 4) with the last
                dimension arranged as (cx, cy, w, h).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
                group collation, including 'num_denoising_queries' and
                'num_denoising_groups'. It will be used for split outputs of
                denoising and matching parts and loss calculation.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # extract denoising and matching part of outputs
        if use_dn:
            (
                all_layers_matching_cls_scores,
                all_layers_matching_bbox_preds,
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds,
            ) = self.split_outputs(all_layers_cls_scores, all_layers_bbox_preds, dn_meta)
        else:
            all_layers_matching_cls_scores = all_layers_cls_scores
            all_layers_matching_bbox_preds = all_layers_bbox_preds
            # print('original_shape', original_all_layers_matching_cls_scores.shape)
            all_layers_denoising_cls_scores = None
            all_layers_denoising_bbox_preds = None

        max_duration = max(durations)
        device = all_layers_matching_cls_scores.device

        keep_list = []
        for i_dur, (duration, inter) in enumerate(zip(durations, inter_idx)):
            keep_list.extend(
                [
                    elt
                    for elt in range(
                        i_dur * max_duration + inter[0],
                        (i_dur * max_duration) + inter[1] + 1,
                    )
                ]
            )
        keep = torch.tensor(keep_list).long().to(device)

        # original_enc_cls_scores = enc_cls_scores
        # original_enc_bbox_preds = enc_bbox_preds

        if use_dn:
            all_layers_denoising_cls_scores = all_layers_denoising_cls_scores[:, keep]
            all_layers_denoising_bbox_preds = all_layers_denoising_bbox_preds[:, keep]
        # loss_dict = super(DeformableDETRHead, self).loss_by_feat(
        #     all_layers_matching_cls_scores,
        #     all_layers_matching_bbox_preds,
        #     batch_gt_instances,
        #     batch_img_metas,
        #     batch_gt_instances_ignore,
        # )
        assert batch_gt_instances_ignore is None, (
            f'{self.__class__.__name__} only supports ' 'for batch_gt_instances_ignore setting to None.'
        )

        losses_cls, losses_bbox, losses_iou, pos_inds_list = multi_apply(
            self.loss_by_feat_single,
            all_layers_matching_cls_scores,
            all_layers_matching_bbox_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            keep=keep,
            use_dn=use_dn
        )

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1

        # NOTE DETRHead.loss_by_feat but not DeformableDETRHead.loss_by_feat
        # is called, because the encoder loss calculations are different
        # between DINO and DeformableDETR.

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            # NOTE The enc_loss calculation of the DINO is
            # different from that of Deformable DETR.
            enc_loss_cls, enc_losses_bbox, enc_losses_iou, enc_pos_inds_list = self.loss_by_feat_single(
                enc_cls_scores,
                enc_bbox_preds,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
                keep=keep,
            )
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou

        if self.use_sted:
            num_boxes = len(keep)
            inter_idx = [batch_img_metas[0]['inter_idx']]
            if inter_idx is not None and time_mask is not None:
                # set frames with box for each batchsize(duration) as positive
                # construct a map such that positive_map[k, i] = True iff num_frame i lies inside the annotated moment k
                positive_map = torch.zeros(time_mask.shape, dtype=torch.bool)
                for k, idx in enumerate(inter_idx):
                    if idx[0] < 0:  # empty intersection
                        continue
                    positive_map[k][idx[0] : idx[1] + 1].fill_(True)

                positive_map = positive_map.to(time_mask.device)
            elif time_mask is None:
                positive_map = None
            if outputs_sted.shape[2] != 1:
                for i in range(outputs_sted.shape[1]):
                    outputs_sted[:,i]=outputs_sted[:,i,pos_inds_list[-1][i]]
            loss_dict.update(self.loss_sted(outputs_sted, num_boxes, inter_idx, positive_map, time_mask))
            if self.use_enc_sted:
                if enc_outputs_sted.shape[1] !=1:
                    for i in range(enc_outputs_sted.shape[0]):
                        enc_outputs_sted[i]=enc_outputs_sted[i,enc_pos_inds_list[i]]
                loss_enc=(self.loss_sted(enc_outputs_sted, num_boxes, inter_idx, positive_map, time_mask, enc_flag=True))
                loss_dict['enc_loss_sted'] = loss_enc['loss_sted']

        if weights is not None:
            loss_dict.update(self.loss_guided_attn(weights, positive_map, time_mask))

        if all_layers_denoising_cls_scores is not None and use_dn:
            # calculate denoising loss from all decoder layers
            dn_losses_cls, dn_losses_bbox, dn_losses_iou = self.loss_dn(
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
                dn_meta=dn_meta,
            )
            # collate denoising loss
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
            for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i) in enumerate(
                zip(dn_losses_cls[:-1], dn_losses_bbox[:-1], dn_losses_iou[:-1])
            ):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i

        '''exclude cls loss'''
        if self.exclude_cls:
            for ls in loss_dict:
                if 'cls' in ls:
                    loss_dict[ls] = loss_dict[ls] * 0
        '''only time loss'''
        if self.time_only:
            for ls in loss_dict:
                if ls != 'loss_sted':
                    loss_dict[ls] = loss_dict[ls] * 0
        '''exclude box loss'''
        if self.exclude_box:
            for ls in loss_dict:
                if 'sted' not in ls and 'cls' not in ls:
                    loss_dict[ls] = loss_dict[ls] * 0
        return loss_dict

    def loss_by_feat_single(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        keep:Tensor,
        use_dn: bool = False,
    ) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images, has shape (bs, num_queries, cls_out_channels).
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape (bs, num_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        if len(keep) != cls_scores.shape[0]:
            # 若有帧没有box，需要去掉这些帧
            sample_flag = True
            sampled_cls_scores = torch.index_select(cls_scores, 0, keep)
            sampled_bbox_preds = torch.index_select(bbox_preds, 0, keep)
            # self.text_masks = torch.index_select(self.text_masks, 0, keep)
            self.keep = keep

            # print('batch_img_metas=', len(batch_img_metas), 'keep=', len(keep))
            img_metas = [batch_img_metas[i] for i in keep]
            gt_instances = [batch_gt_instances[i] for i in keep]
            batch_img_metas = img_metas
            batch_gt_instances = gt_instances
        else:
            sample_flag = False
            self.keep = keep
            sampled_cls_scores = cls_scores
            sampled_bbox_preds = bbox_preds

        num_imgs = sampled_cls_scores.size(0)
        cls_scores_list = [sampled_cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [sampled_bbox_preds[i] for i in range(num_imgs)]
        with torch.no_grad():
            cls_reg_targets = self.get_targets(
                cls_scores_list, bbox_preds_list, batch_gt_instances, batch_img_metas
            )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
            pos_inds_list,
            neg_inds_list
        ) = cls_reg_targets
        if sample_flag:
            new_labels_list = [
                torch.zeros(labels_list[0].shape).to(cls_scores.device) for i in range(cls_scores.shape[0])
            ]
            for i in range(len(self.keep)):
                new_labels_list[self.keep[i]] = labels_list[i]
            labels_list = new_labels_list
            new_label_weights_list = [label_weights_list[0] for i in range(cls_scores.shape[0])]
            label_weights_list = new_label_weights_list

        labels = torch.stack(labels_list, 0)
        label_weights = torch.stack(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # ===== this change =====
        # Loss is not computed for the padded regions of the text.
        assert self.text_masks.dim() == 2
        text_masks = self.text_masks.new_zeros((self.text_masks.size(0), self.max_text_len))
        text_masks[:, : self.text_masks.size(1)] = self.text_masks
        text_mask = (text_masks > 0).unsqueeze(1)
        text_mask = text_mask.repeat(1, cls_scores.size(1), 1)
        cls_scores = torch.masked_select(cls_scores, text_mask).contiguous()

        labels = torch.masked_select(labels, text_mask)
        label_weights = label_weights[..., None].repeat(1, 1, text_mask.size(-1))
        label_weights = torch.masked_select(label_weights, text_mask)

        # classification loss
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if isinstance(self.loss_cls, QualityFocalLoss):
            raise NotImplementedError('QualityFocalLoss for GroundingDINOHead is not supported yet.')
        else:
            loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, sampled_bbox_preds):
            (
                img_h,
                img_w,
            ) = img_meta['img_shape']
            factor = (
                bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0).repeat(bbox_pred.size(0), 1)
            )
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = sampled_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou, pos_inds_list

    def loss_sted(self, outputs_sted, num_boxes, inter_idx, positive_map, time_mask=None, enc_flag=False):
        """Compute the losses related to the start & end prediction, a KL divergence loss
        targets dicts must contain the key "pred_sted" containing a tensor of logits of dim [T, 2]
        """
        # assert "pred_sted" in outputs
        losses = {}
        if self.use_aux_time and not enc_flag:
            num = outputs_sted.shape[0]
        else:
            num = 1
        for i in range(num):
            if enc_flag:
                sted = outputs_sted.transpose(0, 1)  # [1, T, 2]
            else:
                sted = outputs_sted[i].transpose(0, 1)  # [1, T, 2]
            target_start = torch.tensor([x[0] for x in inter_idx], dtype=torch.long).to(sted.device)
            target_end = torch.tensor([x[1] for x in inter_idx], dtype=torch.long).to(sted.device)
            sted = sted.masked_fill(
                ~time_mask[:, :, None], -1e32
            )  # put very low probability on the padded positions before softmax
            eps = 1e-6  # avoid log(0) and division by 0

            sigma = self.sigma

            start_distrib = (
                -((torch.arange(sted.shape[1])[None, :].to(sted.device) - target_start[:, None]) ** 2)
                / (2 * sigma**2)
            ).exp()  # gaussian target, ground-truth time distribution
            start_distrib = F.normalize(start_distrib + eps, p=1, dim=1)
            pred_start_prob = (sted[:, :, 0]).softmax(1)  # 预测每一帧是开始帧的概率
            loss_start = pred_start_prob * ((pred_start_prob + eps) / start_distrib).log()  # KL div loss
            loss_start = loss_start * time_mask  # not count padded values in the loss

            end_distrib = (
                -((torch.arange(sted.shape[1])[None, :].to(sted.device) - target_end[:, None]) ** 2)
                / (2 * sigma**2)
            ).exp()  # gaussian target
            end_distrib = F.normalize(end_distrib + eps, p=1, dim=1)
            pred_end_prob = (sted[:, :, 1]).softmax(1)
            loss_end = pred_end_prob * ((pred_end_prob + eps) / end_distrib).log()  # KL div loss
            loss_end = loss_end * time_mask  # do not count padded values in the loss

            loss_sted = loss_start + loss_end
            if enc_flag:
                losses["loss_sted"] = loss_sted.mean() * self.enc_sted_loss_weight
            elif i != outputs_sted.shape[0] - 1:
                losses[f"loss_sted{i}"] = loss_sted.mean() * self.sted_loss_weight
            else:
                losses["loss_sted"] = loss_sted.mean() * self.sted_loss_weight

        return losses

    def loss_guided_attn(self, weights, positive_map, time_mask=None):
        """Compute guided attention loss
        targets "weights" contains a tensor of attention matrices of dim [B, T, T]
        """
        # weights = outputs["weights"]  # BxTxT

        positive_map = positive_map + (~time_mask)  # the padded positions also have to be taken out
        eps = 1e-6  # avoid log(0) and division by 0

        loss = -(1 - weights + eps).log()
        loss = loss.masked_fill(positive_map[:, :, None], 0)
        nb_neg = (~positive_map).sum(1) + eps
        loss = loss.sum(2) / nb_neg[:, None]  # sum on the column
        loss = loss.sum(1)  # mean on the line normalized by the number of negatives
        loss = loss.mean()  # mean on the batch

        losses = {"loss_guided_attn": loss}
        return losses

    def predict(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: SampleList,
        weights: Optional[Tensor] = None,
        rescale: bool = True,
    ) -> InstanceList:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, num_queries, bs, dim).
            references (List[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.
            memory_text (Tensor): Memory text. It has shape (bs, len_text,
                text_embed_dims).
            text_token_mask (Tensor): Text token mask. It has shape (bs,
                len_text).
            batch_data_samples (SampleList): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Defaults to `True`.

        Returns:
            InstanceList: Detection results of each image
                after the post process.
        """
        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]
        batch_token_positive_maps = [data_samples.token_positive_map for data_samples in batch_data_samples]

        outs = self(hidden_states, references, memory_text, text_token_mask)

        predictions = self.predict_by_feat(
            *outs,
            batch_img_metas=batch_img_metas,
            batch_token_positive_maps=batch_token_positive_maps,
            rescale=rescale,
        )
        return predictions

    def predict_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        outputs_sted,
        batch_img_metas: List[Dict],
        batch_token_positive_maps: Optional[List[dict]] = None,
        rescale: bool = False,
    ) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            all_layers_cls_scores (Tensor):  Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs, num_queries,
                cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and shape (num_decoder_layers, bs, num_queries,
                4) with the last dimension arranged as (cx, cy, w, h).
            batch_img_metas (List[Dict]): _description_
            batch_token_positive_maps (list[dict], Optional): Batch token
                positive map. Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        cls_scores = all_layers_cls_scores[-1]
        bbox_preds = all_layers_bbox_preds[-1]
        model_outputs_sted = outputs_sted[-1]
        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            outputs_sted = model_outputs_sted[img_id]
            img_meta = batch_img_metas[img_id]
            token_positive_maps = batch_token_positive_maps[img_id]
            results = self._predict_by_feat_single(
                cls_score, bbox_pred, outputs_sted, token_positive_maps, img_meta, rescale
            )
            result_list.append(results)
        return result_list
        # if self.use_sted:
        #     num_boxes = len(keep)
        #     inter_idx = [batch_img_metas[0]['inter_idx']]
        #     if inter_idx is not None and time_mask is not None:
        #         # set frames with box for each batchsize(duration) as positive
        #         # construct a map such that positive_map[k, i] = True iff num_frame i lies inside the annotated moment k
        #         positive_map = torch.zeros(time_mask.shape, dtype=torch.bool)
        #         for k, idx in enumerate(inter_idx):
        #             if idx[0] < 0:  # empty intersection
        #                 continue
        #             positive_map[k][idx[0] : idx[1] + 1].fill_(True)

        #         positive_map = positive_map.to(time_mask.device)
        #     elif time_mask is None:
        #         positive_map = None
        #     loss_dict.update(self.loss_sted(outputs_sted, num_boxes, inter_idx, positive_map, time_mask))

    def _predict_by_feat_single(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        outputs_sted: Tensor,
        token_positive_maps: dict,
        img_meta: dict,
        rescale: bool = True,
    ) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_queries, 4].
            token_positive_maps (dict): Token positive map.
            img_meta (dict): Image meta info.
            rescale (bool, optional): If True, return boxes in original image
                space. Default True.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_score) == len(bbox_pred)  # num_queries
        max_per_img = self.test_cfg.get('max_per_img', len(cls_score))
        img_shape = img_meta['img_shape']

        if token_positive_maps is not None:
            cls_score = convert_grounding_to_cls_scores(
                logits=cls_score.sigmoid()[None], positive_maps=[token_positive_maps]
            )[0]
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            num_classes = cls_score.shape[-1]
            det_labels = indexes % num_classes
            bbox_index = indexes // num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            cls_score = cls_score.sigmoid()
            scores, _ = cls_score.max(-1)
            scores, indexes = scores.topk(max_per_img)
            bbox_pred = bbox_pred[indexes]
            det_labels = scores.new_zeros(scores.shape, dtype=torch.long)

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            assert img_meta.get('scale_factor') is not None
            det_bboxes /= det_bboxes.new_tensor(img_meta['scale_factor']).repeat((1, 2))
        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels
        results.sted = outputs_sted
        return results

    def get_targets(
        self,
        cls_scores_list: List[Tensor],
        bbox_preds_list: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
    ) -> tuple:
        """Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image, has shape [num_queries,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_queries, 4].
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple: a tuple containing the following targets.

            - labels_list (list[Tensor]): Labels for all images.
            - label_weights_list (list[Tensor]): Label weights for all images.
            - bbox_targets_list (list[Tensor]): BBox targets for all images.
            - bbox_weights_list (list[Tensor]): BBox weights for all images.
            - num_total_pos (int): Number of positive samples in all images.
            - num_total_neg (int): Number of negative samples in all images.
        """
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, pos_inds_list, neg_inds_list) = (
            multi_apply(
                self._get_targets_single, cls_scores_list, bbox_preds_list, batch_gt_instances, batch_img_metas
            )
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
            pos_inds_list,
            neg_inds_list
        )
