from typing import Dict, List, Optional, Tuple, Union
import sys
import copy

import torch
from torch import Tensor
import torch.nn as nn
from mmcv.cnn import Linear
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from ..layers import inverse_sigmoid
from .grounding_dino_head import ContrastiveEmbed
from .video_grounding_dino_head import VideoGroundingHead
from mmdet.utils import InstanceList, OptInstanceList
from .deformable_detr_head import DeformableDETRHead
from ..layers import MLP

@MODELS.register_module()
class VideoSTCATGroundingHead(VideoGroundingHead):
    def __init__(self, use_actioness=False, **kwargs):
        self.use_actioness = use_actioness
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
        if self.use_sted:
            self.sted_branch = MLP(self.embed_dims, self.embed_dims, 2, 2)
        if self.use_enc_sted:
            if self.share_pred_layer:
                self.sted_branch = nn.ModuleList([self.sted_branch, self.sted_branch])
            else:
                self.sted_branch = nn.ModuleList(
                    [copy.deepcopy(self.sted_branch), copy.deepcopy(self.sted_branch)]
                )
        if self.use_actioness:
            self.action_embed = MLP(self.embed_dims, self.embed_dims, 1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        memory_text: Tensor,
        text_token_mask: Tensor,
        output_time: Optional[Tensor] = None,
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

        if self.use_sted:
            if self.use_enc_sted:
                outputs_sted = self.sted_branch[1](output_time)
            else:
                outputs_sted = self.sted_branch(output_time)
        else:
            outputs_sted = None

        if self.use_actioness:
            outputs_actioness = self.action_embed(output_time)
        else:
            outputs_actioness = None

        for layer_id in range(hidden_states.shape[0]):
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

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)

        return all_layers_outputs_classes, all_layers_outputs_coords, outputs_sted,outputs_actioness

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
        output_time: Optional[Tensor] = None,
        weights: Optional[List[Tensor]] = [],
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

        outs = self(hidden_states, references, memory_text, text_token_mask, output_time=output_time)
        self.text_masks = text_token_mask

        durations = batch_data_samples[0].durations
        inter_idx = [batch_data_samples[duration - 1].inter_idx for duration in durations]

        # set time_mask of valid video frames for each batchsize(duration)
        if self.use_sted:
            device = hidden_states.device
            time_mask = torch.zeros(1, outs[2].shape[0]).bool().to(device)
            # time_mask = torch.zeros(1, outs[0].shape[0]).bool().to(device)
            for i_dur, duration in enumerate(durations):
                time_mask[i_dur, :duration] = True
        else:
            time_mask = None

        loss_inputs = outs + (
            weights[-1],
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
        outputs_actioness,
        weights: Optional[List[Tensor]],
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
            original_all_layers_matching_cls_scores = all_layers_cls_scores
            original_all_layers_matching_bbox_preds = all_layers_bbox_preds
            # print('original_shape', original_all_layers_matching_cls_scores.shape)
            all_layers_denoising_cls_scores = None
            all_layers_denoising_bbox_preds = None

        max_duration = max(durations)
        device = original_all_layers_matching_cls_scores.device

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

        original_enc_cls_scores = enc_cls_scores
        original_enc_bbox_preds = enc_bbox_preds

        if len(keep) != original_all_layers_matching_cls_scores.shape[1]:
            # 若有帧没有box，需要去掉这些帧
            all_layers_matching_cls_scores = torch.index_select(
                original_all_layers_matching_cls_scores, 1, keep
            )
            all_layers_matching_bbox_preds = torch.index_select(
                original_all_layers_matching_bbox_preds, 1, keep
            )
            self.text_masks = torch.index_select(self.text_masks, 0, keep)
            self.keep = keep
            if use_dn:
                all_layers_denoising_cls_scores = all_layers_denoising_cls_scores[:, keep]
                all_layers_denoising_bbox_preds = all_layers_denoising_bbox_preds[:, keep]
            if enc_cls_scores is not None:
                enc_cls_scores = torch.index_select(original_enc_cls_scores, 0, keep)
                enc_bbox_preds = torch.index_select(original_enc_bbox_preds, 0, keep)
            # print('batch_img_metas=', len(batch_img_metas), 'keep=', len(keep))
            img_metas = [batch_img_metas[i] for i in keep]
            gt_instances = [batch_gt_instances[i] for i in keep]
            batch_img_metas = img_metas
            batch_gt_instances = gt_instances
        else:
            self.keep = keep
            all_layers_matching_cls_scores = original_all_layers_matching_cls_scores
            all_layers_matching_bbox_preds = original_all_layers_matching_bbox_preds
            enc_bbox_preds = original_enc_bbox_preds
            enc_cls_scores = original_enc_cls_scores

        loss_dict = super(DeformableDETRHead, self).loss_by_feat(
            all_layers_matching_cls_scores,
            all_layers_matching_bbox_preds,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
        )

        # NOTE DETRHead.loss_by_feat but not DeformableDETRHead.loss_by_feat
        # is called, because the encoder loss calculations are different
        # between DINO and DeformableDETR.

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            # NOTE The enc_loss calculation of the DINO is
            # different from that of Deformable DETR.
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = self.loss_by_feat_single(
                enc_cls_scores,
                enc_bbox_preds,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
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
            loss_dict.update(self.loss_sted(outputs_sted, num_boxes, inter_idx, positive_map, time_mask))
            if self.use_enc_sted:
                loss_enc = self.loss_sted(
                    enc_outputs_sted, num_boxes, inter_idx, positive_map, time_mask, enc_flag=True
                )
                loss_dict['enc_loss_sted'] = loss_enc['loss_sted']

        if weights is not []:
            loss_dict.update(
                self.loss_guided_attn(weights, positive_map, time_mask)
            )
        if self.use_actioness:
            loss_dict.update(self.loss_actioness(outputs_actioness, batch_gt_instances, durations, time_mask))
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

    def predict(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: SampleList,
        output_time: Optional[Tensor] = None,
        weights: Optional[List[Tensor]] = [],
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

        outs = self(hidden_states, references, memory_text, text_token_mask, output_time=output_time)

        predictions = self.predict_by_feat(
            *outs,
            batch_img_metas=batch_img_metas,
            batch_token_positive_maps=batch_token_positive_maps,
            rescale=rescale,
        )
        return predictions

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

    def loss_actioness(self, pred_actioness, targets, gt_temp_bound, time_mask=None):
        # assert "pred_actioness" in outputs
        losses = {}
        pred_actioness = pred_actioness.squeeze(-1)
        target_actioness = torch.stack([target["actioness"] for target in targets], dim=0).float()
        weight = torch.full(pred_actioness.shape, self.eos_coef, device=pred_actioness.device)

        for i_b in range(len(weight)):
            temp_bound = gt_temp_bound[i_b]
            weight[i_b][temp_bound[0] : temp_bound[1] + 1] = 1

        loss_actioness = F.binary_cross_entropy_with_logits(
            pred_actioness, target_actioness, weight=weight, reduction='none'
        )

        loss_actioness = loss_actioness * time_mask
        losses["loss_actioness"] = loss_actioness.mean()
        return losses
