# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import itertools
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MMLogger
from terminaltables import AsciiTable

from mmdet.datasets.api_wrappers import COCO, COCOeval, COCOevalMP
from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results
from ..functional import eval_recalls
import json

from ..functional import bbox_overlaps


@METRICS.register_module()
class VidstgMetric(BaseMetric):
    """COCO evaluation metric.

    Evaluate AR, AP, and mAP for detection tasks including proposal/box
    detection and instance segmentation. Please refer to
    https://cocodataset.org/#detection-eval for more details.

    Args:
        ann_file (str, optional): Path to the coco format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to coco format. Defaults to None.
        metric (str | List[str]): Metrics to be evaluated. Valid metrics
            include 'bbox', 'segm', 'proposal', and 'proposal_fast'.
            Defaults to 'bbox'.
        classwise (bool): Whether to evaluate the metric class-wise.
            Defaults to False.
        proposal_nums (Sequence[int]): Numbers of proposals to be evaluated.
            Defaults to (100, 300, 1000).
        iou_thrs (float | List[float], optional): IoU threshold to compute AP
            and AR. If not specified, IoUs from 0.5 to 0.95 will be used.
            Defaults to None.
        metric_items (List[str], optional): Metric result names to be
            recorded in the evaluation result. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format and submit it to the test server.
            Defaults to False.
        outfile_prefix (str, optional): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
        file_client_args (dict, optional): Arguments to instantiate the
            corresponding backend in mmdet <= 3.0.0rc6. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        sort_categories (bool): Whether sort categories in annotations. Only
            used for `Objects365V1Dataset`. Defaults to False.
        use_mp_eval (bool): Whether to use mul-processing evaluation
    """

    # default_prefix: Optional[str] = 'coco'

    def __init__(
        self,
        ann_file: Optional[str] = None,
        metric: Union[str, List[str]] = 'bbox',
        classwise: bool = False,
        proposal_nums: Sequence[int] = (100, 300, 1000),
        iou_thrs: Optional[Union[float, Sequence[float]]] = None,
        metric_items: Optional[Sequence[str]] = None,
        format_only: bool = False,
        outfile_prefix: Optional[str] = None,
        file_client_args: dict = None,
        backend_args: dict = None,
        collect_device: str = 'cpu',
        prefix: Optional[str] = None,
        sort_categories: bool = False,
        use_mp_eval: bool = False,
        use_sted: bool = False,
        postprocessors: Optional[list] = None,
        tmp_loc: bool = False,
        iou_thresholds: list = [0.3, 0.5],
    ) -> None:
        super().__init__()
        self.metric = metric
        self.use_sted = use_sted
        self.postprocessors = postprocessors
        self.tmp_loc = tmp_loc
        self.iou_thresholds = iou_thresholds

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of output data predictions
        """
        # only keep box predictions in the annotated moment
        data_batch = data_batch['data_samples']
        durations = data_batch[0].durations
        inter_idx = [data_batch[duration - 1].inter_idx for duration in durations]
        img_in_vid_ids = [data_batch[i].img_in_vid_ids for duration in durations for i in range(duration)]
        max_duration = max(durations)
        device = data_samples[0]['pred_instances']['bboxes'].device
        keep_list = []
        for i_dur, (duration, inter) in enumerate(zip(durations, inter_idx)):
            if inter[0] >= 0:
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
        results = {}

        results["pred_boxes"] = []
        results["pred_sted"] = []
        for i in keep_list:
            results["pred_boxes"].append(data_samples[i]['pred_instances']['bboxes'])
        if self.use_sted:
            for i in range(len(data_samples)):
                results["pred_sted"].append(data_samples[i]['pred_instances']['sted'])

        b = len(durations)
        gts = [x for x in data_batch if len(x.gt_instances.bboxes)]  # 判断真实框的数目
        if len(gts) != inter_idx[0][1] - inter_idx[0][0] + 1:
            print("none")

        assert len(gts) == len(results["pred_boxes"]), (
            len(gts),
            len(results["pred_boxes"]),
        )
        # mask with padded positions set to False for loss computation
        if self.use_sted:
            time_mask = torch.zeros(b, data_samples[0]['pred_instances']['sted'].shape[1]).bool().to(device)
            for i_dur, duration in enumerate(durations):
                time_mask[i_dur, :duration] = True
        else:
            time_mask = None

        vidstg_res = {} if "vidstg" in self.postprocessors else None
        vidstg_video_res = {} if "vidstg" in self.postprocessors else None
        hcstvg_res = {} if "hcstvg" in self.postprocessors else None
        hcstvg_video_res = {} if "hcstvg" in self.postprocessors else None

        video_ids = [data_batch[0].video_id]
        frames_id = [data_batch[0].frames_id]
        image_ids = [gt.img_in_vid_ids for gt in gts]

        # ground-truth dictionary for each image of all videos
        gt_dict = {}
        for i, gt in enumerate(gts):
            gt_dict[image_ids[i]] = gt

        if "vidstg" in self.postprocessors:
            if self.use_sted:
                pred_steds = self.PostProcessSTVG(
                    results, frames_id, video_ids=video_ids, time_mask=time_mask
                )
            for im_id, box in zip(image_ids, results["pred_boxes"]):
                vidstg_res[im_id] = {"boxes": [box.detach().cpu()]}

            qtypes = data_batch[0].qtype
            # assert len(set(video_ids)) == len(set('qtypes'))

            if self.use_sted:
                assert len(pred_steds) == len(qtypes)
                for video_id, pred_sted in zip(video_ids, pred_steds):
                    vidstg_video_res[video_id] = {
                        "sted": pred_sted,
                        "qtype": qtypes[video_id],
                    }
            else:
                for video_id in video_ids:
                    vidstg_video_res[video_id] = {
                        "qtype": qtypes[video_id],
                    }
            self.results.append((gt_dict, vidstg_res, vidstg_video_res))
            # res_dict = {gt.img_in_vid_ids: output for gt, output in zip(gts, results)}

        elif "hcstvg" in self.postprocessors:
            if self.use_sted:
                pred_steds = self.PostProcessSTVG(
                    results, frames_id, video_ids=video_ids, time_mask=time_mask
                )
            for im_id, box in zip(image_ids, results["pred_boxes"]):
                hcstvg_res[im_id] = {"boxes": [box.detach().cpu()]}

            if self.use_sted:
                assert len(set(video_ids)) == len(pred_steds)
                for video_id, pred_sted in zip(video_ids, pred_steds):
                    hcstvg_video_res[video_id] = {"sted": pred_sted}
            else:
                hcstvg_video_res[video_id] = {}
            self.results.append((gt_dict, hcstvg_res, hcstvg_video_res))
        #     res = {target["image_id"]: output for target, output in zip(targets, results)}
        # else:
        #     res = {target["image_id"].item(): output for target, output in zip(targets, results)}

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        self.predictions = {}
        self.video_predictions = {}
        self.gts = {}
        # split gt and prediction list
        gts, res, video_res = zip(*results)
        for gt in gts:
            self.gts.update(gt)
        for r in res:
            self.predictions.update(r)
        for r in video_res:
            self.video_predictions.update(r)
        sum = self.summarize()
        return sum

    def summarize(self):
        results = self.eval(self.predictions, self.video_predictions)
        categories = set(x["qtype"] for x in results.values())
        metrics = {}
        counter = {}
        for category in categories:  # init metrics
            metrics[category] = {"gt_viou": 0}
            if self.tmp_loc:
                metrics[category].update({"tiou": 0, "viou": 0})
            for thresh in self.iou_thresholds:
                if self.tmp_loc:
                    metrics[category][f"viou@{thresh}"] = 0
                metrics[category][f"gt_viou@{thresh}"] = 0
            counter[category] = 0
        for x in results.values():  # sum results
            qtype = x["qtype"]
            if self.tmp_loc:
                metrics[qtype]["tiou"] += x["tiou"]
                metrics[qtype]["viou"] += x["viou"]
            metrics[qtype]["gt_viou"] += x["gt_viou"]
            for thresh in self.iou_thresholds:
                if self.tmp_loc:
                    metrics[qtype][f"viou@{thresh}"] += x[f"viou@{thresh}"]
                metrics[qtype][f"gt_viou@{thresh}"] += x[f"gt_viou@{thresh}"]
            counter[qtype] += 1
        for category in categories:  # average results per category
            for key in metrics[qtype]:
                metrics[category][key] = metrics[category][key] / counter[category]
                print(f"{category} {key}: {metrics[category][key]:.4f}")
        out = {f"{qtype}_{name}": metrics[qtype][name] for qtype in metrics for name in metrics[qtype]}
        # if self.save_pred:
        #     out["predictions"] = self.predictions
        #     out["video_predictions"] = self.video_predictions
        #     out["vid_metrics"] = self.results
        #     if len(self.tsa_weights):
        #         out["tsa_weights"] = self.tsa_weights
        #         out["text_weights"] = self.text_weights
        #         out["spatial_weights"] = self.spatial_weights
        #         out["pred_sted"] = self.pred_sted
        return out

    def eval(self, predictions: List[Dict], video_predictions: List[Dict]):
        # if len(video_predictions) < len(self.vid2imgids):
        #     raise RuntimeError(f"{len(self.vid2imgids) - len(video_predictions)} video predictions missing")
        # if len(predictions) < len(self.img2box):
        #     raise RuntimeError(f"{len(self.img2box) - len(predictions)} box predictions missing")
        vid_metrics = {}

        # evaluate per video
        for video_id, video_pred in video_predictions.items():
            if video_id in vid_metrics:
                print(f"Warning, multiple predictions found for video {video_id}")
                continue
            if self.tmp_loc:
                for key in self.gts.keys():
                    if int(key.split("_")[0]) == video_id:
                        gt_sted = [self.gts[key].tube_start_frame, self.gts[key].tube_end_frame]
                        break

                pred_sted = video_pred["sted"]
            qtype = video_pred["qtype"]

            # collect frames_id and inter_frames in this video
            inter_frames = []
            for key in self.gts.keys():
                if int(key.split("_")[0]) == video_id:
                    frame_ids = self.gts[key].frames_id
                    inter_frames = self.gts[key].inter_frames
                    break

            # compute temporal iou
            if self.tmp_loc:
                max_start = max(gt_sted[0], pred_sted[0])
                min_end = min(gt_sted[1], pred_sted[1])
                min_start = min(gt_sted[0], pred_sted[0])
                max_end = max(gt_sted[1], pred_sted[1])
                if min_end <= max_start:
                    tiou = 0
                else:
                    intersection = min_end - max_start
                    gt_span = gt_sted[1] - gt_sted[0]
                    pred_span = pred_sted[1] - pred_sted[0]
                    union = gt_span + pred_span - intersection
                    tiou = intersection / union

                # compute viou and gt_viou
                vid_metrics[video_id] = {
                    "gt_sted": gt_sted,
                    "pred_sted": pred_sted,
                    "tiou": tiou,
                    "qtype": qtype,
                    "img_metrics": {},
                }
                union_predgt = [frame_id for frame_id in frame_ids if min_start <= frame_id < max_end]
                inter_predgt = set([frame_id for frame_id in frame_ids if max_start <= frame_id < min_end])
                viou = 0
            else:
                vid_metrics[video_id] = {
                    "qtype": qtype,
                    "img_metrics": {},
                }
                union_predgt = frame_ids
                inter_predgt = frame_ids
            gt_viou = 0

            for i_img, image_id in enumerate(
                inter_frames
            ):  # iterate on all frames of the annotated moment to update GT metrics
                if image_id not in predictions:
                    raise RuntimeError(f"No prediction for frame {image_id}")
                else:
                    pred_boxes = predictions[image_id]['boxes']
                gt_boxes = self.gts[image_id].gt_instances.bboxes
                iou = bbox_overlaps(np.array(pred_boxes[0]), np.array(gt_boxes))[0][0]
                frame_id = int(image_id.split("_")[1])
                vid_metrics[video_id]["img_metrics"][image_id] = {
                    "iou": iou,
                    "pred_box": pred_boxes[0],
                    "gt_box": gt_boxes[0],
                }
                if (
                    frame_id in inter_predgt and self.tmp_loc
                ):  # update viou if this frame is in the intersection between the annotated moment and the predicted moment
                    viou += iou
                gt_viou += iou

            if self.tmp_loc:  # compute viou@R
                viou = viou / max(len(union_predgt), 1)
                vid_metrics[video_id]["viou"] = viou
                recalls = {thresh: 0 for thresh in self.iou_thresholds}
                for thresh in self.iou_thresholds:
                    if viou > thresh:
                        recalls[thresh] += 1
                vid_metrics[video_id].update(
                    {f"viou@{thresh}": recalls[thresh] for thresh in self.iou_thresholds}
                )

            # compute gt_viou@R
            gt_viou = gt_viou / max(len(inter_frames), 1)
            vid_metrics[video_id]["gt_viou"] = gt_viou
            gt_recalls = {thresh: 0 for thresh in self.iou_thresholds}
            for thresh in self.iou_thresholds:
                if gt_viou > thresh:
                    gt_recalls[thresh] += 1
            vid_metrics[video_id].update(
                {f"gt_viou@{thresh}": gt_recalls[thresh] for thresh in self.iou_thresholds}
            )

        return vid_metrics

    def PostProcessSTVG(self, outputs, frames_id=None, video_ids=None, time_mask=None):
        """
        :param outputs: must contain a key pred_sted mapped to a [B, T, 2] tensor of logits for the start and end predictions
        :param frames_id: list of B lists which contains the increasing list of frame ids corresponding to the indexes of the decoder outputs
        :param video_ids: list of B video_ids, used to ensemble predictions when video_max_len_train < video_max_len
        :param time_mask: [B, T] tensor with False on the padded positions, used to take out padded frames from the possible predictions
        :return: list of B [start_frame, end_frame] for each video
        """
        steds = torch.stack(outputs["pred_sted"], dim=0).transpose(0, 1)  # BxTx2
        if len(set(video_ids)) != len(
            video_ids
        ):  # concatenate start and end probabilities predictions across all clips
            steds_list = [steds[0].masked_fill(~time_mask[0][:, None], -float("inf"))]
            for i_vid in range(1, len(video_ids)):
                if video_ids[i_vid] == video_ids[i_vid - 1]:  # same video, concatenate prob logits
                    steds_list[-1] = torch.cat(
                        [
                            steds_list[-1],
                            steds[i_vid].masked_fill(~time_mask[i_vid][:, None], -float("inf")),
                        ],
                        0,
                    )
                else:  # new video
                    steds_list.append(steds[i_vid].masked_fill(~time_mask[i_vid][:, None], -float("inf")))
            n_videos = len(set(video_ids))
            max_dur = max(len(x) for x in steds_list)
            eff_steds = torch.ones(n_videos, max_dur, 2) * float("-inf")
            for i_v in range(len(steds_list)):
                eff_steds[i_v, : len(steds_list[i_v])] = steds_list[i_v]
            steds = eff_steds
        # put 0 probability to positions corresponding to end <= start
        mask = (
            (torch.ones(steds.shape[1], steds.shape[1]) * float("-inf"))
            .to(steds.device)
            .tril(0)
            .unsqueeze(0)
            .expand(steds.shape[0], -1, -1)
        )  # BxTxT
        starts_distribution = steds[:, :, 0].log_softmax(1)  # BxT
        ends_distribution = steds[:, :, 1].log_softmax(1)  # BxT
        # add log <=> multiply probs
        score = (starts_distribution.unsqueeze(2) + ends_distribution.unsqueeze(1)) + mask  # BxTxT
        score, s_idx = score.max(dim=1)  # both BxT
        score, e_idx = score.max(dim=1)  # both B
        s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze(1)  # B
        pred_steds = torch.stack([s_idx, e_idx], 1)  # Bx2
        # max_length = max([len(x) for x in frames_id])
        max_length = steds.shape[1]
        frames_id = (
            torch.tensor([row + [0] * (max_length - len(row)) for row in frames_id])
            .long()
            .to(pred_steds.device)
        )  # padded up to BxT
        # get corresponding frames id from the indexes
        pred_steds = torch.gather(frames_id, 1, pred_steds)
        pred_steds = pred_steds.float()
        pred_steds[:, 1] += 1  # the end frame is excluded in evaluation

        pred_steds = pred_steds.cpu().tolist()
        return pred_steds
