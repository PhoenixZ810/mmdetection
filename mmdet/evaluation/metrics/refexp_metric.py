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
from mmengine.fileio import get_local_path
from mmengine.logging import MMLogger

from mmdet.datasets.api_wrappers import COCO
from mmdet.registry import METRICS
from ..functional import bbox_overlaps


@METRICS.register_module()
class RefExpMetric(BaseMetric):
    default_prefix: Optional[str] = 'refexp'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 metric: str = 'bbox',
                 topk=(1, 5, 10),
                 iou_thrs: float = 0.5,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.metric = metric
        self.topk = topk
        self.iou_thrs = iou_thrs

        with get_local_path(ann_file) as local_path:
            self.coco = COCO(local_path)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()

        dataset2score = {
            "refcoco": {k: 0.0 for k in self.k},
            "refcoco+": {k: 0.0 for k in self.k},
            "refcocog": {k: 0.0 for k in self.k},
        }
        dataset2count = {"refcoco": 0.0, "refcoco+": 0.0, "refcocog": 0.0}

        for result in results:
            img_id = result['img_id']

            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            assert len(ann_ids) == 1
            img_info = self.coco.loadImgs(img_id)[0]
            target = self.coco.loadAnns(ann_ids[0])

            sorted_scores_boxes = sorted(
                zip(result["scores"].tolist(), result["boxes"].tolist()), reverse=True
            )
            sorted_scores, sorted_boxes = zip(*sorted_scores_boxes)
            sorted_boxes = torch.cat([torch.as_tensor(x).view(1, 4) for x in sorted_boxes])
            target_bbox = target[0]["bbox"]
            converted_bbox = [
                target_bbox[0],
                target_bbox[1],
                target_bbox[2] + target_bbox[0],
                target_bbox[3] + target_bbox[1],
            ]
            giou = bbox_overlaps(sorted_boxes, torch.as_tensor(converted_bbox).view(-1, 4))
            for k in self.topk:
                if max(giou[:k]) >= self.iou_thrs:
                    dataset2score[img_info["dataset_mode"]][k] += 1.0
            dataset2count[img_info["dataset_mode"]] += 1.0

        for key, value in dataset2score.items():
            for k in self.topk:
                try:
                    value[k] /= dataset2count[key]
                except:
                    pass

        results = {}
        for key, value in dataset2score.items():
            results[key] = sorted([v for k, v in value.items()])
            print(f" Dataset: {key} - Precision @ 1, 5, 10: {results[key]} \n")

        return results
