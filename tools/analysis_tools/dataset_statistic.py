# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import json
from tqdm import tqdm
import numpy as np
from mmcv.image import imwrite
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope

from mmdet.registry import DATASETS


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--output-dir', '-o', default=None, type=str, help='If there is no display interface, you can save it'
    )
    parser.add_argument('--dataset', '-d',type=str, default=None,  help='train config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.',
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register all modules in mmdet into the registries
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    dataset = DATASETS.build(cfg.train_dataloader.dataset)

    dataset_index = list(range(len(dataset)))

    statistic_label_dict = {}
    statistic_box_dict = {}
    print('start statistic')
    for i in tqdm(dataset_index, total=len(dataset_index)):
        item = dataset[i]
#         data_sample = item['data_samples'].numpy()
#         gt_instances = data_sample.gt_instances

#         gt_labels = gt_instances.labels
#         gt_bboxes = gt_instances.get('bboxes', None)

#         if len(gt_labels) not in statistic_label_dict:
#             statistic_label_dict[len(gt_labels)] = 0
#         if len(gt_bboxes) not in statistic_box_dict:
#             statistic_box_dict[len(gt_bboxes)] = 0
#         statistic_label_dict[len(gt_labels)] += 1
#         statistic_box_dict[len(gt_bboxes)] += 1

#     statistic_label_dict = json.dumps(statistic_label_dict)
#     statistic_box_dict = json.dumps(statistic_box_dict)
#     with open('statistic_dict.txt', 'a') as f:
#         f.write(args.dataset + f', total = {len(dataset)}\n')
#         f.write('statistic_label_dict\n')
#         f.write(statistic_label_dict+'\n')
#         f.write('statistic_box_dict\n')
#         f.write(statistic_box_dict+'\n')


if __name__ == '__main__':
    main()
