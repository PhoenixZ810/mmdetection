# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from typing import List, Optional
import torch

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .base_det_dataset import BaseDetDataset

import os
from pathlib import Path
import time
import ffmpeg
import numpy as np
import random


@DATASETS.register_module()
class HCVideoModulatedSTGrounding(BaseDetDataset):
    def __init__(
        self,
        *args,
        data_root,
        is_train=False,
        tmp_loc=True,
        stride=None,
        fps=5,
        video_max_len=200,
        **kwargs,
    ):
        """
        :param vid_folder: path to the folder containing a folder "video"
        :param ann_file: path to the json annotation file
        :param transforms: video data transforms to be applied on the videos and boxes
        :param is_train: whether training or not
        :param video_max_len: maximum number of frames to be extracted from a video
        :param video_max_len_train: maximum number of frames to be extracted from a video at training time
        :param fps: number of frames per second
        :param tmp_crop: whether to use temporal cropping preserving the annotated moment
        :param tmp_loc: whether to use temporal localization annotations
        :param stride: temporal stride k
        """

        self.is_train = is_train
        self.video_max_len = video_max_len
        self.fps = fps
        self.tmp_loc = tmp_loc
        self.stride = stride
        super().__init__(*args, data_root=data_root, **kwargs)

        # self.vid2imgids = (
        #     {}
        # )  # map video_id to [list of frames to be forwarded, list of frames in the annotated moment]

    def load_data_list(self):
        """
        :param idx: int
        :return:
        images: a CTHW video tensor
        targets: list of frame-level target, one per frame, dictionary with keys image_id, boxes, orig_sizes
        tmp_target: video-level target, dictionary with keys video_id, qtype, inter_idx, frames_id, caption
        """
        with get_local_path(self.ann_file, backend_args=self.backend_args) as local_path:
            with open(local_path, 'r') as f:
                data_list = json.load(f)
        out_data_list = []
        empty_video = []
        for i_vid, video in enumerate(data_list):
            video_num_images = video["frame_count"]
            video_fps = video_num_images / 20  # duration of videos in HC-STVG is 20s
            sampling_rate = self.fps / video_fps  # 采样率
            assert sampling_rate <= 1  # downsampling at fps
            start_frame = 0 if self.tmp_loc else video["tube_start_frame"]
            end_frame = video_num_images - 1 if self.tmp_loc else video["tube_end_frame"]
            frame_ids = [start_frame]
            for frame_id in range(start_frame, end_frame):  # 按照采样率保留帧
                if int(frame_ids[-1] * sampling_rate) < int(frame_id * sampling_rate):
                    frame_ids.append(frame_id)

            if len(frame_ids) > self.video_max_len:  # subsample at video_max_len
                frame_ids = [
                    frame_ids[(j * len(frame_ids)) // self.video_max_len] for j in range(self.video_max_len)
                ]

            inter_frames = set(
                [
                    frame_id
                    for frame_id in frame_ids
                    if video["tube_start_frame"] <= frame_id < video["tube_end_frame"]
                ]
            )  # frames in the annotated moment
            if inter_frames == set():  # if no frame in the annotated moment, take the first frame
                empty_video.append([video["video_path"], video["original_video_id"], video["target_id"]])
                continue

            # self.vid2imgids[video["video_id"]] = [frame_ids, inter_frames]

            # if video["video_path"] == '1124/8588003727.mp4':
            #     print('----------------------------------------------')
            caption = video["caption"]
            video_id = video["video_id"]
            video_original_id = video["original_video_id"]
            clip_start = 0
            clip_end = video["frame_count"] - 1
            # frame_ids, inter_frames = self.vid2imgids[video_id]
            trajectory = video["trajectory"]
            if video['video_path'] not in ['47_uNT6HrrnqPU.mp4'] and video['video_path'] not in [
                '45_1XZZnWMP4CU.mkv',
                '142_1XZZnWMP4CU.mkv',
                '106_1XZZnWMP4CU.mkv',
                '73_1XZZnWMP4CU.mkv',
                '76_1XZZnWMP4CU.mkv',
                '37_kLDpP9QEVBs.mp4',
                '65_1XZZnWMP4CU.mkv',
                '149_1XZZnWMP4CU.mkv',
                '102_1XZZnWMP4CU.mkv',
                '21_1XZZnWMP4CU.mkv',
                '101_1XZZnWMP4CU.mkv',
                '99_UOfuzrwkclM.mkv',
                '169_UOyyTUX5Vo4.mp4',
                '77_VsYPP2I0aUQ.mp4',
                '107_1XZZnWMP4CU.mkv',
                '115_1XZZnWMP4CU.mkv',
                '133_1XZZnWMP4CU.mkv',
                '92_1ReZIMmD_8E.mp4',
                '53_1XZZnWMP4CU.mkv',
                '47_1XZZnWMP4CU.mkv',
                '24_1XZZnWMP4CU.mkv',
                '34_1XZZnWMP4CU.mkv',
                '77_1XZZnWMP4CU.mkv',
                '118_1XZZnWMP4CU.mkv',
                '70_1XZZnWMP4CU.mkv',
                '111_1XZZnWMP4CU.mkv',
                '242_kW5WyJ1QNpM.mkv',
                '94_1XZZnWMP4CU.mkv',
                '71_1XZZnWMP4CU.mkv',
                '47_uNT6HrrnqPU.mp4',
                '43_1XZZnWMP4CU.mkv',
                '105_1XZZnWMP4CU.mkv',
                '51_1XZZnWMP4CU.mkv',
                '99_1XZZnWMP4CU.mkv',
                '130_1XZZnWMP4CU.mkv',
                '147_1XZZnWMP4CU.mkv',
                '55_RW-H3fN_79I.mp4',
                '144_1XZZnWMP4CU.mkv',
                '44_1XZZnWMP4CU.mkv',
                '1_aMYcLyh9OhU.mkv',
            ]:
                vid_path = os.path.join(self.data_prefix['img'], video["video_path"])
            else:
                print(video['video_path'])
                continue
            w = video["width"]
            h = video["height"]
            tube_start_frame = video["tube_start_frame"]
            tube_end_frame = video["tube_end_frame"]
            dataset_info = {
                "video_id": video_id,
                "video_original_id": video_original_id,
                "video_path": vid_path,
                "height": h,
                "width": w,
                "video_fps": video_fps,
                "clip_start": clip_start,
                "clip_end": clip_end,
                "frame_ids": frame_ids,
                "inter_frames": inter_frames,
                "trajectory": trajectory,
                "text": caption,
                # "text":'person.',
                "dataset_mode": 'hcstvg',
                "tube_start_frame": tube_start_frame,
                "tube_end_frame": tube_end_frame,
                'tokens_positive': -1,
            }
            out_data_list.append(dataset_info)
        print(f"empty_video: {len(empty_video)}")
        print(empty_video)
        return out_data_list
