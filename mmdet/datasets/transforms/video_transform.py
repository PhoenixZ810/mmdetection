from typing import Dict, List, Optional, Tuple, Union
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
import numpy as np
import ffmpeg
import random
import torch


@TRANSFORMS.register_module()
class mp4_to_image(BaseTransform):
    def __init__(
        self,
        stride=None,
        video_max_len_train=None,
        video_max_len_val=None,
        fps=5,
        time_crop=False,
        spatial_transform=None,
        is_train=False,
    ):
        self.stride = stride
        if video_max_len_train:
            self.video_max_len = video_max_len_train
        elif video_max_len_val:
            self.video_max_len = video_max_len_val
        self.fps = fps
        self.time_crop = time_crop
        self.is_train = is_train
        self.spatial_transform = TRANSFORMS.build(spatial_transform)

    def transform(self, results):
        if results["dataset_mode"] == "vidstg":
            return self.vidstg_transform(results)

    def vidstg_transform(self, results):
        # ffmpeg decoding
        clip_start = results["clip_start"]
        clip_end = results["clip_end"]
        video_fps = results["video_fps"]
        video_id = results["video_id"]
        frame_ids = results["frame_ids"]
        inter_frames = results["inter_frames"]
        vid_path = results["video_path"]
        trajectory = results["trajectory"]

        ss = clip_start / video_fps  # 视频开始时间
        t = (clip_end - clip_start) / video_fps  # 视频持续时间
        # print(vid_path)
        try:
            cmd = ffmpeg.input(vid_path, ss=ss, t=t).filter(
                "fps", fps=len(frame_ids) / t
            )  # 根据片段的帧数除以时间计算切割出来的帧率
            out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
                capture_stdout=True, quiet=True
            )
        except:
            print(vid_path)
            raise
        w = results["width"]
        h = results["height"]
        images_list = np.frombuffer(out, np.uint8).reshape(
            [-1, h, w, 3]
        )  # 将out转换为np.unit8数组并reshape为[T,H,W,3]

        '''若抽样后的帧数无法对齐，补救措施'''
        if len(images_list) != len(frame_ids):
            print('----------------------------------------------')
            print(len(images_list), len(frame_ids), vid_path)

            probe = ffmpeg.probe(vid_path)
            video_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None
            )
            origin_fps = int(video_stream['avg_frame_rate'].split('/')[0])  # 视频的帧率

            # 根据帧数列表计算视频片段的开始时间、结束时间和帧率
            clip_start = min(frame_ids)  # 片段的开始帧
            clip_end = max(frame_ids)  # 片段的结束帧
            ss = clip_start / origin_fps  # 片段的开始时间
            t = (clip_end - clip_start) / origin_fps  # 片段的持续时间
            new_fps = len(frame_ids) / t  # 片段的帧率
            new_fps = len(frame_ids) / t  # 片段的帧率
            # 使用ffmpeg.trim方法根据视频的时间戳来剪切视频片段
            cmd = ffmpeg.input(vid_path).trim(start=ss, end=ss + t)
            # 使用ffmpeg.setpts方法重新设置视频的时间戳
            cmd = cmd.setpts('PTS-STARTPTS')
            # 使用ffmpeg.filter方法根据视频的帧率来过滤视频片段
            cmd = cmd.filter('fps', fps=new_fps)
            # 使用ffmpeg.output方法将视频片段输出为图像列表
            out, _ = cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24').run(
                capture_stdout=True, quiet=True
            )
            images_list = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])

            print(len(images_list), len(frame_ids), results["video_path"])
            print('----------------------------------------------')

        # assert len(images_list) == len(frame_ids)
        if len(images_list) != len(frame_ids):
            raise

        # prepare frame-level targets
        targets_list = []
        inter_idx = []  # list of indexes of the frames in the annotated moment
        for i_img, img_id in enumerate(frame_ids):
            if img_id in inter_frames:
                anns = trajectory[str(img_id)]  # dictionary with bbox [left, top, width, height] key
                anns = [anns]
                inter_idx.append(i_img)
            else:
                anns = []
            target = prepare(w, h, anns)
            target["image_id"] = f"{video_id}_{img_id}"
            targets_list.append(target)

        # video spatial transform
        if self.spatial_transform is not None:
            [images, targets] = self.spatial_transform([images_list, targets_list])
        else:
            [images, targets] = [images_list, targets_list]

        if inter_idx:  # number of boxes should be the number of frames in annotated moment
            assert len([x for x in targets if len(x["boxes"])]) == inter_idx[-1] - inter_idx[0] + 1, (
                len([x for x in targets if len(x["boxes"])]),
                inter_idx,
            )

        if self.time_crop:
            # temporal crop
            p = random.random()
            if p > 0.5:  # random crop
                # list possible start indexes
                if inter_idx:
                    starts_list = [i for i in range(len(frame_ids)) if i < inter_idx[0]]
                else:
                    starts_list = [i for i in range(len(frame_ids))]

                # sample a new start index
                if starts_list:
                    new_start_idx = random.choice(starts_list)
                else:
                    new_start_idx = 0

                # list possible end indexes
                if inter_idx:
                    ends_list = [i for i in range(len(frame_ids)) if i > inter_idx[-1]]
                else:
                    ends_list = [i for i in range(len(frame_ids)) if i > new_start_idx]

                # sample a new end index
                if ends_list:
                    new_end_idx = random.choice(ends_list)
                else:
                    new_end_idx = len(frame_ids) - 1

                # update everything
                prev_start_frame = frame_ids[0]
                prev_end_frame = frame_ids[-1]
                frame_ids = [x for i, x in enumerate(frame_ids) if new_start_idx <= i <= new_end_idx]
                images = images[:, new_start_idx : new_end_idx + 1]  # CTHW
                targets = [x for i, x in enumerate(targets) if new_start_idx <= i <= new_end_idx]
                clip_start += frame_ids[0] - prev_start_frame
                clip_end += frame_ids[-1] - prev_end_frame
                if inter_idx:
                    inter_idx = [x - new_start_idx for x in inter_idx]

        if (
            self.is_train and len(frame_ids) > self.video_max_len
        ):  # densely sample video_max_len_train frames
            if inter_idx:
                starts_list = [
                    i
                    for i in range(len(frame_ids))
                    if inter_idx[0] - self.video_max_len < i <= inter_idx[-1]
                ]
            else:
                starts_list = [i for i in range(len(frame_ids))]

            # sample a new start index
            if starts_list:
                new_start_idx = random.choice(starts_list)
            else:
                new_start_idx = 0

            # select the end index
            new_end_idx = min(new_start_idx + self.video_max_len - 1, len(frame_ids) - 1)

            # update everything
            prev_start_frame = frame_ids[0]
            prev_end_frame = frame_ids[-1]
            frame_ids = [x for i, x in enumerate(frame_ids) if new_start_idx <= i <= new_end_idx]
            images = images[:, new_start_idx : new_end_idx + 1]  # [C,T,H,W]
            targets = [x for i, x in enumerate(targets) if new_start_idx <= i <= new_end_idx]
            clip_start += frame_ids[0] - prev_start_frame
            clip_end += frame_ids[-1] - prev_end_frame
            if inter_idx:
                inter_idx = [x - new_start_idx for x in inter_idx if new_start_idx <= x <= new_end_idx]

        boxes = [x["boxes"] for x in targets]
        image_ids = np.array([x["image_id"] for x in targets])
        box_labels = np.array([0 for _ in range(len(targets))])
        inter_frames = []
        for i in inter_idx:
            inter_frames.append(f"{video_id}_{frame_ids[i]}")
        results["inter_idx"] = [inter_idx[0], inter_idx[-1]] if inter_idx else [-100, -100]

        if self.stride:
            results["img"] = images[:, :: self.stride]
            results["img_all"] = images
            results['img_shape'] = [h, w]
            # results['img_shape'] = [images.shape[2], images.shape[3]]
            results["gt_bboxes"] = boxes
            results['img_in_vid_ids'] = image_ids  # video_id + frame_id for all images
            results["gt_bboxes_labels"] = box_labels
            results["inter_idx"] = [inter_idx[0], inter_idx[-1]] if inter_idx else [-100, -100]
            results["inter_frames"] = inter_frames
            results["frames_id"] = frame_ids
        else:
            results["img"] = images
            results['img_shape'] = [h, w]
            # results['img_shape'] = [images.shape[2], images.shape[3]]
            results["gt_bboxes"] = boxes
            results['img_in_vid_ids'] = image_ids
            results["gt_bboxes_labels"] = box_labels
            results["inter_idx"] = [inter_idx[0], inter_idx[-1]] if inter_idx else [-100, -100]
            results["inter_frames"] = inter_frames
            results["frames_id"] = frame_ids
        return results


def prepare(w, h, anno):
    """
    :param w: pixel width of the frame
    :param h: pixel height of the frame
    :param anno: dictionary with key bbox
    :return: dictionary with preprocessed keys tensors boxes and orig_size
    """
    boxes = [obj["bbox"] for obj in anno]
    # guard against no boxes via resizing
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2].clamp_(min=0, max=w)
    boxes[:, 1::2].clamp_(min=0, max=h)

    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    boxes = boxes[keep]

    target = {}
    target["boxes"] = boxes
    target["orig_size"] = torch.as_tensor([int(h), int(w)])

    return target


# def make_video_transforms(image_set, cautious, resolution=224):
#     """
#     :param image_set: train val or test
#     :param cautious: whether to preserve bounding box annotations in the spatial random crop
#     :param resolution: spatial pixel resolution for the shortest side of each frame
#     :return: composition of spatial data transforms to be applied to every frame of a video
#     """

#     normalizeop = Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#     if resolution == 128:
#         scales = [96, 128]
#         max_size = 213
#         resizes = [80, 100, 120]
#         crop = 64
#         test_size = [128]
#     elif resolution == 224:
#         scales = [128, 160, 192, 224]
#         max_size = 373
#         resizes = [100, 150, 200]
#         crop = 96
#         test_size = [224]
#     elif resolution == 256:
#         scales = [160, 192, 224, 256]
#         max_size = 427
#         resizes = [140, 180, 220]
#         crop = 128
#         test_size = [256]
#     elif resolution == 288:
#         scales = [160, 192, 224, 256, 288]
#         max_size = 480
#         resizes = [150, 200, 250]
#         crop = 128
#         test_size = [288]
#     elif resolution == 320:
#         scales = [192, 224, 256, 288, 320]
#         max_size = 533
#         resizes = [200, 240, 280]
#         crop = 160
#         test_size = [320]
#     elif resolution == 352:
#         scales = [224, 256, 288, 320, 352]
#         max_size = 587
#         resizes = [200, 250, 300]
#         crop = 192
#         test_size = [352]
#     elif resolution == 384:
#         scales = [224, 256, 288, 320, 352, 384]
#         max_size = 640
#         resizes = [200, 250, 300]
#         crop = 192
#         test_size = [384]
#     elif resolution == 416:
#         scales = [256, 288, 320, 352, 384, 416]
#         max_size = 693
#         resizes = [240, 300, 360]
#         crop = 224
#         test_size = [416]
#     elif resolution == 448:
#         scales = [256, 288, 320, 352, 384, 416, 448]
#         max_size = 746
#         resizes = [240, 300, 360]
#         crop = 224
#         test_size = [448]
#     elif resolution == 480:
#         scales = [288, 320, 352, 384, 416, 448, 480]
#         max_size = 800
#         resizes = [240, 300, 360]
#         crop = 240
#         test_size = [480]
#     elif resolution == 800:
#         scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
#         max_size = 1333
#         resizes = [400, 500, 600]
#         crop = 384
#         test_size = [800]
#     else:
#         raise NotImplementedError

#     if image_set == "train":
#         horizontal = [] if cautious else [RandomHorizontalFlip()]
#         return Compose(
#             horizontal
#             + [
#                 RandomSelect(
#                     RandomResize(scales, max_size=max_size),
#                     Compose(
#                         [
#                             RandomResize(resizes),
#                             RandomSizeCrop(crop, max_size, respect_boxes=cautious),
#                             RandomResize(scales, max_size=max_size),
#                         ]
#                     ),
#                 ),
#                 normalizeop,
#             ]
#         )

#     if image_set in ["val", "test"]:
#         return Compose(
#             [
#                 RandomResize(test_size, max_size=max_size),
#                 normalizeop,
#             ]
#         )

#     raise ValueError(f"unknown {image_set}")
