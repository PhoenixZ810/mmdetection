# Adapted from
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .video_vision_torch import ClipToTensor, normalize, resize_clip, crop_clip
import torch
import random
import numpy as np
import copy
import PIL

# from util.misc import interpolate

from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class Video_Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, video, targets):
        for t in self.transforms:
            video, targets = t(video, targets)
        return video, targets

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


@TRANSFORMS.register_module()
class VideoToTensor(object):
    def __init__(self, channel_nb=3, div_255=True, numpy=False):
        self.ClipToTensor = ClipToTensor(channel_nb, div_255, numpy)

    def __call__(self, data):
        video = data[0]
        targets = data[1]
        return [self.ClipToTensor(video), targets]


@TRANSFORMS.register_module()
class VideoBoxNormalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        video = data[0]
        targets = data[1]
        # video = normalize(video, mean=self.mean, std=self.std)  # torch functional videotransforms
        if targets is None:
            return video, None
        targets = targets.copy()
        h, w = video.shape[-2:]
        if "boxes" in targets[0]:  # apply for every image of the clip
            for i_tgt in range(len(targets)):
                boxes = targets[i_tgt]["boxes"]
                # boxes = box_xyxy_to_cxcywh(boxes)
                boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
                targets[i_tgt]["boxes"] = boxes
        return [video, targets]


@TRANSFORMS.register_module()
class VideoRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video, targets):
        if random.random() < self.p:
            return hflip(video, targets)
        return video, targets


def hflip(clip, targets):
    if isinstance(clip[0], np.ndarray):
        flipped_clip = [np.fliplr(img) for img in clip]  # apply for every image of the clip
    elif isinstance(clip[0], PIL.Image.Image):
        flipped_clip = [
            img.transpose(PIL.Image.FLIP_LEFT_RIGHT) for img in clip
        ]  # apply for every image of the clip

    w, h = clip[0].size

    targets = targets.copy()
    if "boxes" in targets[0]:  # apply for every image of the clip
        for i_tgt in range(len(targets)):
            boxes = targets[i_tgt]["boxes"]
            boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
            targets[i_tgt]["boxes"] = boxes

    if "masks" in targets[0]:
        for i_tgt in range(len(targets)):  # apply for every image of the clip
            targets[i_tgt]["masks"] = targets[i_tgt]["masks"].flip(-1)

    if (
        "caption" in targets[0]
    ):  # TODO: quick hack, only modify the first one as all of them should be the same
        caption = (
            targets[0]["caption"].replace("left", "[TMP]").replace("right", "left").replace("[TMP]", "right")
        )
        targets[0]["caption"] = caption

    return flipped_clip, targets


@TRANSFORMS.register_module()
class VideoRandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, video, targets):
        if random.random() < self.p:
            return self.transforms1(video, targets)
        return self.transforms2(video, targets)


def resize(clip, targets, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    if isinstance(clip[0], PIL.Image.Image):
        s = clip[0].size
    elif isinstance(clip[0], np.ndarray):
        h, w, ch = list(clip[0].shape)
        s = [w, h]
    else:
        raise NotImplementedError
    size = get_size(s, size, max_size)  # apply for first image, all images of the same clip have the same h w
    rescaled_clip = resize_clip(clip, size)  # torch video transforms functional
    if isinstance(clip[0], np.ndarray):
        h2, w2, c2 = list(rescaled_clip[0].shape)
        s2 = [w2, h2]
    elif isinstance(clip[0], PIL.Image.Image):
        s2 = rescaled_clip[0].size
    else:
        raise NotImplementedError

    if targets is None:
        return rescaled_clip, None

    ratios = tuple(float(s_mod) / float(s_orig) for s_mod, s_orig in zip(s2, s))
    ratio_width, ratio_height = ratios

    # targets = targets.copy()
    # if "boxes" in targets[0]:
    #     for i_tgt in range(len(targets)):  # apply for every image of the clip
    #         boxes = targets[i_tgt]["boxes"]
    #         scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
    #         targets[i_tgt]["boxes"] = scaled_boxes

    if "area" in targets[0]:  # TODO: not sure if it is needed to do for all images from the clip
        for i_tgt in range(len(targets)):  # apply for every image of the clip
            area = targets[i_tgt]["area"]
            scaled_area = area * (ratio_width * ratio_height)
            targets[i_tgt]["area"] = scaled_area

    h, w = size
    for i_tgt in range(len(targets)):  # TODO: not sure if it is needed to do for all images from the clip
        targets[i_tgt]["size"] = torch.tensor([h, w])

    if "masks" in targets[0]:
        for i_tgt in range(len(targets)):  # apply for every image of the clip
            targets[i_tgt]["masks"] = (
                interpolate(targets[i_tgt]["masks"][:, None].float(), size, mode="nearest")[:, 0] > 0.5
            )

    return rescaled_clip, targets


@TRANSFORMS.register_module()
class VideoRandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, data):
        video = data[0]
        target = None if len(data) == 1 else data[1]
        size = random.choice(self.sizes)
        rescaled_clip, targets = resize(video, target, size, self.max_size)
        return [rescaled_clip, targets]


def crop(clip, targets, region):
    cropped_clip = crop_clip(clip, *region)
    # cropped_clip = [F.crop(img, *region) for img in clip] # other possibility is to use torch_videovision.torchvideotransforms.functional.crop_clip

    targets = targets.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    for i_tgt in range(len(targets)):  # TODO: not sure if it is needed to do for all images from the clip
        targets[i_tgt]["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd", "positive_map", "isfinal"]

    if "boxes" in targets[0]:
        for i_tgt in range(len(targets)):  # apply for every image of the clip
            boxes = targets[i_tgt]["boxes"]
            max_size = torch.as_tensor([w, h], dtype=torch.float32)
            cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
            cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
            cropped_boxes = cropped_boxes.clamp(min=0)
            area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
            targets[i_tgt]["boxes"] = cropped_boxes.reshape(-1, 4)
            targets[i_tgt]["area"] = area
        fields.append("boxes")

    if "masks" in targets[0]:
        # FIXME should we update the area here if there are no boxes?
        for i_tgt in range(len(targets)):  # apply for every image of the clip
            targets[i_tgt]["masks"] = targets[i_tgt]["masks"][:, i : i + h, j : j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in targets[0] or "masks" in targets[0]:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        for i_tgt in range(len(targets)):
            if "boxes" in targets[0]:
                cropped_boxes = targets[i_tgt]["boxes"].reshape(-1, 2, 2)
                keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
            else:
                keep = targets[i_tgt]["masks"].flatten(1).any(1)

            for field in fields:
                if field in targets[i_tgt]:
                    targets[i_tgt][field] = targets[i_tgt][field][keep]
    return cropped_clip, targets


@TRANSFORMS.register_module()
class VideoRandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int, respect_boxes: bool = False):
        self.min_size = min_size
        self.max_size = max_size
        self.respect_boxes = respect_boxes  # if True we can't crop a box out

    def __call__(self, data):
        clip = data[0]
        targets = data[1]
        orig_targets = copy.deepcopy(targets)  # used to conserve ALL BOXES ANYWAY
        init_boxes = sum(len(targets[i_tgt]["boxes"]) for i_tgt in range(len(targets)))
        max_patience = 100  # TODO: maybe it is gonna requery lots of time with a clip than an image as it involves more boxes
        for i_patience in range(max_patience):
            if isinstance(clip[0], PIL.Image.Image):
                h = clip[0].height
                w = clip[0].width
            elif isinstance(clip[0], np.ndarray):
                h = clip[0].shape[0]
                w = clip[0].shape[1]
            else:
                raise NotImplementedError
            tw = random.randint(self.min_size, min(w, self.max_size))
            th = random.randint(self.min_size, min(h, self.max_size))
            # region = T.RandomCrop.get_params(clip[0], [th, tw]) # h w sizes are the same for all images of the clip; we can just get parameters for the first image

            if h + 1 < th or w + 1 < tw:
                raise ValueError(
                    "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
                )

            if w == tw and h == th:
                region = 0, 0, h, w
            else:
                i = torch.randint(0, h - th + 1, size=(1,)).item()
                j = torch.randint(0, w - tw + 1, size=(1,)).item()
                region = i, j, th, tw

            result_clip, result_targets = crop(clip, targets, region)
            if (not self.respect_boxes) or sum(
                len(result_targets[i_patience]["boxes"]) for i_patience in range(len(result_targets))
            ) == init_boxes:
                return result_clip, result_targets
            elif self.respect_boxes and i_patience == max_patience - 1:
                # avoid disappearing boxes, targets = result_targets here
                return clip, orig_targets
        return [result_clip, result_targets]


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    """
    Equivalent to nn.functional.interpolate, but with support for empty channel sizes.
    """
    if input.numel() > 0:
        return torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners)

    assert (
        input.shape[0] != 0 or input.shape[1] != 0
    ), "At least one of the two first dimensions must be non zero"

    if input.shape[1] == 0:
        # Pytorch doesn't support null dimension on the channel dimension, so we transpose to fake a null batch dim
        return torch.nn.functional.interpolate(
            input.transpose(0, 1), size, scale_factor, mode, align_corners
        ).transpose(0, 1)

    # empty batch dimension is now supported in pytorch
    return torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


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
