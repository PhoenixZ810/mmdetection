import json
import os
import requests
from urllib.parse import urlparse
from requests.exceptions import HTTPError
from tqdm import tqdm

import sys
from pathlib import Path
import textwrap

import ast
import os
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

pylab.rcParams['figure.figsize'] = 20, 12

import cv2
import base64
import io
import argparse
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar

# from mmdet.models.utils import mask2ndarray
from mmdet.registry import DATASETS, VISUALIZERS
# from mmdet.structures.bbox import BaseBoxes
from mmdet.apis import VideoInferencer


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '-o',
        '--output-dir',
        default='debug/visual_res',
        type=str,
        help='If there is no display interface, you can save it',
    )
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument('--show-interval', type=float, default=2, help='the interval of show (s)')
    parser.add_argument('--i_per_v', type=int, default=0, help='show images per video')
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
    parser.add_argument('--sample', '-s', type=int, default=30, help='sample number')

    # parser.add_argument(
    #     '--model',
    #     type=str,
    #     default=None,
    #     help='Config or checkpoint .pth file or the model name '
    #     'and alias defined in metafile. The model configuration '
    #     'file will try to read from .pth if the parameter is '
    #     'a .pth weights file.',
    # )
    parser.add_argument('--weights', default=None, help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # register all modules in mmdet into the registries
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    visualizer = video_visualizer()
    if args.weights is not None:
        inferencer = VideoInferencer(model=args.config, weights=args.weights, device=args.device)
        for item in dataset:
            predicts = inferencer(item)
            visualizer.vis_image(item=item, args=args, predicts=predicts)
    else:
        try:
            for item in tqdm(dataset, total=len(dataset)):
                pass
        except:
            print(f"Error in dataset:{item['data_samples'].video_path}")
            # raise
        # for item in dataset:
        #     visualizer.vis_image(item=item, args=args)


class video_visualizer:
    def __init__(self):
        pass

    def vis_image(self, item, args, predicts=None):
        # img_path = osp.basename(item['data_samples'].video_path)

        # out_file = (
        #     osp.join(args.output_dir, osp.basename(img_path).split('.')[0])
        #     if args.output_dir is not None
        #     else None
        # )
        if predicts is not None:
            pred_boxes = [pred.pred_instances.bboxes for pred in predicts]
            pred_scores = [pred.pred_instances.scores for pred in predicts]
            # pred_sted = [pred.pred_instances.sted for pred in predicts]

            if item['inputs'].ndim == 4:
                if args.i_per_v != 0:
                    i_per_v = args.i_per_v
                else:
                    i_per_v = item['inputs'].shape[0]
                data_samples = item['data_samples']
                # gt_instances = data_sample.gt_instances
                # gt_bboxes = gt_instances.get('bboxes', None)
                gt_bboxes = [
                    datasample.gt_instances.get('bboxes', None).numpy() for datasample in data_samples
                ]
                for i in range(i_per_v):
                    if gt_bboxes[i].shape[0] != 0:
                        img = item['inputs'].permute(0, 2, 3, 1).numpy()[i]  # from bchw to bhwc
                        # img = Image.fromarray(img)
                        img = img[..., [2, 1, 0]]  # bgr to rgb
                        phrase = item['data_samples'][0].text
                        image_h = img.shape[0]
                        image_w = img.shape[1]
                        new_image = img.copy()
                        img_name = item['data_samples'][i].img_in_vid_ids

                        file_key_name = img_name + '.jpg'
                        # plt.imshow(img[..., [2, 1, 0]])
                        # plt.axis('off')
                        # plt.savefig(file_key_name,bbox_inches='tight', pad_inches=0)

                        previous_locations = []
                        previous_bboxes = []
                        text_offset = 10
                        text_offset_original = 4
                        gt_bbox = gt_bboxes[i].squeeze(0)
                        pred_bbox = pred_boxes[i].squeeze(0)
                        (
                            new_image,
                            previous_locations,
                            previous_bboxes,
                            text_offset,
                            text_offset_original,
                        ) = self.visual_per_box(
                            new_image,
                            img_name,
                            f'GT',
                            image_h,
                            image_w,
                            gt_bbox,
                            previous_locations,
                            previous_bboxes,
                            text_offset,
                            text_offset_original,
                            args,
                            caption=phrase,
                        )
                        (
                            new_image,
                            previous_locations,
                            previous_bboxes,
                            text_offset,
                            text_offset_original,
                        ) = self.visual_per_box(
                            new_image,
                            img_name,
                            f'Pred  {float(pred_scores[i][0])}',
                            image_h,
                            image_w,
                            pred_bbox,
                            previous_locations,
                            previous_bboxes,
                            text_offset,
                            text_offset_original,
                            args,
                            caption=phrase,
                        )
                    else:
                        img = item['inputs'].permute(0, 2, 3, 1).numpy()[i]  # from bchw to bhwc
                        # img = Image.fromarray(img)
                        img = img[..., [2, 1, 0]]  # bgr to rgb
                        phrase = item['data_samples'][0].text
                        image_h = img.shape[0]
                        image_w = img.shape[1]
                        new_image = img.copy()
                        img_name = item['data_samples'][i].img_in_vid_ids

                        file_key_name = img_name + '.jpg'
                        # plt.imshow(img[..., [2, 1, 0]])
                        # plt.axis('off')
                        # plt.savefig(file_key_name,bbox_inches='tight', pad_inches=0)
                        previous_locations = []
                        previous_bboxes = []
                        text_offset = 10
                        text_offset_original = 4
                        file_key_name = img_name + '.jpg'
                        output_path = os.path.join(args.output_dir, file_key_name)
                        self.imshow(new_image, file_name=output_path, caption=phrase)

        else:
            if item['inputs'].ndim == 4:
                if args.i_per_v != 0:
                    i_per_v = args.i_per_v
                else:
                    i_per_v = item['inputs'].shape[1]
                for i in range(i_per_v):
                    data_samples = item['data_samples']
                    gt_instances = data_samples.gt_instances
                    gt_bboxes = gt_instances.get('bboxes', None)
                    if gt_bboxes[i].shape[0] != 0:
                        img = item['inputs'].permute(1, 2, 3, 0).numpy()[i]  # from cbhw to bhwc
                        # img = Image.fromarray(img)
                        img = img[..., [2, 1, 0]]  # bgr to rgb
                        phrase = item['data_samples'].text
                        image_h = img.shape[0]
                        image_w = img.shape[1]
                        new_image = img.copy()
                        img_name = item['data_samples'].img_in_vid_ids[i]
                        previous_locations = []
                        previous_bboxes = []
                        text_offset = 10
                        text_offset_original = 4
                        (
                            new_image,
                            previous_locations,
                            previous_bboxes,
                            text_offset,
                            text_offset_original,
                        ) = self.visual_per_box(
                            new_image,
                            img_name,
                            f'GT',
                            image_h,
                            image_w,
                            gt_bboxes[i].squeeze(0),
                            previous_locations,
                            previous_bboxes,
                            text_offset,
                            text_offset_original,
                            args,
                            caption=phrase,
                        )

    def is_overlapping(self, rect1, rect2):
        x1, y1, x2, y2 = rect1
        x3, y3, x4, y4 = rect2
        return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)

    def visual_per_box(
        self,
        new_image,
        img_name,
        phrase,
        image_h,
        image_w,
        bbox,
        previous_locations,
        previous_bboxes,
        text_offset,
        text_offset_original,
        args,
        caption=None,
    ):
        text_size = max(0.07 * min(image_h, image_w) / 100, 0.5)
        text_line = int(max(1 * min(image_h, image_w) / 512, 1))
        box_line = int(max(2 * min(image_h, image_w) / 512, 2))
        text_height = text_offset  # init
        x1, y1, x2, y2 = (
            int(bbox[0]),
            int(bbox[1]),
            int(bbox[2]),
            int(bbox[3]),
        )
        # print(f"Decode results: {phrase} - {[x1, y1, x2, y2]}")
        # draw bbox
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        new_image = cv2.rectangle(new_image, (x1, y1), (x2, y2), color, box_line)

        # add phrase name
        # decide the text location first
        for x_prev, y_prev in previous_locations:
            if abs(x1 - x_prev) < abs(text_offset) and abs(y1 - y_prev) < abs(text_offset):
                y1 += text_height

        if y1 < 2 * text_offset:
            y1 += text_offset + text_offset_original

        # add text background
        (text_width, text_height), _ = cv2.getTextSize(phrase, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_line)
        text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2 = (
            x1,
            y1 - text_height - text_offset_original,
            x1 + text_width,
            y1,
        )

        for prev_bbox in previous_bboxes:
            while self.is_overlapping((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox):
                text_bg_y1 += text_offset
                text_bg_y2 += text_offset
                y1 += text_offset

                if text_bg_y2 >= image_h:
                    text_bg_y1 = max(0, image_h - text_height - text_offset_original)
                    text_bg_y2 = image_h
                    y1 = max(0, image_h - text_height - text_offset_original + text_offset)
                    break

        alpha = 0.5
        for i in range(text_bg_y1, text_bg_y2):
            for j in range(text_bg_x1, text_bg_x2):
                if i < image_h and j < image_w:
                    new_image[i, j] = (alpha * new_image[i, j] + (1 - alpha) * np.array(color)).astype(
                        np.uint8
                    )

        new_image = cv2.putText(
            new_image,
            phrase,
            (x1, y1 - text_offset_original),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            (0, 0, 0),
            text_line,
            cv2.LINE_AA,
        )
        previous_locations.append((x1, y1))
        previous_bboxes.append((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2))

        file_key_name = img_name + '.jpg'
        output_path = os.path.join(args.output_dir, file_key_name)
        self.imshow(new_image, file_name=output_path, caption=caption)
        return (
            new_image,
            previous_locations,
            previous_bboxes,
            text_offset,
            text_offset_original,
        )

    def show_images_from_json(self, args):
        with open(args.json, 'r') as json_file:
            j = json.load(json_file)
            for json_obj in j:
                self.vis_image(json_obj, args)

    def show_images_from_jsonl(self, args):
        with open(args.json, 'r') as jsonl_file:
            for j in jsonl_file:
                json_obj = json.loads(j)
                self.vis_image(json_obj, args)

    def download_image(url, output_folder):
        try:
            response = requests.get(url)
            response.raise_for_status()
        except HTTPError as e:
            print(f"Error while downloading {url}: {e}")
            return

        file_name = os.path.basename(urlparse(url).path)
        output_path = os.path.join(output_folder, file_name)

        with open(output_path, 'wb') as file:
            file.write(response.content)

    def imshow(self, img, file_name="tmp.jpg", caption='test'):
        # Create figure and axis objects
        fig, ax = plt.subplots()
        # Show image on axis
        ax.imshow(img[:, :, [2, 1, 0]])
        ax.set_axis_off()
        # Set caption text
        # Add caption below image
        ax.text(
            0.5,
            -0.2,
            '\n'.join(textwrap.wrap(caption, 120)),
            ha='center',
            transform=ax.transAxes,
            fontsize=18,
        )
        plt.savefig(file_name, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    # import pdb;pdb.set_trace()
    main()
    # you need to download the jsonl before run this file

    # if not os.path.exists(args.output):
    #     os.makedirs(args.output)
    # if args.json.endswith('.json'):
    #     show_images_from_json(args)
    # elif args.json.endswith('.jsonl'):
    #     show_images_from_jsonl(args)
