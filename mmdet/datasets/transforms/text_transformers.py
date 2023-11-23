# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.transforms import BaseTransform

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import BaseBoxes

try:
    from transformers import AutoTokenizer
    from transformers import BertModel as HFBertModel
except ImportError:
    AutoTokenizer = None
    HFBertModel = None

import random
import re

import numpy as np


def clean_name(name):
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = re.sub(r'  ', ' ', name)
    name = name.lower()
    return name


def check_for_positive_overflow(gt_bboxes, gt_labels, text, tokenizer, max_tokens):
    # NOTE: Only call this function for OD/REF data; DO NOT USE IT FOR GROUNDING DATA
    # NOTE: called only in coco_dt
    kept_lables = []
    length = 0
    # Check if we have too many positive labels
    # generate a caption by appending the positive labels
    positive_label_list = np.unique(gt_labels).tolist()
    # random shuffule so we can sample different annotations at different epochs
    random.shuffle(positive_label_list)

    for index, label in enumerate(positive_label_list):
        label_text = clean_name(text[str(label)]) + '. '

        tokenized = tokenizer.tokenize(label_text)

        length += len(tokenized)

        if length > max_tokens:
            break
        else:
            kept_lables.append(label)

    keep_box_index = []
    keep_gt_labels = []
    for i in range(len(gt_labels)):
        if gt_labels[i] in kept_lables:
            keep_box_index.append(i)
            keep_gt_labels.append(gt_labels[i])

    return gt_bboxes[keep_box_index], np.array(keep_gt_labels, dtype=np.longlong), length


def generate_senetence_given_labels(positive_label_list, negative_label_list, text):
    label_to_positions = {}

    label_list = negative_label_list + positive_label_list

    random.shuffle(label_list)

    pheso_caption = ''

    label_remap_dict = {}
    for index, label in enumerate(label_list):
        start_index = len(pheso_caption)

        pheso_caption += clean_name(text[str(label)])

        end_index = len(pheso_caption)

        if label in positive_label_list:
            label_to_positions[index] = [[start_index, end_index]]
            label_remap_dict[int(label)] = index

        # if index != len(label_list) - 1:
        #     pheso_caption += '. '
        pheso_caption += '. '

    return label_to_positions, pheso_caption, label_remap_dict


def find_substring_indices(string, substring):
    pattern = re.escape(substring)
    matches = re.finditer(pattern, string)
    indices = [(match.start(), match.end()) for match in matches]
    return indices


@TRANSFORMS.register_module()
class RandomSamplingNegPos(BaseTransform):
    def __init__(
        self, tokenizer_name, num_sample_negative=85, max_tokens=256, full_sampling_prob=0.5, num_ref_neg=3
    ):
        if AutoTokenizer is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: ' 'pip install transformers.'
            )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.num_sample_negative = num_sample_negative
        self.full_sampling_prob = full_sampling_prob
        self.max_tokens = max_tokens
        self.num_ref_neg = num_ref_neg

    def transform(self, results: dict) -> dict:
        # if 'phrases' in results:
        #     return self.vg_aug(results)
        # else:
        #     return self.od_aug(results)
        if results['dataset_mode'] == 'OD':
            return self.od_aug(results)
        elif results['dataset_mode'] == 'VG':
            return self.vg_aug(results)
        elif results['dataset_mode'] == 'REF':
            return self.ref_aug(results)

    def vg_aug(self, results):
        gt_bboxes = results['gt_bboxes']
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor
        gt_labels = results['gt_bboxes_labels']
        text = results['text'].lower().strip()
        if not text.endswith('.'):
            text = text + '. '

        phrases = results['phrases']
        # TODO: add neg
        positive_label_list = np.unique(gt_labels).tolist()
        label_to_positions = {}
        for label in positive_label_list:
            label_to_positions[label] = phrases[label]['tokens_positive']

        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_labels'] = gt_labels

        results['text'] = text
        results['tokens_positive'] = label_to_positions
        return results

    def od_aug(self, results):
        gt_bboxes = results['gt_bboxes']
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor
        gt_labels = results['gt_bboxes_labels']
        text = results['text']

        original_box_num = len(gt_labels)
        # If the category name is in the format of 'a/b' (in object365),
        # we randomly select one of them.
        for key, value in text.items():
            if '/' in value:
                text[key] = random.choice(value.split('/')).strip()

        gt_bboxes, gt_labels, positive_caption_length = check_for_positive_overflow(
            gt_bboxes, gt_labels, text, self.tokenizer, self.max_tokens
        )

        if len(gt_bboxes) < original_box_num:
            print(
                'WARNING: removed {} boxes due to positive caption overflow'.format(
                    original_box_num - len(gt_bboxes)
                )
            )

        valid_negative_indexes = list(text.keys())

        positive_label_list = np.unique(gt_labels).tolist()
        full_negative = self.num_sample_negative

        if full_negative > len(valid_negative_indexes):
            full_negative = len(valid_negative_indexes)

        outer_prob = random.random()

        if outer_prob < self.full_sampling_prob:
            # c. probability_full: add both all positive and all negatives
            num_negatives = full_negative
        else:
            if random.random() < 1.0:
                num_negatives = np.random.choice(max(1, full_negative)) + 1
            else:
                num_negatives = full_negative

        # Keep some negatives
        negative_label_list = set()
        if num_negatives != -1:
            if num_negatives > len(valid_negative_indexes):
                num_negatives = len(valid_negative_indexes)

            for i in np.random.choice(valid_negative_indexes, size=num_negatives, replace=False):
                if i not in positive_label_list:
                    negative_label_list.add(i)

        random.shuffle(positive_label_list)

        negative_label_list = list(negative_label_list)
        random.shuffle(negative_label_list)

        negative_max_length = self.max_tokens - positive_caption_length
        screened_negative_label_list = []

        for negative_label in negative_label_list:
            label_text = clean_name(text[str(negative_label)]) + '. '

            tokenized = self.tokenizer.tokenize(label_text)

            negative_max_length -= len(tokenized)

            if negative_max_length > 0:
                screened_negative_label_list.append(negative_label)
            else:
                break
        negative_label_list = screened_negative_label_list
        label_to_positions, pheso_caption, label_remap_dict = generate_senetence_given_labels(
            positive_label_list, negative_label_list, text
        )

        # label remap
        if len(gt_labels) > 0:
            gt_labels = np.vectorize(lambda x: label_remap_dict[x])(gt_labels)

        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_labels'] = gt_labels

        results['text'] = pheso_caption
        results['tokens_positive'] = label_to_positions

        return results

    def ref_aug(self, results):
        gt_bboxes = results['gt_bboxes']
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor
        gt_labels = results['gt_bboxes_labels']
        text = results['text']

        original_box_num = len(gt_labels)

        gt_bboxes, gt_labels, positive_caption_length = check_for_positive_overflow(
            gt_bboxes, gt_labels, text, self.tokenizer, self.max_tokens
        )

        if len(gt_bboxes) < original_box_num:
            print(
                'WARNING: removed {} boxes due to positive caption overflow'.format(
                    original_box_num - len(gt_bboxes)
                )
            )

        valid_negative_indexes = list(text.keys())

        positive_label_list = np.unique(gt_labels).tolist()
        positive_phrase_list = [text[str(positive_label_list[i])] for i in range(len(positive_label_list))]
        full_negative = self.num_ref_neg

        if full_negative > len(valid_negative_indexes):
            full_negative = len(valid_negative_indexes)

        outer_prob = random.random()

        if outer_prob < self.full_sampling_prob:
            # c. probability_full: add both all positive and all negatives
            num_negatives = full_negative
        else:
            if random.random() < 1.0:
                num_negatives = np.random.choice(max(1, full_negative)) + 1
            else:
                num_negatives = full_negative

        # Keep some negatives
        negative_phrase_list = set()
        if num_negatives != -1:
            if num_negatives > len(valid_negative_indexes):
                num_negatives = len(valid_negative_indexes)

            assert 'dataset' in results
            dataset = results.pop('dataset', None)
            neg_list = random.sample(dataset.ref_phrases, len(positive_label_list) + num_negatives)
            for i in range(len(neg_list)):
                if i >= num_negatives:
                    break
                neg = neg_list[i]
                if neg not in positive_phrase_list:
                    negative_phrase_list.add(neg)

        # negative_label_list = list(negative_label_list)
        # random.shuffle(negative_label_list)
        negative_phrase_list = list(negative_phrase_list)
        random.shuffle(negative_phrase_list)

        negative_max_length = self.max_tokens - positive_caption_length
        screened_negative_phrase_list = []

        for negative_phrase in negative_phrase_list:
            label_text = clean_name(negative_phrase) + '. '

            tokenized = self.tokenizer.tokenize(label_text)

            negative_max_length -= len(tokenized)

            if negative_max_length > 0:
                screened_negative_phrase_list.append(negative_phrase)
            else:
                break
        negative_phrase_list = screened_negative_phrase_list

        all_list = positive_phrase_list + negative_phrase_list
        all_dict = {str(i): all_list[i] for i in range(len(all_list))}
        positive_label_list = [i for i in range(len(positive_phrase_list))]
        negative_label_list = [i for i in range(len(positive_phrase_list), len(all_list))]
        # negative_label_list = screened_negative_label_list
        try:
            label_to_positions, pheso_caption, label_remap_dict = generate_senetence_given_labels(
                positive_label_list, negative_label_list, all_dict
            )
        except:
            print('error!!')
            print(positive_label_list, negative_label_list, all_dict)

        # label remap
        gt_to_positive = []
        for i in range(len(gt_labels)):
            try:
                gt_to_positive.append(positive_phrase_list.index(text[str(gt_labels[i])]))
            except:
                print('gt_to_positive error!!')
                print(gt_labels, positive_phrase_list, text)

        if len(gt_labels) > 0:
            gt_labels = np.vectorize(lambda x: label_remap_dict[x])(gt_to_positive)

        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_labels'] = gt_labels

        results['text'] = pheso_caption
        results['tokens_positive'] = label_to_positions

        return results
