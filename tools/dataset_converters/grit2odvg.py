import jsonlines
from tqdm import tqdm
import random
import json
import os
from multiprocessing import Pool
from functools import partial
import emoji
import numpy as np

import argparse

'''对caption中的标点符号进行清洗，中文标点转为英文标点，去掉句末的标点'''


def clean_span(span):
    span = span.rstrip()
    span = span.replace('"', "'").replace('\"', "'").replace('“', "'").replace('”', "'")
    span = span.replace('‘', "'").replace('’', "'").replace('–', "—")
    if span.endswith('/') or span.endswith('.'):
        span = span[:-1]
    return span


'''检查grit caption中是否有除了ASCII文本以外的其他元素（如.*/-emoji[CLS]等)，如果有返回False'''


def check_caption(cap):
    check_anno = cap["caption"].rstrip()[:-1]
    if not str.isascii(check_anno):
        return False
    # "The view is better from here 🦅 (Chouf" wtf??
    # check_list = {"↙️", "-", ",", " ", "*", "/", "$", "[CLS]", "[SEP]", "?"}
    check_list = {"↙️", " ", "*", "/", "$", "[CLS]", "[SEP]", "?"}
    for ch in check_list:
        if ch in check_anno:
            return False
    if '.' in check_anno[:-1]:
        return False
    if emoji.emoji_count(check_anno):
        print(check_anno)
        return False
    return True


def check_box(box_list, w, h):
    # import pdb;pdb.set_trace()
    box_w = box_list[2] - box_list[0]
    box_h = box_list[3] - box_list[1]
    out_range = False
    if not all(0 <= box_list[i] <= w - 1 if i % 2 == 0 else 0 <= box_list[i] <= h - 1 for i in range(4)):
        # print('warning! box out of range')
        box_list[0] = max(box_list[0], 0)
        box_list[1] = max(box_list[1], 0)
        box_list[2] = min(box_list[2], w - 1)
        box_list[3] = min(box_list[3], h - 1)
        out_range = True
    if box_w < 1 or box_h < 1:
        print('warning! box too small')
        return False, out_range
    else:
        return True, out_range


def get_regions(nc, anno):
    h = anno["height"]
    w = anno["width"]
    out_ranges = 0
    if not isinstance(nc, list):
        phrase = clean_span(anno["caption"][int(nc[0][0]) : int(nc[0][1])])
        index = [[int(nc[0][0]), int(nc[0][1])]]
        bbox = []
        for noun in nc:
            box = [round(noun[2] * w, 3), round(noun[3] * h, 3), round(noun[4] * w, 3), round(noun[5] * h, 3)]
            box_small, out_range = check_box(box, w, h)
            out_ranges += 1 if out_range == True else 0
            if not box_small:
                return None, out_ranges
            else:
                bbox.append(box)
    else:
        phrase = clean_span(anno["caption"][int(nc[0]) : int(nc[1])])
        index = [[int(nc[0]), int(nc[1])]]
        bbox = [round(nc[2] * w, 3), round(nc[3] * h, 3), round(nc[4] * w, 3), round(nc[5] * h, 3)]
        box_small, out_range = check_box(bbox, w, h)
        out_ranges += 1 if out_range == True else 0
        if not box_small:
            return None, out_ranges

    return {"phrase": phrase, "tokens_positive": index, "bbox": bbox}, out_ranges


# '''从所有图片中选择random_sampled张图片'''
# def prepare_list(file_name: str, random_samples):
#     with open(file_name, "r") as f:
#         # metas = [line.strip() for line in f]
#         metas = json.load(f)
#     num_of_files = len(metas)
#     print(num_of_files)
#     metas = random.sample(metas, random_samples)
#     num_of_files = len(metas)
#     print("after sample:", num_of_files)
#     return metas, num_of_files


def process_json(js, args):
    odvg = []
    caption_false = 0
    box_small = 0
    out_ranges = 0
    if os.path.isdir(args.input_files):
        js = os.path.join(args.input_files, js)
    with open(js, "r") as f:
        metas = json.load(f)
        metas_len = len(metas)

    for meta in metas:
        result, out_range_per_image = process_item(meta, args)
        if result is not None:
            if len(result['grounding']['regions']) != 0:
                odvg.append(result)
            else:
                box_small += 1
            out_ranges += out_range_per_image
        else:
            caption_false += 1
    print(
        f"images num in {js} = {metas_len}, after filter: {len(odvg)}, num of existing boxes out of range: {out_ranges}"
    )
    print(f"caption false: {caption_false}, box too small: {box_small}")
    odvg = list(filter(None, odvg))
    return odvg


def process_item(file, args):
    # file_dir = file[0:5]
    # json_name = file_dir + '.json'
    # with open(os.path.join(os.path.dirname(args.root), 'annotations/', json_name)) as f:
    #     anno = json.load(f)
    # for i in anno:
    #     if i['key'] == os.path.splitext(file)[0]:
    #         anno = i
    #         break
    annotation = file
    # 若caption有除了ASCII文本以外的其他元素（.*/-emoji[CLS]等)，返回None
    if not check_caption(annotation):
        return None, None
    noun_chunks = annotation['noun_chunks']
    ref_exps = annotation['ref_exps']
    regions = []
    random_num = random.random()
    out_ranges = 0

    # if annotation['clip_similarity_vitb32'] < 0.3:
    #     return None
    noun_list = [i[0:2] for i in noun_chunks]
    repeat_noun = []
    exist_now = []
    for i in range(len(noun_chunks)):
        anno = noun_chunks[i]
        noun = anno[0:2]
        if len(noun_chunks) == 1:
            repeat_noun.append(anno)
        else:
            if noun not in exist_now:
                exist_now.append(noun)
                num_repeat = noun_list.count(noun)

                if num_repeat > 1:
                    repeat_annos = [np.array(anno)]
                    for j in range(i + 1, len(noun_chunks)):
                        if noun_chunks[j][0:2] == noun:
                            repeat_annos.append(np.array(noun_chunks[j]))
                    repeat_noun.append(np.stack(repeat_annos, axis=0))
                    if num_repeat == len(noun_chunks):
                        break

                elif num_repeat == 1:
                    repeat_noun.append(anno)
    for nc in repeat_noun:
        region, out_range = get_regions(nc, annotation)
        if region is not None:
            if str.isascii(region["phrase"]):
                regions.append(region)
            out_ranges += out_range
    # for re in ref_exps:
    #     region = get_regions(re, anno)
    #     if str.isascii(region["phrase"]):
    #         regions.append(region)
    # if len(regions) < args.min_phrase:
    #     return None

    odvg_anno = {
        "filename": file['key'][0:5] + '/' + file['key'] + '.jpg',
        "height": annotation["height"],
        "width": annotation["width"],
        "grounding": {"caption": clean_span(annotation["caption"]), "regions": regions},
    }
    return odvg_anno, out_ranges


if __name__ == "__main__":
    # jsons = "/share_data/mllm/kosmos-2/GRIT-20M/anno/14m_anno.list"
    # root = "/share_data/mllm/kosmos-2/GRIT-20M/data"
    # output_name = "./girt_14m_odvg.jsonl"
    parser = argparse.ArgumentParser(description="GRIT2ODVG List.")
    parser.add_argument("input_files", type=str)  # jsons directory
    parser.add_argument("json_name", type=str)
    parser.add_argument("--root", type=str, default="", help="Source image root")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--random_samples", type=int, default=2000)
    parser.add_argument("--chunk_or_ref", type=float, default=0.5)
    parser.add_argument("--min_phrase", type=int, default=6)
    parser.add_argument("--process_num", type=int, default=32, help="the number of processes")
    args = parser.parse_args()
    print(args)

    # # random sample
    # metas, metas_len = prepare_list(args.input_file, args.random_samples)

    # process single json file
    odvg_anno = process_json(args.input_files, args)

    # process jsons directory
    # odvg_anno = []
    # jsons = [i for i in os.listdir(args.input_files) if i.endswith('.json')]
    # func = partial(process_json, args=args)
    # with Pool(processes=args.process_num) as pool:
    #     for result in tqdm(pool.imap(func=func, iterable=jsons), total=len(jsons)):
    #         odvg_anno.extend(result)

    # print(f"after filter: {len(odvg_anno)}")
    json_name = os.path.join(args.output_dir, args.json_name)
    with jsonlines.open(json_name, mode="w") as fwriter:
        fwriter.write_all(odvg_anno)
