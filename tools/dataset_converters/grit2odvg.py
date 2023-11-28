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

import en_core_web_trf

nlp = en_core_web_trf.load()
ban_entity_labels = ["CARDINAL", "DATE", "GPE", "LANGUAGE", "ORDINAL", "PERCENT", "TIME"]

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
        # print('warning! box too small')
        return False, out_range
    else:
        return True, out_range


def get_regions(nc, anno):
    h = anno["height"]
    w = anno["width"]
    out_ranges = 0
    if isinstance(nc[0], list):
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
    return {"phrase": phrase, "bbox": bbox}, out_ranges
    # return {"phrase": phrase, "tokens_positive": index, "bbox": bbox}, out_ranges
    # return {"phrase": phrase}, out_ranges


def check_multi_boxes(noun_chunks):
    noun_list = [i[0:2] for i in noun_chunks]
    exist_now = []
    repeat_noun = []
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
                    repeat_annos = [anno]
                    for j in range(i + 1, len(noun_chunks)):
                        if noun_chunks[j][0:2] == noun:
                            repeat_annos.append(noun_chunks[j])
                    repeat_noun.append(repeat_annos)
                    if num_repeat == len(noun_chunks):
                        break

                elif num_repeat == 1:
                    repeat_noun.append(anno)
    return repeat_noun


def process_json(js, args):
    odvg = []
    caption_false = 0
    box_small = 0
    out_ranges = {'noun': 0, 'ref': 0}
    few_caption = 0
    noun_same_with_ref = 0
    filter_informs = []
    if os.path.isdir(args.input_files):
        js = os.path.join(args.input_files, js)
        # print(js)
    with open(js, "r") as f:
        metas = json.load(f)
        metas_len = len(metas)

        for meta in metas:
            result, outrange_or_fewcaption, noun_ref, filter_inform = process_item(meta, args)
            noun_same_with_ref += noun_ref
            if filter_inform != []:
                filter_informs.append(filter_inform)
            if result is not None:
                if len(result['referring']) != 0:
                    odvg.append(result)
                else:
                    box_small += 1
                out_ranges['noun'] += outrange_or_fewcaption['noun']
                out_ranges['ref'] += outrange_or_fewcaption['ref']
            elif outrange_or_fewcaption is None:
                caption_false += 1
            elif outrange_or_fewcaption is True:
                few_caption += 1

    # print(
    #     f"images num in {js} = {metas_len}, after filter: {len(odvg)}, num of existing boxes out of range: {out_ranges}"
    # )
    # print(f"caption false: {caption_false}, box too small: {box_small}")
    odvg = list(filter(None, odvg))
    # print(noun_same_with_ref)
    return [odvg, js, metas_len, len(odvg), out_ranges, caption_false, box_small, few_caption, filter_informs]


def process_item(file, args):
    annotation = file
    # 若caption有除了ASCII文本以外的其他元素（.*/-emoji[CLS]等)，返回None
    if not check_caption(annotation):
        return None, None, 0, []
    noun_chunks = annotation['noun_chunks']
    ref_exps = annotation['ref_exps']
    regions = []
    noun_same_with_ref = 0
    random_num = random.random()
    out_ranges = {'noun': 0, 'ref': 0}

    # judge same noun with multi boxes
    repeat_noun = check_multi_boxes(noun_chunks)
    repeat_ref = check_multi_boxes(ref_exps)
    # get regions
    nouns = [i[0:2] if not isinstance(i[0], list) else i[0][0:2] for i in repeat_noun]
    nouns_box = [i[2:6] for i in noun_chunks]
    refs = [i[0:2] if not isinstance(i[0], list) else i[0][0:2] for i in repeat_ref]

    for nc in repeat_noun:
        region, out_range = get_regions(nc, annotation)
        if region is not None:
            if str.isascii(region["phrase"]):
                regions.append(region)
            out_ranges['noun'] += out_range

    if nouns != refs:
        for nc in repeat_ref:
            if not isinstance(nc[0], list):  # 不是一文对多框
                if nc[0:2] not in nouns:  # 若ref不在noun中， 若ref在noun中说明anno相同无需操作
                    if nc[2:6] not in nouns_box:  # ref不在noun中且box也不同，则当做一个全新anno处理
                        region, out_range = get_regions(nc, annotation)
                        if region is not None:
                            if str.isascii(region["phrase"]):
                                regions.append(region)
                    else:  # ref不在noun中但box相同，则将box和ref添加到一个列表里
                        region, out_range = get_regions(nc, annotation)
                        if region is not None:
                            for i in range(len(regions)):
                                if regions[i]['bbox'] == region['bbox']:
                                    regions[i]['phrase'] = [regions[i]['phrase'], region['phrase']]
                                    break
            else:  # 一文对多框
                if nc[0][0:2] not in nouns:  # 若ref不在noun中，若ref在noun中说明anno相同无需操作
                    flag = 0
                    for n in nc:
                        if n[2:6] in nouns_box:  # 若box存在，则则将box和ref添加到一个列表里
                            flag = 1
                            region, out_range = get_regions(nc, annotation)
                            if region is not None:
                                for i in range(len(regions)):
                                    if regions[i]['bbox'] == region['bbox']:
                                        regions[i]['phrase'] = [regions[i]['phrase'], region['phrase']]
                                        break
                            break
                    if flag == 0:  # ref不在noun中且box也不同，则当做一个全新anno处理
                        region, out_range = get_regions(nc, annotation)
                        if region is not None:
                            if str.isascii(region["phrase"]):
                                regions.append(region)

    if len(regions) < args.min_phrase:
        return None, True, 0, []

    odvg_anno = None

    # nlp filter
    filter_flag = 0
    doc = nlp(annotation['caption'])
    entities = {ent.text: ent.label_ for ent in doc.ents if ent.label_ in ban_entity_labels}
    entities_text = entities.keys()
    regions_after_filter = []
    for region in regions:
        if isinstance(region['phrase'], list):
            new_phrase_list = [x for x in region['phrase'] if x not in entities_text and len(x) > 1]
            if len(new_phrase_list) != len(region['phrase']):
                filter_flag = 1
            if len(new_phrase_list) == 1:
                new_phrase_list = new_phrase_list[0]
            region['phrase'] = new_phrase_list
            regions_after_filter.append(region)
        else:
            if region['phrase'] not in entities_text and len(region['phrase']) > 1:
                regions_after_filter.append(region)
    if len(regions_after_filter) != len(regions):
        filter_flag = 1
    # noun_same_with_ref = 1
    odvg_anno = {
        "filename": file['key'][0:5] + '/' + file['key'] + '.jpg',
        "height": annotation["height"],
        "width": annotation["width"],
        # "referring": {"caption": clean_span(annotation["caption"]), "regions": regions},
        "referring": regions_after_filter,
    }
    filter_inform = []
    if filter_flag == 1:
        if entities_text == None:
            entities_text = []
        filter_inform = [odvg_anno['filename'], entities_text]
    return odvg_anno, out_ranges, noun_same_with_ref, filter_inform


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRIT2ODVG List.")
    parser.add_argument("input_files", type=str, help='input json file ')  # jsons directory
    parser.add_argument("json_name", type=str, help='output json file name')
    parser.add_argument('--ban_json', '-b', type=str, default=None, help='ban json file')
    parser.add_argument("--random_samples", type=int, default=2000)
    parser.add_argument("--min_phrase", "-m", type=int, default=3)
    parser.add_argument("--process_num", type=int, default=4, help="the number of processes")
    args = parser.parse_args()
    print(args)

    # process jsons directory
    if os.path.isdir(args.input_files):
        odvg_anno = []
        jsons = [i for i in os.listdir(args.input_files) if i.endswith('.json')]
        func = partial(process_json, args=args)
        # with Pool(processes=args.process_num) as pool:
        #     for result in tqdm(pool.imap(func=func, iterable=jsons), total=len(jsons)):
        #         odvg_anno.extend(result[0])
        #         with open('grit.log', 'a') as f:
        #             f.write(
        #                 f"images num in {result[1]} = {result[2]}, after filter: {result[3]}, num of existing boxes out of range: {result[4]}, caption false: {result[5]}, box too small: {result[6]}, few_caption: {result[7]}\n"
        #             )
        for js in tqdm(jsons, total=len(jsons)):
            result = func(js)
            odvg_anno.extend(result[0])
            with open('grit_filter.log', 'a') as f:
                if len(result[8]) != 0:
                    try:
                        for re in result[8]:
                            f.write(f"{re[0]}, ban: {re[1]}\n")
                    except:
                        print(result[8])
                # f.write(
                #     f"images num in {result[1]} = {result[2]}, after filter: {result[3]}, num of existing boxes out of range: {result[4]}, caption false: {result[5]}, box too small: {result[6]}, few_caption: {result[7]}\n"
                # )

    else:
        result = process_json(args.input_files, args)
        odvg_anno = result[0]
        print(
            f"images num in {result[1]} = {result[2]}, after filter: {result[3]}, num of existing boxes out of range: {result[4]}, caption false: {result[5]}, box too small: {result[6]}, few_caption: {result[7]}"
        )
    # process single json file
    print(len(odvg_anno))
    with jsonlines.open(args.json_name, mode="w") as fwriter:
        fwriter.write_all(odvg_anno)
