import jsonlines
from tqdm import tqdm
import random
import json
import os
from multiprocessing import Pool
from functools import partial
import emoji
import numpy as np
import time
import argparse


def multi_check_jsonl(input, ban_dict):
    num = 0
    odvg_annos = []
    num_entity = 0
    start_time = time.time()
    with open(input, 'r') as f:
        fun = partial(process_jsonl, ban_dict=ban_dict)
        num_jsonl = len(f.readlines())
        f.seek(0)
        with Pool(args.process) as pool:
            iters = pool.imap(func=fun, iterable=f)
            for iter in tqdm(iters, total=num_jsonl):
                if iter is not None:
                    odvg_annos.extend(iter)
                    num_entity += 1
    print('\nnum_entity:', num_entity)
    end_time = time.time()
    print("程序运行时间：", end_time - start_time, "秒")


def process_jsonl(js, ban_dict):
    odvg = []
    flag = 0
    anno = json.loads(js)
    for i in anno['referring']:
        for j in ban_dict.keys():
            phrase = i['phrase'] if not isinstance(i, str) else min(i['phrase'], key=len)
            # import pdb;pdb.set_trace()
            if phrase in ban_dict[j]['list']:
                odvg.append(anno)
                flag = 1
                break
        if flag == 1:
            break

    odvg = list(filter(None, odvg))
    return [odvg]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRIT2ODVG List.")
    parser.add_argument("input_files", type=str, help='input jsonl file ')  # jsons directory
    parser.add_argument("json_name", type=str, help='output json file name')
    parser.add_argument(
        '--ban_json', '-b', type=str, default='grit_v1_min3_noun_ban.json', help='ban json file'
    )
    parser.add_argument("--process", type=int, default=32, help="the number of processes")
    args = parser.parse_args()
    print(args)

    # # random sample
    # metas, metas_len = prepare_list(args.input_file, args.random_samples)
    ban_dict = {}
    if args.ban_json is not None:
        with open(args.ban_json, 'r') as f:
            ban_json = json.load(f)
        passive_entity_labels = [
            "EVENT",
            "FAC",
            "LAW",
            "LOC",
            "MONEY",
            "NORP",
            "PERSON",
            "PRODUCT",
            "QUANTITY",
            "WORK_OF_ART",
        ]
        ban_entity_labels = ["CARDINAL", "DATE", "GPE", "LANGUAGE", "ORDINAL", "ORG", "PERCENT", "TIME"]
        for i in ban_entity_labels:
            ban_dict[i] = ban_json[i]


    # multi process
    # result = multi_check_jsonl(args.input_files, ban_dict)

    # process single process
    with open(args.input_files, "r") as f:
        metas_len = len(f.readlines())
        f.seek(0)
        for meta in tqdm(f, total=metas_len):
            result = process_jsonl(meta, ban_dict)

    odvg_anno = result[0]
    print(
        f"images num in {result[1]} = {result[2]}, after filter: {result[3]}, num of existing boxes out of range: {result[4]}, caption false: {result[5]}, box too small: {result[6]}, few_caption: {result[7]}"
    )

    print(len(odvg_anno))
    with jsonlines.open(args.json_name, mode="w") as fwriter:
        fwriter.write_all(odvg_anno)
