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

from nltk import Tree
import en_core_web_trf
import re

abstract_noun = [
    'ability',
    'adoration',
    'advantage',
    'adventure',
    'amazement',
    'anger',
    'annoyance',
    'anxiety',
    'appetite',
    'apprehension',
    'artistry',
    'autumn',
    'awareness',
    'awe',
    'beauty',
    'belief',
    'bravery',
    'brilliance',
    'brutality',
    'calm',
    'care',
    'chaos',
    'charity',
    'childhood',
    'clarity',
    'cleverness',
    'coldness',
    'comfort',
    'communication',
    'compassion',
    'confidence',
    'confusion',
    'contentment',
    'courage',
    'crime',
    'curiosity',
    'customerservice',
    'death',
    'deceit',
    'dedication',
    'defeat',
    'delay',
    'delight',
    'despair',
    'determination',
    'dexterity',
    'dictatorship',
    'disappointment',
    'disbelief',
    'dishonesty',
    'disquiet',
    'disregard',
    'disturbance',
    'divorce',
    'dream',
    'education',
    'ego',
    'elegance',
    'envy',
    'evil',
    'failure',
    'faith',
    'fascination',
    'fear',
    'fiction',
    'fragility',
    'freedom',
    'friendship',
    'gain',
    'generation',
    'generosity',
    'goal',
    'goodness',
    'gossip',
    'growth',
    'happiness',
    'hate',
    'hatred',
    'hope',
    'horror',
    'hurt',
    'idea',
    'infancy',
    'infatuation',
    'inflation',
    'insanity',
    'intelligence',
    'irritation',
    'joy',
    'justice',
    'kindness',
    'laughter',
    'law',
    'liberty',
    'lie',
    'life',
    'loneliness',
    'loss',
    'love',
    'luck',
    'luxury',
    'maturity',
    'mercy',
    'movement',
    'music',
    'nap',
    'need',
    'opinion',
    'opportunity',
    'pain',
    'patience',
    'peace',
    'peculiarity',
    'perseverance',
    'pleasure',
    'poverty',
    'power',
    'pride',
    'principle',
    'reality',
    'relaxation',
    'relief',
    'religion',
    'restoration',
    'riches',
    'right',
    'rumour',
    'sacrifice',
    'sanity',
    'satisfaction',
    'self-control',
    'season',
    'sensitivity',
    'service',
    'shock',
    'silliness',
    'skill',
    'sleep',
    'sorrow',
    'speed',
    'spring',
    'strenght',
    'strictness',
    'success',
    'summer',
    'surprise',
    'sunlight',
    'talent',
    'thrill',
    'time',
    'tiredness',
    'tolerance',
    'trend',
    'trust',
    'uncertainty',
    'unemployment',
    'union',
    'unreality',
    'victory',
    'wariness',
    'warmth',
    'weakness',
    'wealth',
    'weariness',
    'winter',
    'wisdom',
    'wit',
    'worry',
    'scene',
    'viewpoint',
]

abstract_noun_end = ('tion', 'ty', 'ness', 'ity', 'ment', 'ism')

nlp = en_core_web_trf.load()
ban_entity_labels = ["CARDINAL", "DATE", "GPE", "LANGUAGE", "ORDINAL", "PERCENT", "TIME"]


# 对caption中的标点符号进行清洗，中文标点转为英文标点，去掉句末的标点
def clean_span(span):
    span = span.rstrip()
    span = span.replace('"', "'").replace('\"', "'").replace('“', "'").replace('”', "'")
    span = span.replace('‘', "'").replace('’', "'").replace('–', "—")
    if span.endswith('/') or span.endswith('.'):
        span = span[:-1]
    return span


# 检查grit caption中是否有除了ASCII文本以外的其他元素（如.*/-emoji[CLS]等)，如果有返回False
def check_caption(cap):
    check_anno = cap["conversations"][1]["value"].rstrip()[:-1]
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


def remove_markdown(text):
    # 用正则表达式匹配**加粗**的部分，并替换为去掉**的内容
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    # 用正则表达式匹配"引用"的部分，并替换为去掉"的内容
    text = re.sub(r"\"(.*?)\"", r"\1", text)
    # 用正则表达式匹配\n换行符，并替换为空格
    text = re.sub(r"\n", " ", text)
    # 返回处理后的字符串
    return text


def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


def process_json(js, args):
    odvg = []
    if os.path.isdir(args.input_files):
        js = os.path.join(args.input_files, js)
        # print(js)
    with open(js, "r") as f:
        metas = json.load(f)
        metas_len = len(metas)

        for meta in metas:
            result = process_item(meta, args)
            odvg.append(result)
            break
            # if result is not None:
            #     if len(result['referring']) != 0:
            # odvg.append(result)

    # print(
    #     f"images num in {js} = {metas_len}, after filter: {len(odvg)}, num of existing boxes out of range: {out_ranges}"
    # )
    # print(f"caption false: {caption_false}, box too small: {box_small}")
    odvg = list(filter(None, odvg))
    # print(noun_same_with_ref)
    return odvg


def process_item(file, args):
    annotation = file
    image_id = annotation["id"]
    filename = annotation["image"]
    raw_des = annotation["conversations"][1]["value"]
    raw_sents = raw_des.split('\n\n')
    raw_sents.pop()
    # print(raw_sents)
    description = ' '.join(raw_sents)
    # 除去markdown格式
    print(description)
    description = remove_markdown(description)
    print(description)

    odvg_anno = None

    # nlp filter
    filter_flag = 0

    doc = nlp(description)
    entities = {ent.text: ent.label_ for ent in doc.ents if ent.label_ in ban_entity_labels}
    entities_text = entities.keys()

    # [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]
    phrases = []
    for sent in doc.sents:
        for token in sent:  # 遍历每个词
            if (
                token.pos_ == "NOUN"
                and not token.text.endswith(abstract_noun_end)
                and token.ent_type_ not in ban_entity_labels
                and token.text not in abstract_noun
            ):  # 如果词的词性是名词
                if token.head.pos_ == "NOUN" and token.n_lefts + token.n_rights == 0:
                    continue
                else:
                    print(token.text, token.dep_)
                    phrase = " ".join([t.text for t in token.subtree])  # 将词及其所有的依存子节点的文本用空格连接
                    if len(phrases) == 0 and 'image' in phrase:  # 将gpt回答中的第一句的image去掉
                        continue
                    else:
                        phrases.append(phrase)
    print(phrases)
    odvg_anno = {
        "filename": filename,
        # "referring": {"caption": clean_span(annotation["caption"]), "regions": regions},
        "referring": [description],
    }
    # filter_inform = []
    # if filter_flag == 1:
    #     if entities_text == None:
    #         entities_text = []
    #     filter_inform = [odvg_anno['filename'], entities_text]
    return odvg_anno


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ShareGPT4V2ODVG List.")
    parser.add_argument("input_files", type=str, help='input json file ')  # jsons directory
    parser.add_argument("json_name", type=str, help='output json file name')
    parser.add_argument('--ban_json', '-b', type=str, default=None, help='ban json file')
    parser.add_argument("--min_phrase", "-m", type=int, default=3)
    parser.add_argument("--process_num", type=int, default=4, help="the number of processes")
    args = parser.parse_args()
    # print(args)

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
        odvg_anno = result
        # print(
        #     f"images num in {result[1]} = {result[2]}, after filter: {result[3]}, num of existing boxes out of range: {result[4]}, caption false: {result[5]}, box too small: {result[6]}, few_caption: {result[7]}"
        # )
    # process single json file
    print(len(odvg_anno))
    with jsonlines.open(args.json_name, mode="w") as fwriter:
        fwriter.write_all(odvg_anno)
