import argparse
import json

import jsonlines
from pycocotools.coco import COCO
from tqdm import tqdm

# this id_map is only for coco dataset which has 80 classes used for training but 90 categories in total.
# which change the start label -> 0
# {"0": "person", "1": "bicycle", "2": "car", "3": "motorcycle", "4": "airplane", "5": "bus", "6": "train", "7": "truck", "8": "boat", "9": "traffic light", "10": "fire hydrant", "11": "stop sign", "12": "parking meter", "13": "bench", "14": "bird", "15": "cat", "16": "dog", "17": "horse", "18": "sheep", "19": "cow", "20": "elephant", "21": "bear", "22": "zebra", "23": "giraffe", "24": "backpack", "25": "umbrella", "26": "handbag", "27": "tie", "28": "suitcase", "29": "frisbee", "30": "skis", "31": "snowboard", "32": "sports ball", "33": "kite", "34": "baseball bat", "35": "baseball glove", "36": "skateboard", "37": "surfboard", "38": "tennis racket", "39": "bottle", "40": "wine glass", "41": "cup", "42": "fork", "43": "knife", "44": "spoon", "45": "bowl", "46": "banana", "47": "apple", "48": "sandwich", "49": "orange", "50": "broccoli", "51": "carrot", "52": "hot dog", "53": "pizza", "54": "donut", "55": "cake", "56": "chair", "57": "couch", "58": "potted plant", "59": "bed", "60": "dining table", "61": "toilet", "62": "tv", "63": "laptop", "64": "mouse", "65": "remote", "66": "keyboard", "67": "cell phone", "68": "microwave", "69": "oven", "70": "toaster", "71": "sink", "72": "refrigerator", "73": "book", "74": "clock", "75": "vase", "76": "scissors", "77": "teddy bear", "78": "hair drier", "79": "toothbrush"}

id_map = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
    10: 11,
    11: 13,
    12: 14,
    13: 15,
    14: 16,
    15: 17,
    16: 18,
    17: 19,
    18: 20,
    19: 21,
    20: 22,
    21: 23,
    22: 24,
    23: 25,
    24: 27,
    25: 28,
    26: 31,
    27: 32,
    28: 33,
    29: 34,
    30: 35,
    31: 36,
    32: 37,
    33: 38,
    34: 39,
    35: 40,
    36: 41,
    37: 42,
    38: 43,
    39: 44,
    40: 46,
    41: 47,
    42: 48,
    43: 49,
    44: 50,
    45: 51,
    46: 52,
    47: 53,
    48: 54,
    49: 55,
    50: 56,
    51: 57,
    52: 58,
    53: 59,
    54: 60,
    55: 61,
    56: 62,
    57: 63,
    58: 64,
    59: 65,
    60: 67,
    61: 70,
    62: 72,
    63: 73,
    64: 74,
    65: 75,
    66: 76,
    67: 77,
    68: 78,
    69: 79,
    70: 80,
    71: 81,
    72: 82,
    73: 84,
    74: 85,
    75: 86,
    76: 87,
    77: 88,
    78: 89,
    79: 90,
}
key_list_coco = list(id_map.keys())
val_list_coco = list(id_map.values())
key_list_o365 = [i for i in range(365)]
val_list_o365 = [i for i in range(1, 366)]


def dump_label_map(output='./out.json'):
    ori_map = {
        '1': 'person',
        '2': 'bicycle',
        '3': 'car',
        '4': 'motorcycle',
        '5': 'airplane',
        '6': 'bus',
        '7': 'train',
        '8': 'truck',
        '9': 'boat',
        '10': 'traffic light',
        '11': 'fire hydrant',
        '13': 'stop sign',
        '14': 'parking meter',
        '15': 'bench',
        '16': 'bird',
        '17': 'cat',
        '18': 'dog',
        '19': 'horse',
        '20': 'sheep',
        '21': 'cow',
        '22': 'elephant',
        '23': 'bear',
        '24': 'zebra',
        '25': 'giraffe',
        '27': 'backpack',
        '28': 'umbrella',
        '31': 'handbag',
        '32': 'tie',
        '33': 'suitcase',
        '34': 'frisbee',
        '35': 'skis',
        '36': 'snowboard',
        '37': 'sports ball',
        '38': 'kite',
        '39': 'baseball bat',
        '40': 'baseball glove',
        '41': 'skateboard',
        '42': 'surfboard',
        '43': 'tennis racket',
        '44': 'bottle',
        '46': 'wine glass',
        '47': 'cup',
        '48': 'fork',
        '49': 'knife',
        '50': 'spoon',
        '51': 'bowl',
        '52': 'banana',
        '53': 'apple',
        '54': 'sandwich',
        '55': 'orange',
        '56': 'broccoli',
        '57': 'carrot',
        '58': 'hot dog',
        '59': 'pizza',
        '60': 'donut',
        '61': 'cake',
        '62': 'chair',
        '63': 'couch',
        '64': 'potted plant',
        '65': 'bed',
        '67': 'dining table',
        '70': 'toilet',
        '72': 'tv',
        '73': 'laptop',
        '74': 'mouse',
        '75': 'remote',
        '76': 'keyboard',
        '77': 'cell phone',
        '78': 'microwave',
        '79': 'oven',
        '80': 'toaster',
        '81': 'sink',
        '82': 'refrigerator',
        '84': 'book',
        '85': 'clock',
        '86': 'vase',
        '87': 'scissors',
        '88': 'teddy bear',
        '89': 'hair drier',
        '90': 'toothbrush',
    }
    new_map = {}
    for key, value in ori_map.items():
        label = int(key)
        ind = val_list.index(label)
        label_trans = key_list[ind]
        new_map[label_trans] = value
    with open(output, 'w') as f:
        json.dump(new_map, f)


def dump_o365_label_map(args):
    with open(args.input, 'r') as f:
        j = json.load(f)
    o_dict = {}
    for category in j['categories']:
        index = str(int(category['id']) - 1)
        name = category['name']
        o_dict[index] = name
    with open(args.output, 'w') as f:
        json.dump(o_dict, f)


def coco2odvg(args):
    coco = COCO(args.input)
    cats = coco.loadCats(coco.getCatIds())
    nms = {cat['id']: cat['name'] for cat in cats}
    metas = []
    if args.dataset == coco:
        key_list = key_list_coco
        val_list = val_list_coco
    else:
        key_list = key_list_o365
        val_list = val_list_o365
    for img_id, img_info in tqdm(coco.imgs.items()):
        if args.dataset == 'o365v2' and img_id in [908726, 320532, 320534]:
            print(img_info["file_name"])
            continue
        if args.dataset == 'o365v1' and img_id in [6, 19, 23]:
            print(img_info["file_name"])
            continue
        else:
            ann_ids = coco.getAnnIds(imgIds=img_id)
            instance_list = []
            for ann_id in ann_ids:
                ann = coco.anns[ann_id]
                if ann.get('ignore', False):
                    continue
                x1, y1, w, h = ann['bbox']
                inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
                inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
                if inter_w * inter_h == 0:
                    continue
                if ann['area'] <= 0 or w < 1 or h < 1:
                    continue

                if ann.get('iscrowd', False):
                    continue

                bbox_xyxy = [x1, y1, x1 + w, y1 + h]
                label = ann['category_id']
                category = nms[label]
                ind = val_list.index(label)
                label_trans = key_list[ind]
                instance_list.append({'bbox': bbox_xyxy, 'label': label_trans, 'category': category})
            metas.append(
                {
                    'filename': img_info['file_name'],
                    'height': img_info['height'],
                    'width': img_info['width'],
                    'detection': {'instances': instance_list},
                }
            )
    print('  == dump meta ...')
    with jsonlines.open(args.output, mode='w') as writer:
        writer.write_all(metas)
    print('  == done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('coco to odvg format.', add_help=True)
    parser.add_argument('input', type=str, help='input list name')
    parser.add_argument("output", type=str, help="output list name")
    parser.add_argument(
        "--dataset",
        '-d',
        required=True,
        type=str,
        help='coco or o365v2 or o365v1',
        choices=['coco', 'o365v2', 'o365v1'],
    )
    args = parser.parse_args()

    coco2odvg(args)
