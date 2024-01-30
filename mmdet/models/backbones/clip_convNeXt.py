# import warnings
# from collections import OrderedDict
# from copy import deepcopy

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.checkpoint as cp
# from mmcv.cnn import build_norm_layer
# from mmcv.cnn.bricks.transformer import FFN, build_dropout
# from mmengine.logging import MMLogger
# from mmengine.model import BaseModule, ModuleList
# from mmengine.model.weight_init import constant_init, trunc_normal_, trunc_normal_init
# from mmengine.runner.checkpoint import CheckpointLoader
# from mmengine.utils import to_2tuple

# from mmdet.registry import MODELS
# from ..layers import PatchEmbed, PatchMerging

# @MODELS.register_module()
# class ClipConvNeXt(BaseModule):



# from urllib.request import urlopen
# from PIL import Image
# import timm

# # img = Image.open(urlopen(
# #     'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
# # ))

# model = timm.create_model(
#     'CLIP-convnext_large_d.laion2B-s26B-b102K-augreg',
#     pretrained=True,
#     features_only=True,
# )
# model = model.eval()

# # get model specific transforms (normalization, resize)
# data_config = timm.data.resolve_model_data_config(model)
# transforms = timm.data.create_transform(**data_config, is_training=False)

# output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

# for o in output:
#     # print shape of each feature map in output
#     # e.g.:
#     #  torch.Size([1, 128, 64, 64])
#     #  torch.Size([1, 256, 32, 32])
#     #  torch.Size([1, 512, 16, 16])
#     #  torch.Size([1, 1024, 8, 8])

#     print(o.shape)

# import clip
# model, preprocess = clip.load("laion/CLIP-convnext_large_d.laion2B-s26B-b102K-augreg")

import open_clip

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('convnext_large_d',pretrained='/mnt/data/mmperc/zhaoxiangyu/code_new/clip_conv_NeXt/CLIP-convnext_large_d.laion2B-s26B-b102K-augreg/open_clip_pytorch_model.bin')
tokenizer = open_clip.get_tokenizer('convnext_large_d',pretrained = '/mnt/data/mmperc/zhaoxiangyu/code_new/clip_conv_NeXt/CLIP-convnext_large_d.laion2B-s26B-b102K-augreg/')