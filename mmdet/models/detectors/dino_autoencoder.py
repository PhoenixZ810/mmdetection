# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn.init import normal_

from feat_vision import feature_vision
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .dino import DINO


@MODELS.register_module()
class DINO_autoencoder(DINO):

    def __init__(self, *args,
                #  backbone2: ConfigType, backbone3: ConfigType,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # self.backbone2 = MODELS.build(backbone2)
        # self.backbone3 = MODELS.build(backbone3)

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).

        Returns:
            tuple[Tensor]: Tuple of feature maps from neck. Each feature map
            has shape (bs, dim, H, W).
        """
        with torch.no_grad():
            import pdb;pdb.set_trace()
            x1 = self.backbone(batch_inputs)
            feature_vision(x1[0][0], 'x1_512')
            feature_vision(x1[1][0], 'x1_1024')
            feature_vision(x1[2][0], 'x1_2048')
            # x2 = self.backbone2(batch_inputs)
            # feature_vision(x2[0], 'x2', 4)

            # x3 = self.backbone3(batch_inputs)
            # feature_vision(x3[0], 'x3')
        import pdb
        pdb.set_trace()
        if self.with_neck:
            x = self.neck(x1)
        return x
