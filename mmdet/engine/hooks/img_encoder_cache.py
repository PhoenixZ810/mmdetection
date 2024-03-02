# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook
from mmengine.model.wrappers import is_model_wrapper

from mmdet.registry import HOOKS


@HOOKS.register_module()
class SetEpochInfoHook(Hook):

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        model.set_epoch(epoch)