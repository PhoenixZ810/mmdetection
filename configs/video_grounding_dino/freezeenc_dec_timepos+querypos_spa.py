_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]
# pretrained = '/mnt/data/mmperc/huanghaian/code/GLIP/swin_tiny_patch4_window7_224.pth'  # noqa
load_from = '/mnt/data/mmperc/huanghaian/code/mm_rtdetr/mmdetection/grounding_dino/v3det_1/epoch_30.pth'
lang_model_name = '/mnt/data/mmperc/huanghaian/code/GLIP/bert-base-uncased'

# find_unused_parameters = True

model = dict(
    type='VideoGroundingDINO',
    num_queries=1,
    with_box_refine=True,
    as_two_stage=True,
    use_time_embed=True,
    max_time_pos_frames=200,
    freeze_backbone=True,
    freeze_language_model=True,
    freeze_encoder=False,
    data_preprocessor=dict(
        type='VideoDataPreprocessor',
        do_round=False,
        div_vid=0,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    language_model=dict(
        type='BertModel',
        name=lang_model_name,
        max_tokens=256,
        pad_to_max=False,
        use_sub_sentence_represent=True,
        special_tokens_list=['[CLS]', '[SEP]', '.', '?'],
        add_pooling_layer=False,
    ),
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=True,
        convert_weights=True,
        frozen_stages=-1,
        # init_cfg=dict(type='Pretrained', checkpoint=pretrained),
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[192, 384, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        bias=True,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4,
    ),
    encoder=dict(
        num_layers=6,
        num_cp=6,
        # # visual temporal self-attention config
        # time_attn_layer_cfg=dict(
        #     self_attn_cfg=dict(num_heads=4, embed_dims=256, dropout=0.0),
        #     ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, ffn_drop=0.0),
        # ),
        # visual layer config, MultiScaleDeformableAttention
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0, im2col_step=1),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
        ),
        # text layer config
        text_layer_cfg=dict(
            self_attn_cfg=dict(num_heads=4, embed_dims=256, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, ffn_drop=0.0),
        ),
        # fusion layer config
        fusion_layer_cfg=dict(v_dim=256, l_dim=256, embed_dim=1024, num_heads=4, init_values=1e-4),
    ),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            # query temporal self-attention layer
            time_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # time query type
            time_query_type='tq',
            # query self attention layer
            use_self_attn=True,
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to text
            cross_attn_text_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to image, MultiScaleDeformableAttention
            cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0, im2col_step=1),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
        ),
        post_norm_cfg=None,
    ),
    positional_encoding=dict(num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type='VideoGroundingHead',
        num_classes=256,
        sync_cls_avg_factor=True,
        contrastive_cfg=dict(max_text_len=256, log_scale='auto', bias=True),
        loss_cls=dict(
            type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0
        ),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        use_sted=True,
        sted_loss_weight=10.0,
        time_only=False,
        exclude_cls=False,
    ),
    use_dn=False,
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100),
    ),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='BinaryFocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0),
            ],
        )
    ),
    test_cfg=dict(max_per_img=1),
)

# dataset settings
scales = [96, 128]
max_size = 213
resizes = [80, 100, 120]
crop = 64
test_size = [128]
cautious = True
video_train_pipeline = [
    dict(
        type='mp4_to_image',
        video_max_len_train=100,
        fps=5,
        time_crop=True,
        is_train=True,
        spatial_transform=dict(
            type='Compose',
            transforms=[
                dict(type='VideoRandomHorizontalFlip'),
                dict(type='VideoRandomResize', sizes=scales, max_size=max_size),
                # dict(
                #     type='RandomChoice',
                #     transforms=[
                #         dict(type='VideoRandomResize', sizes=scales, max_size=max_size),
                #         dict(
                #             type='Compose',
                #             transforms=[
                #                 dict(type='VideoRandomResize', sizes=resizes, max_size=max_size),
                #                 dict(
                #                     type='VideoRandomSizeCrop',
                #                     min_size=crop,
                #                     max_size=max_size,
                #                     respect_boxes=cautious,
                #                 ),
                #                 dict(type='VideoRandomResize', sizes=scales, max_size=max_size),
                #             ],
                #         ),
                #     ],
                # ),
                dict(type='VideoToTensor'),
                # dict(type='VideoBoxNormalize'), # ordinate normalization, xyxy
            ],
        ),
    ),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'video_id',
            'video_path',
            'ori_shape',
            'img_shape',
            'text',
            'frames_id',
            'inter_idx',
            'img_in_vid_ids',
            # 'qtype',
            # 'tokens_positive',
            'dataset_mode',
        ),
    ),
]

video_test_pipeline = [
    dict(
        type='mp4_to_image',
        video_max_len_val=200,
        fps=5,
        is_train=False,
        spatial_transform=dict(
            type='Compose',
            transforms=[
                dict(type='VideoRandomResize', sizes=test_size, max_size=max_size),
                dict(type='VideoToTensor'),
            ],
        )
        # dict(type='VideoBoxNormalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'video_id',
            'video_path',
            'ori_shape',
            'img_shape',
            'text',
            'frames_id',
            'inter_idx',
            'inter_frames',
            'img_in_vid_ids',
            'qtype',
            'tube_start_frame',
            'tube_end_frame',
            'tokens_positive',
            'dataset_mode',
        ),
    ),
]

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    keep_ratio=True,
                )
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True,
                ),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True,
                ),
                dict(
                    type='RandomChoiceResize',
                    scales=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    keep_ratio=True,
                ),
            ],
        ],
    ),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomSamplingNegPos', tokenizer_name=lang_model_name, num_sample_negative=85, max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'flip',
            'flip_direction',
            'text',
            'custom_entities',
            'tokens_positive',
            'dataset_mode',
        ),
    ),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, imdecode_backend='pillow'),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True, backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'text',
            'custom_entities',
            'tokens_positive',
        ),
    ),
]

dataset_type = 'ODVGDataset'
data_root = 'data/'

# coco_od_dataset = dict(
#     type=dataset_type,
#     data_root=data_root,
#     ann_file='o365v1_train_odvg.json',
#     label_map_file='o365v1_label_map.json',
#     data_prefix=dict(img='train/'),
#     filter_cfg=dict(filter_empty_gt=False),
#     pipeline=train_pipeline,
#     return_classes=True,
#     backend_args=None,
# )

vidstg_video_dataset = dict(
    type='VideoModulatedSTGrounding',
    data_root=data_root,
    # ann_file='/mnt/workspace/zhaoxiangyu/code_new/video_mmdetection/debug/debug.json',
    ann_file='/mnt/data/mmperc/zhaoxiangyu/code_new/video_mmdetection/data/VidSTG/annotations/train_filter.json',
    data_prefix=dict(img='VidSTG/video/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=video_train_pipeline,
    is_train=True,
)

train_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type='ConcatDataset', datasets=[vidstg_video_dataset]),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VideoModulatedSTGrounding',
        data_root=data_root,
        # ann_file='/mnt/data/mmperc/zhaoxiangyu/code_new/video_mmdetection/debug/debug.json',
        ann_file='/mnt/data/mmperc/zhaoxiangyu/code_new/video_mmdetection/data/VidSTG/annotations/val.json',
        data_prefix=dict(img='VidSTG/video/'),
        test_mode=True,
        video_max_len=100,
        pipeline=video_test_pipeline,
        backend_args=None,
    ),
)
# val_dataloader = None
# val_dataloader = dict(dataset=dict(pipeline=test_pipeline, return_classes=True))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='VidstgMetric',
    # ann_file='/mnt/data/mmperc/zhaoxiangyu/code_new/video_mmdetection/debug/debug.json',
    ann_file='/mnt/data/mmperc/zhaoxiangyu/code_new/video_mmdetection/data/VidSTG/annotations/val.json',
    metric='bbox',
    format_only=False,
    backend_args=None,
    use_sted=True,
    tmp_loc=True,
    postprocessors=['vidstg'],
)
# val_evaluator = None
test_evaluator = val_evaluator

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00002, weight_decay=0.0001),  # bs=16 0.0001
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.0),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0.1),
        }
    ),
)

# learning policy
max_epochs = 30
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[19, 26], gamma=0.1),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)

default_hooks = dict(visualization=dict(type='GroundingVisualizationHook'))
