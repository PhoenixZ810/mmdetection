_base_ = [
    './freezeenc_dec_timepos+querypos_spa.py',
]
# pretrained = '/mnt/data/mmperc/huanghaian/code/GLIP/swin_tiny_patch4_window7_224.pth'  # noqa
load_from = '/mnt/data/mmperc/huanghaian/code/mm_rtdetr/mmdetection/grounding_dino/v3det_1/epoch_30.pth'
lang_model_name = '/mnt/data/mmperc/huanghaian/code/GLIP/bert-base-uncased'

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
        bgr_to_rgb=True,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
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
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=True,
        convert_weights=True,
        frozen_stages=-1,
        # init_cfg=dict(type='Pretrained', checkpoint=pretrained),
    ),
    neck=dict(in_channels=[256, 512, 1024]),
    encoder=dict(
        num_layers=6,
        num_cp=6,
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
            time_attn_cfg=None,
            # time query type
            time_query_type='q',
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

dataset_type = 'ODVGDataset'
data_root = 'data/'

# scales = [96, 128]
# max_size = 213
# resizes = [80, 100, 120]
# crop = 64
# test_size = [800]

scales = [192, 224, 256, 288, 320]
max_size = 533
resizes = [200, 240, 280]
crop = 160
test_size = [320]

cautious = True

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
                # dict(type='VideoNormalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
