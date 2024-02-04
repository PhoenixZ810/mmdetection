_base_ = '../mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py'
load_from = '/mnt/data/mmperc/huanghaian/code/mm_rtdetr/mmdetection/grounding_dino/v3det_1/epoch_30.pth'
lang_model_name = '/mnt/data/mmperc/huanghaian/code/GLIP/bert-base-uncased'
model = dict(
    type='GroundingDINO',
    num_queries=900,
    data_preprocessor=dict(
        type='VideoDataPreprocessor',
        do_round=False,
        div_vid=0,
        bgr_to_rgb=False,
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
            # query self attention layer
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to text
            cross_attn_text_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to image, MultiScaleDeformableAttention
            cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0, im2col_step=1),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
        ),
        post_norm_cfg=None,
    ),
    test_cfg=dict(max_per_img=1),
)
scales = [192, 224, 256, 288, 320]
max_size = 533
resizes = [200, 240, 280]
crop = 160
test_size = [320]

# scales = [96, 128]
# max_size = 213
# resizes = [80, 100, 120]
# crop = 64
# test_size = [128]

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
dataset_type = 'ODVGDataset'
data_root = 'data/'


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
    use_sted=False,
    tmp_loc=False,
    postprocessors=['vidstg'],
)
# val_evaluator = None
test_evaluator = val_evaluator
