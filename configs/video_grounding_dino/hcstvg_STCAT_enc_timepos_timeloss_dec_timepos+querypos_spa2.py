_base_ = [
    './hcstvg_STCAT_enc_timepos_timeloss_dec_timepos+querypos_spa.py',
]
lang_model_name = '/mnt/data/mmperc/huanghaian/code/GLIP/bert-base-uncased'
model = dict(
    type='VideoSTCATGroundingDINO',
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
    # frame & video layer config
    frame_layer_cfg=dict(
        self_attn_cfg=dict(num_heads=4, embed_dims=256, dropout=0.0),
        ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, ffn_drop=0.0),
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
        use_weight_loss=False,
        use_time_cross_img=False,
        layer_cfg=dict(
            # query temporal self-attention layer
            time_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # time query type
            time_query_type='t',
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
        type='VideoSTCATGroundingHead',
        num_classes=256,
        sync_cls_avg_factor=True,
        contrastive_cfg=dict(max_text_len=256, log_scale='auto', bias=True),
        loss_cls=dict(
            type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0
        ),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        use_sted=True,
        use_enc_sted=True,
        use_actioness=False,
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
