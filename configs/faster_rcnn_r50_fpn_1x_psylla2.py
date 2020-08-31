# model settings
model = dict(
    type='FasterRCNN',
    pretrained='torchvision://resnet50',###
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
#        norm_cfg=dict(
#            type='BN',
#            requires_grad=True),
#        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0),
#        loss_bbox=dict(
#            type='L1Loss',
#            loss_weight=1.0)),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),###
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign',
                out_size=7,
                sample_num=2),###
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=2,###
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss',
                loss_weight=1.0,
                beta=1.0))))#)),
#        train_cfg=dict(
#            assigner=dict(
#                type='MaxIoUAssigner',
#                pos_iou_thr=0.5,
#                neg_iou_thr=0.5,
#                min_pos_iou=0.5,
#                match_low_quality=False,
#                ignore_iof_thr=-1),
#            sampler=dict(
#                type='RandomSampler',
#                num=512,
#                pos_fraction=0.25,
#                neg_pos_ub=-1,
#                add_gt_as_proposals=True),
#            pos_weight=-1,
#            debug=False),
#        test_cfg=dict(
#            score_thr=0.05,
#            nms=dict(
#                type='nms',
#                iou_thr=0.5),
#            max_per_img=100)))
train_cfg=dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,###
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,###
        max_num=2000,###
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
#            match_low_quality=False,###
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg=dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(
            type='nms',
            iou_thr=0.5),
        max_per_img=100))
# dataset settings
dataset_type = 'PsyllaDataset2'
classes = ('psylla', 'wasp')
#dataset_type = 'Hazeka'
#classes = ('Psyllaephagus', 'Glycaspis')
data_root = 'data/psylla_voc/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(500, 500), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(500, 500), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(500, 500),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/train_annotations.json',
        img_prefix=data_root + 'train/images',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/val_annotations.json',
        img_prefix=data_root + 'val/images',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/test_annotations.json',
        img_prefix=data_root + 'test/images',
        pipeline=test_pipeline))
#evaluation = dict(interval=5, metric='mAP')
evaluation = dict(  # The config to build the evaluation hook, refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/evaluation/eval_hooks.py#L7 for more details.
    interval=1,  # Evaluation interval
    metric=['mAP'])  # Metrics used during evaluation
#    metric='bbox')
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 100
dist_params = dict(backend='nccl')
log_level = 'INFO'
#work_dir = './work_dirs/faster_rcnn_r50_fpn_1x_hazeka'
work_dir = './work_dirs/faster_rcnn_r50_fpn_1x_psylla3'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
