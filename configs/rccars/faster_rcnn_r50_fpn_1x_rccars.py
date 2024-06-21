# The new config inherits a base config to highlight the necessary modification
_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/rccars_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    backbone=dict(init_cfg=None),
    roi_head=dict(
        bbox_head=dict(
            num_classes=4,
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))))

# optimizer
# lr of 0.01 is set for a batch size of 8
optim_wrapper = dict(optimizer=dict(lr=0.01 / 8))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=250),
    dict(
        type='MultiStepLR',
        begin=0,
        end=8,
        by_epoch=True,
        milestones=[7],
        gamma=0.1)
]

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (2 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=8)

# We can use the pre-trained model to obtain higher performance
load_from = '/data/shared/models/mmdet/checkpoints/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth'
