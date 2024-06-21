# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class RcCarsDataset(CocoDataset):
    """Dataset for DeepFashion."""

    METAINFO = {
        'classes': ('black car', 'blue car', 'brown truck', 'red car'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(0, 0, 0), (0, 32, 255), (160, 128, 96), (196, 32, 32)]
    }
