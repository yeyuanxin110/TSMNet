# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class SWJTUDataset(CustomDataset):
    """SWJTU dataset.

    Induce_zero_label`` should be set to False. The ``img_suffix`` and
    ``se segmentation map annotation for OpenEarthMap, 0 is the ignore index.
    ``reg_map_suffix`` are both fixed to '.tif'.
    """
    # METAINFO = dict(
    #     classes=('background', 'bareland', 'low vegetation', 'trees', 'houses', 'water',
    #              'roads'),
    #     palette=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0],
    #              [4, 0, 0], [5, 0, 0], [6, 0, 0]])
    #     #palette=[[0, 0, 0], [128, 0, 0], [0, 255, 36], [148, 148, 148], [255, 255, 255], [34, 97, 38], [0, 69, 255]])

    CLASSES  = ('background', 'bareland', 'low vegetation', 'trees', 'houses', 'water',
                 'roads')
    palette=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0],[4, 0, 0], [5, 0, 0], [6, 0, 0]]
    #palette = [[0], [1], [2], [3], [4], [5], [6]]

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class For_ContestantsDataset(CustomDataset):
    """SWJTU dataset.

    Induce_zero_label`` should be set to False. The ``img_suffix`` and
    ``se segmentation map annotation for OpenEarthMap, 0 is the ignore index.
    ``reg_map_suffix`` are both fixed to '.tif'.
    """
    # METAINFO = dict(
    #     classes=('background', 'bareland', 'low vegetation', 'trees', 'houses', 'water',
    #              'roads'),
    #     palette=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0],
    #              [4, 0, 0], [5, 0, 0], [6, 0, 0]])
    #     #palette=[[0, 0, 0], [128, 0, 0], [0, 255, 36], [148, 148, 148], [255, 255, 255], [34, 97, 38], [0, 69, 255]])

    CLASSES  = ('other', 'forest land', 'grassland', 'farmland', 'impervious surface', 'water area')
    palette=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0],[4, 0, 0], [5, 0, 0]]
    #PALETTE = [[0], [1], [2], [3], [4], [5]]

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

@DATASETS.register_module()
class For_ContestantsDataset0(CustomDataset):
    """SWJTU dataset.

    Induce_zero_label`` should be set to False. The ``img_suffix`` and
    ``se segmentation map annotation for OpenEarthMap, 0 is the ignore index.
    ``reg_map_suffix`` are both fixed to '.tif'.
    """
    # METAINFO = dict(
    #     classes=('background', 'bareland', 'low vegetation', 'trees', 'houses', 'water',
    #              'roads'),
    #     palette=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0],
    #              [4, 0, 0], [5, 0, 0], [6, 0, 0]])
    #     #palette=[[0, 0, 0], [128, 0, 0], [0, 255, 36], [148, 148, 148], [255, 255, 255], [34, 97, 38], [0, 69, 255]])

    CLASSES  = ( 'forest land', 'grassland', 'farmland', 'impervious surface', 'water area')
    palette=[[1, 0, 0], [2, 0, 0], [3, 0, 0],[4, 0, 0], [5, 0, 0]]
    #PALETTE = [[0], [1], [2], [3], [4], [5]]

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)