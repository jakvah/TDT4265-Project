from ssd.modeling.box_head.prior_box import PriorBox
from .target_transform import SSDTargetTransform
from .transforms import *

import numpy as np
import albumentations as A


def build_transforms(cfg, is_train=True):
    if is_train:
        """
        Randomly select either
        - Do nothing
        - Randomly sample patch
        -Sample a patch so that the minimum jaccard overlap with the objects is 0.1, 0.3,
            0.5, 0.7, or 0.9

        The size of each sampled patch is [0.1, 1] of the original image size, 
        and the aspect ratio is between 1/2 and 2.           
        """

        transform = [
            ConvertFromInts(),
            ImageDistortion(),

            # Fikser the main shit
            RandomSampleCrop(),
            # Flipperino med 0.5 sannsynlighet
            RandomMirror(),
            ToPercentCoords(),

            # Skalere til correct input size
            Resize(cfg.INPUT.IMAGE_SIZE),

            SubtractMeans(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD),
            ToTensor(),
        ]

    else:
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD),
            ToTensor()
        ]
    transform = Compose(transform)
    return transform


def random_patch():
    pass


def build_target_transform(cfg):
    transform = SSDTargetTransform(PriorBox(cfg)(),
                                   cfg.MODEL.CENTER_VARIANCE,
                                   cfg.MODEL.SIZE_VARIANCE,
                                   cfg.MODEL.THRESHOLD)
    return transform
