import torch
from torch import nn


class ImprovedModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """

    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS

        # Defining the backbone CNN
        module1 = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=self.output_channels[0],
                kernel_size=3,
                stride=2,
                padding=1
            ),

        )
        self.add_module("module1", module1)

        module2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.output_channels[0],
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                in_channels=128,
                out_channels=self.output_channels[1],
                kernel_size=3,
                stride=2,
                padding=1
            )
        )
        self.add_module("module2", module2)

        module3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.output_channels[1],
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(
                in_channels=256,
                out_channels=self.output_channels[2],
                kernel_size=3,
                stride=2,
                padding=1
            )
        )
        self.add_module("module3", module3)

        module4 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.output_channels[2],
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(
                in_channels=256,
                out_channels=self.output_channels[3],
                kernel_size=3,
                stride=2,
                padding=1
            )
        )
        self.add_module("module4", module4)

        module5 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.output_channels[3],
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                in_channels=128,
                out_channels=self.output_channels[4],
                kernel_size=3,
                stride=2,
                padding=1
            )
        )
        self.add_module("module5", module5)

        module6 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.output_channels[4],
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                in_channels=128,
                out_channels=self.output_channels[5],
                kernel_size=3,
                stride=1,
                padding=0
            )
        )
        self.add_module("module6", module6)

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        # Get modules
        for module in self.children():
            # pass data through the network
            x = module(x)
            # Save the output
            out_features.append(x)
        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            expected_shape = (self.output_channels[idx], h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)
