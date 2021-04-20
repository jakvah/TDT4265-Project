import torch
from torch import nn
from torchvision.models import wide_resnet50_2
import copy

class ResNetModelDeep(torch.nn.Module):
    """
    This is a resnet backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """

    def __init__(self, cfg,):
        super().__init__()
        self.check = False  # Only for checking output dim
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS

        # Loading the resnet backbone
        self.resnet = wide_resnet50_2(pretrained=cfg.MODEL.BACKBONE.PRETRAINED, zero_init_residual=True)
        del self.resnet.avgpool
        del self.resnet.fc

        self.module1 = nn.Sequential(
            nn.Conv2d(in_channels=self.output_channels[3], out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            # nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(
                in_channels=256,
                out_channels=self.output_channels[4],
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(self.output_channels[4]),
            nn.ReLU(),
        )

        self.module2 = nn.Sequential(
            nn.Conv2d(in_channels=self.output_channels[4], out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            # nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(
                in_channels=256,
                out_channels=self.output_channels[5],
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(self.output_channels[5]),
            nn.ReLU(),
        )
        

        self.module3 = nn.Sequential(
            nn.Conv2d(in_channels=self.output_channels[5], out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            # nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(
                in_channels=256,
                out_channels=self.output_channels[6],
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(self.output_channels[6]),
            nn.ReLU(),
        )
        
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
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        out_features.append(x)
        x = self.resnet.layer2(x)
        out_features.append(x)
        x = self.resnet.layer3(x)
        out_features.append(x)
        x = self.resnet.layer4(x)
        out_features.append(x)
        x = self.module1(x)
        out_features.append(x)
        x = self.module2(x)
        out_features.append(x)
        x = self.module3(x)
        out_features.append(x)

        # Only use 6 outputs
        # out_features = out_features[-6:]
        # If we only want to check output dimensions
        if self.check:
            import numpy as np
            out_channels = []
            feature_maps = []
            input_dim = (300, 300)
            for i, output in enumerate(out_features):
                out_channels.append(output.shape[1])
                feature_maps.append([output.shape[3], output.shape[2]])
            print("OUT_CHANNELS:", out_channels)
            print("FEATURE_MAPS:", feature_maps)
            print("STRIDES:", [[int(np.floor((input_dim[1])/(i[1]))), int(np.floor((input_dim[0])/(i[0])))] for i in feature_maps])

        return tuple(out_features)
