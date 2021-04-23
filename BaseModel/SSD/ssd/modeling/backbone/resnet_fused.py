import torch
from torch import nn
from torchvision.models import resnet34
# import torch.nn.functional as F
import copy

class ResNetModelFusion(torch.nn.Module):
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
        self.resnet = resnet34(pretrained=cfg.MODEL.BACKBONE.PRETRAINED, zero_init_residual=True)
        del self.resnet.avgpool
        del self.resnet.fc

        self.down_module1 = nn.Sequential(
            nn.Conv2d(in_channels=self.output_channels[2], out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=self.output_channels[3],
                kernel_size=3,
                stride=2,
                padding=1
            ),
        )
        self.module1 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(self.output_channels[3], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(in_channels=self.output_channels[3], out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(
                in_channels=512,
                out_channels=self.output_channels[3],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(self.output_channels[3], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(),
        )

        self.down_module2 = nn.Sequential(
            nn.Conv2d(in_channels=self.output_channels[3], out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=self.output_channels[4],
                kernel_size=3,
                stride=2,
                padding=1
            ),
        )
        self.module2 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(self.output_channels[4], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(in_channels=self.output_channels[4], out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(
                in_channels=512,
                out_channels=self.output_channels[4],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(self.output_channels[4], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(),
        )
        

        self.down_module3 = nn.Sequential(
            nn.Conv2d(in_channels=self.output_channels[4], out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=self.output_channels[5],
                kernel_size=3,
                stride=2,
                padding=1
            ),
        )
        self.module3 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(self.output_channels[5], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(in_channels=self.output_channels[5], out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(
                in_channels=512,
                out_channels=self.output_channels[5],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(self.output_channels[5], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(),
        )
        self.relu = nn.ReLU(inplace=True)

        """
        self.pre_stage_fuser19 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=2,
                stride=2,
                padding=0,
                dilation=1
            ),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(512),
        )

        self.pre_stage_fuser38 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(512),
        )

    def fuse(self,output_38, output_19):
        out19_staged = self.pre_stage_fuser19(output_19)
        out38_staged = self.pre_stage_fuser38(output_38)

        out = out19_staged + out38_staged

        return self.relu(out)

    """
        
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
        out_38 = self.resnet.layer2(x)

        out_19 = self.resnet.layer3(out_38)

        # fused = self.fuse(out_38, out_19)

        # out_features.append(fused)
        out_features.append(out_38)
        out_features.append(out_19)

        x = self.resnet.layer4(out_19)
        out_features.append(x)
        x = self.down_module1(x)
        identity = x
        x = self.module1(x)
        x = self.relu(x+identity)
        out_features.append(x)
        x = self.down_module2(x)
        identity = x
        x = self.module2(x)
        x = self.relu(x+identity)
        out_features.append(x)
        x = self.down_module3(x)
        identity = x    
        x = self.module3(x)
        x = self.relu(x+identity)
        out_features.append(x)
        
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
