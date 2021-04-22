import torch
from torch import nn
from torchvision.models import resnet50
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

    def __init__(self, cfg):
        super().__init__()
        self.check = False  # Only for checking output dim
        self.scale_lr = True
        self.base_lr = 1e-2

        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS

        data_len = 12000
        self.t_0 = 0.8  # 0.8 default in paper
        self.epochs = cfg.SOLVER.MAX_ITER // data_len

        scale_fn = {'linear': lambda x: x,
                    'squared': lambda x: x**2,
                    'cubic': lambda x: x**3}

        # Loading the resnet backbone
        self.resnet = resnet50(pretrained=cfg.MODEL.BACKBONE.PRETRAINED, zero_init_residual=True)
        del self.resnet.avgpool
        del self.resnet.fc

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
        self.resnet.add_module("p1", self.pre_stage_fuser38)

        self.pre_stage_fuser19 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=2,
                stride=2,
                padding=0,
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
        self.resnet.add_module("p2", self.pre_stage_fuser19)

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
        self.resnet.add_module("d1", self.down_module1)

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
        self.resnet.add_module("m1", self.module1)

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
        self.resnet.add_module("d2", self.down_module2)

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
        self.resnet.add_module("m2", self.module2)

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
        self.resnet.add_module("d3", self.down_module3)

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
        self.resnet.add_module("m3", self.module3)

        # self.custom_modules = nn.ModuleDict({"d1": self.down_module1, "d2": self.down_module2, "d3": self.down_module3, "m1": self.module1,
        #                                      "m2": self.module2, "m3": self.module3, "p1": self.pre_stage_fuser19, "p2": self.pre_stage_fuser38})
        self.relu = nn.ReLU(inplace=True)

        self.num_layers = sum([1 for _ in self.resnet.children()])

        for i, module in enumerate(self.resnet.children()):
            print(module)
            if isinstance(module, (nn.ReLU, nn.MaxPool2d, nn.BatchNorm2d)):
                continue

            module.active = True
            module.layer_index = i
            module.lr_ratio = scale_fn["cubic"](self.t_0 + (1 - self.t_0) * float(module.layer_index) / self.num_layers)
            module.max_iter = self.epochs * 1000 * module.lr_ratio

            # Optionally scale the learning rates to have the same total
            # distance traveled (modulo the gradients).
            module.lr = self.base_lr / module.lr_ratio if self.scale_lr else self.base_lr
            print(module.lr)

    def fuse(self, output_38, output_19):
        out19_staged = self.resnet.p2(output_19)
        out38_staged = self.resnet.p1(output_38)

        out = out19_staged + out38_staged

        return self.relu(out)

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

        fused = self.fuse(out_38, out_19)

        out_features.append(fused)
        out_features.append(out_19)

        x = self.resnet.layer4(out_19)
        out_features.append(x)
        x = self.resnet.d1(x)
        identity = x
        x = self.resnet.m1(x)
        out_features.append(self.relu(x+identity))
        x = self.resnet.d2(x)
        identity = x
        x = self.resnet.m2(x)
        out_features.append(self.relu(x+identity))
        x = self.resnet.d3(x)
        identity = x
        x = self.resnet.m3(x)
        out_features.append(self.relu(x+identity))

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
