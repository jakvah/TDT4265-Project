from torch import nn
from ssd.modeling.backbone.resnet_backbone import ResNetModel
from ssd.modeling.backbone.assignment4_backbone import ImprovedModel
from ssd.modeling.backbone.resnet_fused import ResNetModelFused


from ssd.modeling.box_head.box_head import SSDBoxHead
from ssd.utils.model_zoo import load_state_dict_from_url
from ssd import torch_utils


class SSDDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.box_head = SSDBoxHead(cfg)
        print(
            "Detector initialized. Total Number of params: ",
            f"{torch_utils.format_params(self)}")
        print(
            f"Backbone number of parameters: {torch_utils.format_params(self.backbone)}")
        print(
            f"SSD Head number of parameters: {torch_utils.format_params(self.box_head)}")

    def forward(self, images, targets=None):
        features = self.backbone(images)
        detections, detector_losses = self.box_head(features, targets)
        if self.training:
            return detector_losses
        return detections


def build_backbone(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    if backbone_name == "resnet":
        model = ResNetModel(cfg)
        return model

    if backbone_name == "ass4":
        model = ImprovedModel(cfg)
        return model

    if backbone_name == "fused":
        model = ResNetModelFused(cfg)
        return model
