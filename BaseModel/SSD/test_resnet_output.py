# For test, and finding OUT_CHANNELS
if __name__ == "__main__":
    import torch
    import numpy as np
    from ssd.config.defaults import cfg
    from ssd.modeling.backbone.resnet_backbone import ResNetModel
    # Change this tensor to correspond with image:
    # Parameters are: batch_size, channels, height, width
    input_tensor = torch.rand(16, 3, 300, 300)
    resnet = ResNetModel(cfg, no_check=True)
    output_tensor = resnet(input_tensor)
    # print("input shape is: \n channels:", input_tensor.shape[1], "height:", input_tensor.shape[2], "width:", input_tensor.shape[3])
    out_channels = []
    feature_maps = []
    for i, output in enumerate(output_tensor):
        out_channels.append(output.shape[1])
        feature_maps.append([output.shape[2], output.shape[3]])
        # print("output_channels["+str(i)+"]:", output.shape[1], "height:", output.shape[2], "width:", output.shape[3])
    print("OUT_CHANNELS:", out_channels)
    print("FEATURE_MAPS:", feature_maps)
    print("STRIDES:", [[np.floor((300-1)/(i[0]-1)), np.floor((300-1)/(i[1]-1))] for i in feature_maps])
    print("Note: Strides tror jeg ikke trenger stemme helt\n")
    
