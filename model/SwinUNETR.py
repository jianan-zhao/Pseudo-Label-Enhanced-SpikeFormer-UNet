from monai.networks.nets import SwinUNETR, SegResNet
import torch

def swin_unetr():
    model = SwinUNETR(
        img_size=(128, 128, 128),   
        in_channels=4,            
        out_channels=3,      
        feature_size=24,      
        use_checkpoint=False
    )
    return model


def seg_resnet():
    model = SegResNet(
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        init_filters=32,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
    )
    return model


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    return total_params, trainable_params
