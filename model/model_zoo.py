from model.ResNet import resnet34, resnet50, resnet101, resnext50_32x4d, resnext101_32x8d
from model.DenseNet import densenet121, densenet161, densenet169, densenet201
from model.EfficientNet import efficientnetv2_s, efficientnetv2_m, efficientnetv2_l
from model.ConvNeXt import convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge

model_dict = {
    "resnet34" : resnet34,
    "resnet50" : resnet50,
    "resnet101" : resnet101,
    "resnext50_32x4d" : resnext50_32x4d,
    "resnext101_32x8d" : resnext101_32x8d,
    "densenet121" : densenet121,
    "densenet161" : densenet161,
    "densenet169" : densenet169,
    "densenet201" : densenet201,
    "efficientnetv2_s" : efficientnetv2_s,
    "efficientnetv2_m" : efficientnetv2_m,
    "efficientnetv2_l" : efficientnetv2_l,
    "convnext_tiny" : convnext_tiny,
    "convnext_small" : convnext_small,
    "convnext_base" : convnext_base,
    "convnext_large" : convnext_large,
    "convnext_xlarge" : convnext_xlarge,
}