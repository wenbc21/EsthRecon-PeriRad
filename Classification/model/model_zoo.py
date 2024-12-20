from model.ResNet import resnet34, resnet50, resnet101, resnext50_32x4d, resnext101_32x8d
from model.DenseNet import densenet121, densenet161, densenet169, densenet201
from model.EfficientNet import efficientnetv2_s, efficientnetv2_m, efficientnetv2_l
from model.ConvNeXt import convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge

model_dict = {
    "ResNet34" : resnet34,
    "ResNet50" : resnet50,
    "ResNet101" : resnet101,
    "ResNeXt50_32x4d" : resnext50_32x4d,
    "ResNeXt101_32x8d" : resnext101_32x8d,
    "DenseNet121" : densenet121,
    "DenseNet161" : densenet161,
    "DenseNet169" : densenet169,
    "DenseNet201" : densenet201,
    "EfficientNetV2_s" : efficientnetv2_s,
    "EfficientNetV2_m" : efficientnetv2_m,
    "EfficientNetV2_l" : efficientnetv2_l,
    "ConvNeXt_tiny" : convnext_tiny,
    "ConvNeXt_small" : convnext_small,
    "ConvNeXt_base" : convnext_base,
    "ConvNeXt_large" : convnext_large,
    "ConvNeXt_xlarge" : convnext_xlarge
}