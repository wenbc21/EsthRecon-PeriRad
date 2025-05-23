from torchvision import models

model_dict = {
    "ResNet18": models.resnet18,
    "ResNet34": models.resnet34,
    "ResNet50": models.resnet50,
    "ResNet101": models.resnet101, 
    "ResNeXt50_32x4d" : models.resnext50_32x4d,
    "ResNeXt101_32x8d" : models.resnext101_32x8d,
    "DenseNet121": models.densenet121, 
    "DenseNet161": models.densenet161, 
    "DenseNet169": models.densenet169, 
    "DenseNet201": models.densenet201,
    "EfficientNetV2_s" : models.efficientnet_v2_s,
    "EfficientNetV2_m" : models.efficientnetv2_m,
    "EfficientNetV2_l" : models.efficientnetv2_l,
    "ConvNeXt_tiny" : models.convnext_tiny,
    "ConvNeXt_small" : models.convnext_small,
    "ConvNeXt_base" : models.convnext_base,
    "ConvNeXt_large" : models.convnext_large,
    "ConvNeXt_xlarge" : models.convnext_xlarge
}