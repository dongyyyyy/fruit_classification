import torchvision.models as models

def select_model(model_name='resnet18',pretrained=False):
    model_resNet = ['resnet18', 'resnet34', 'resnet50', 'reset101', 'resnet152']
    model_denseNet = ['densenet121', 'densenet169', 'densenet201', 'densenet161']
    model_resNeXt = ['resnext50-32x4d', 'resnext101-32x8d', ]
    model_wideResNet = ['wideresnet50', 'wideresnet101']
    model_efficientNet = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
                          'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7', ]

    if model_name == model_resNet[0]:
        model = models.resnet18(pretrained=pretrained)
    elif model_name == model_resNet[1]:
        model = models.resnet34(pretrained=pretrained)
    elif model_name == model_resNet[2]:
        model = models.resnet50(pretrained=pretrained)
    elif model_name == model_resNet[3]:
        model = models.resnet101(pretrained=pretrained)
    elif model_name == model_resNet[4]:
        model = models.resnet152(pretrained=pretrained)
    elif model_name == model_resNeXt[0]:
        model = models.resnext50_32x4d(pretrained=pretrained)
    elif model_name == model_resNeXt[1]:
        model = models.resnext101_32x8d(pretrained=pretrained)
    elif model_name == model_denseNet[0]:
        model = models.densenet121(pretrained=pretrained)
    elif model_name == model_denseNet[1]:
        model = models.densenet169(pretrained=pretrained)
    elif model_name == model_denseNet[2]:
        model = models.densenet201(pretrained=pretrained)
    elif model_name == model_denseNet[3]:
        model = models.densenet161(pretrained=pretrained)
    elif model_name == model_efficientNet[0]:
        model = models.efficientnet_b0(pretrained=pretrained)
    elif model_name == model_efficientNet[1]:
        model = models.efficientnet_b1(pretrained=pretrained)
    elif model_name == model_efficientNet[2]:
        model = models.efficientnet_b2(pretrained=pretrained)
    elif model_name == model_efficientNet[3]:
        model = models.efficientnet_b3(pretrained=pretrained)
    elif model_name == model_efficientNet[4]:
        model = models.efficientnet_b4(pretrained=pretrained)
    elif model_name == model_efficientNet[5]:
        model = models.efficientnet_b5(pretrained=pretrained)
    elif model_name == model_efficientNet[6]:
        model = models.efficientnet_b6(pretrained=pretrained)
    elif model_name == model_efficientNet[7]:
        model = models.efficientnet_b7(pretrained=pretrained)
    return model