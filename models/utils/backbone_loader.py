import torch as t
import torch.nn as nn
import torchvision as tv
from utils.config import opt


def load_vgg16(pretrained=True):
    model = tv.models.vgg16(pretrained=pretrained)
    # drop the last max-pooling layer
    features = list(model.features)[:-1]
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    features = nn.Sequential(*features)

    top_layer = list(model.classifier)[:6]
    if not opt.use_drop:
        del top_layer[5]
        del top_layer[2]
    top_layer = nn.Sequential(*top_layer)

    return features, top_layer


def load_vgg16_as_fully_convolutional(pool_conv5, pretrained=True):
    model = tv.models.vgg16(pretrained=pretrained)

    if pool_conv5:
        features = list(model.features)
    else:
        features = list(model.features[:-1])

    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

    # reshape pretrained weight
    conv6_weight = model.classifier[0].weight.view(4096, 512, 7, 7)
    conv6_bias = model.classifier[0].bias

    conv7_weight = model.classifier[3].weight.view(4096, 4096, 1, 1)
    conv7_bias = model.classifier[3].bias

    # subsampling weight
    conv6.weight = nn.Parameter(decimate(conv6_weight, m=[4, None, 3, 3]))
    conv6.bias = nn.Parameter(decimate(conv6_bias, m=[4]))

    conv7.weight = nn.Parameter(decimate(conv7_weight, m=[4, 4, None, None]))
    conv7.bias = nn.Parameter(decimate(conv7_bias, m=[4]))

    features += [conv6, nn.ReLU(True), conv7, nn.ReLU(True)]

    return nn.Sequential(*features)


def decimate(tensor, m):
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(
                dim=d, index=t.arange(start=0, end=tensor.size(d), step=m[d]).long()
            )

    return tensor