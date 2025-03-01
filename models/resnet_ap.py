# Original code: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch.nn.functional as F
import torch.nn as nn
import math
from torch.nn.utils import spectral_norm


def conv_stride1(in_planes, out_planes, kernel_size=3, norm_type="instance"):
    "3x3 convolution with padding"
    if norm_type in ["sn", "none"]:
        bias = True
    else:
        bias = False

    layer = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        bias=bias,
    )

    if norm_type == "sn":
        return spectral_norm(layer)
    else:
        return layer


class Null(nn.Module):
    def __init__(self):
        super(Null, self).__init__()

    def forward(self, x):
        return x


def normalization(inplanes, norm_type):
    if norm_type == "batch":
        bn = nn.BatchNorm2d(inplanes)
    elif norm_type == "instance":
        bn = nn.GroupNorm(inplanes, inplanes)
    elif norm_type in ["sn", "none"]:
        bn = Null()
    else:
        raise AssertionError(f"Check normalization type! {norm_type}")
    return bn


class IntroBlock(nn.Module):
    def __init__(self, size, planes, norm_type, nch=3):
        super(IntroBlock, self).__init__()
        self.size = size
        if size == "large":
            self.conv1 = conv_stride1(nch, planes, kernel_size=7, norm_type=norm_type)
            self.bn1 = normalization(planes, norm_type)
            self.relu = nn.ReLU(inplace=True)
            self.pool = nn.AvgPool2d(kernel_size=4, stride=4)
        elif size == "mid":
            self.conv1 = conv_stride1(nch, planes, kernel_size=3, norm_type=norm_type)
            self.bn1 = normalization(planes, norm_type)
            self.relu = nn.ReLU(inplace=True)
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        elif size == "small":
            self.conv1 = conv_stride1(nch, planes, kernel_size=3, norm_type=norm_type)
            self.bn1 = normalization(planes, norm_type)
            self.relu = nn.ReLU(inplace=True)
        else:
            raise AssertionError("Check network size type!")

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.size != "small":
            x = self.pool(x)

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, norm_type="batch", stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_stride1(
            inplanes, planes, kernel_size=3, norm_type=norm_type
        )  # Modification
        self.bn1 = normalization(planes, norm_type)
        self.conv2 = conv_stride1(planes, planes, kernel_size=3, norm_type=norm_type)
        self.bn2 = normalization(planes, norm_type)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.stride != 1:  # modification
            out = F.avg_pool2d(out, kernel_size=self.stride, stride=self.stride)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, norm_type="batch", stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = normalization(planes, norm_type)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, bias=False
        )  # modification
        self.bn2 = normalization(planes, norm_type)
        self.conv3 = nn.Conv2d(
            planes, planes * Bottleneck.expansion, kernel_size=1, bias=False
        )
        self.bn3 = normalization(planes * Bottleneck.expansion, norm_type)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.stride != 1:  # modification
            out = F.avg_pool2d(out, kernel_size=self.stride, stride=self.stride)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetAP(nn.Module):
    def __init__(
        self, dataset, depth, num_classes, width=1.0, norm_type="batch", size=-1, nch=3
    ):
        super(ResNetAP, self).__init__()
        self.dataset = dataset
        self.norm_type = norm_type
        self.nch = nch

        if self.dataset.startswith("cifar") or (0 < size and size <= 64):
            self.net_size = "small"
        elif 64 < size and size <= 128:
            self.net_size = "mid"
        else:
            self.net_size = "large"

        # print(f"ResNetAP-{depth}-{self.net_size} norm: {self.norm_type}, width: {width}")
        if self.dataset.startswith("cifar"):
            self.inplanes = 32
            n = int((depth - 2) / 6)
            block = BasicBlock

            self.layer0 = IntroBlock(self.net_size, self.inplanes, norm_type, nch=nch)
            self.layer1 = self._make_layer(block, 32, n, stride=1)
            self.layer2 = self._make_layer(block, 64, n, stride=2)
            self.layer3 = self._make_layer(block, 128, n, stride=2)
            self.layer4 = self._make_layer(block, 256, n, stride=2)
            self.avgpool = nn.AvgPool2d(4)
            self.fc = nn.Linear(256 * block.expansion, num_classes)

        else:
            blocks = {
                10: BasicBlock,
                18: BasicBlock,
                34: BasicBlock,
                50: Bottleneck,
                101: Bottleneck,
                152: Bottleneck,
                200: Bottleneck,
            }
            layers = {
                10: [1, 1, 1, 1],
                18: [2, 2, 2, 2],
                34: [3, 4, 6, 3],
                50: [3, 4, 6, 3],
                101: [3, 4, 23, 3],
                152: [3, 8, 36, 3],
                200: [3, 24, 36, 3],
            }
            assert layers[depth], "invalid detph for ResNet"

            self.inplanes = int(64 * width)
            self.layer0 = IntroBlock(self.net_size, self.inplanes, norm_type, nch=nch)
            nc = self.inplanes
            self.layer1 = self._make_layer(blocks[depth], nc, layers[depth][0])
            self.layer2 = self._make_layer(
                blocks[depth], nc * 2, layers[depth][1], stride=2
            )
            self.layer3 = self._make_layer(
                blocks[depth], nc * 4, layers[depth][2], stride=2
            )
            self.layer4 = self._make_layer(
                blocks[depth], nc * 8, layers[depth][3], stride=2
            )
            self.avgpool = nn.AvgPool2d(7)
            self.fc = nn.Linear(self.inplanes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv_stride1(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    norm_type=self.norm_type,
                ),
                nn.AvgPool2d(kernel_size=stride, stride=stride),
                normalization(planes * block.expansion, self.norm_type),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                norm_type=self.norm_type,
                stride=stride,
                downsample=downsample,
            )
        )
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_type=self.norm_type))

        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, x.shape[-1])
        x = x.view(x.size(0), -1)

        out = self.fc(x)
        if return_features:
            return out, x
        else:
            return out

    def get_feature(self, x, idx_from, idx_to=-1):
        if idx_to == -1:
            idx_to = idx_from

        features = []
        x = self.layer0(x)
        features.append(x)  # starts from 0
        if idx_to < len(features):
            return features[idx_from : idx_to + 1]

        x = self.layer1(x)
        features.append(x)
        if idx_to < len(features):
            return features[idx_from : idx_to + 1]

        x = self.layer2(x)
        features.append(x)
        if idx_to < len(features):
            return features[idx_from : idx_to + 1]

        x = self.layer3(x)
        features.append(x)
        if idx_to < len(features):
            return features[idx_from : idx_to + 1]

        x = self.layer4(x)
        features.append(x)
        if idx_to < len(features):
            return features[idx_from : idx_to + 1]

        x = F.avg_pool2d(x, x.shape[-1])
        x = x.view(x.size(0), -1)
        features.append(x)
        if idx_to < len(features):
            return features[idx_from : idx_to + 1]

        x = self.fc(x)
        features.append(x)  # logit is 6
        return features[idx_from : idx_to + 1]

    def get_feature_mutil(self, x, layer_num=7):
        features = []
        x = self.layer0(x)
        features.append(x.view(x.size(0), -1))
        if layer_num == 1:
            return features

        x = self.layer1(x)
        features.append(x.view(x.size(0), -1))
        if layer_num == 2:
            return features

        x = self.layer2(x)
        features.append(x.view(x.size(0), -1))
        if layer_num == 3:
            return features

        x = self.layer3(x)
        features.append(x.view(x.size(0), -1))
        if layer_num == 4:
            return features

        x = self.layer4(x)
        features.append(x.view(x.size(0), -1))
        if layer_num == 5:
            return features

        x = F.avg_pool2d(x, x.shape[-1])
        x = x.view(x.size(0), -1)
        features.append(x)
        if layer_num == 6:
            return features

        x = self.fc(x)
        features.append(x)
        if layer_num == 7:
            return features
        # features len == 7 ,and index 0-5 is the feature,index 6 logit is
        return features


if __name__ == "__main__":
    import torch

    dataset = "imagenet"
    num_classes = 10
    size = int(224 * 0.5)
    depth = 10
    width = 1.0
    norm_type = "instance"
    nch = 1

    model = ResNetAP(
        dataset,
        depth,
        num_classes,
        size=size,
        width=width,
        norm_type=norm_type,
        nch=nch,
    ).cuda()
    # print(model)
    print(
        "# model parameters: {:.1f}M".format(
            sum([p.data.nelement() for p in model.parameters()]) / 10**6
        )
    )

    model.train()
    for name, param in model.named_parameters():
        if len(param.shape) > 2:
            print(name, param.shape)
    # print(model)

    data = torch.ones([128, nch, size, size]).to("cuda")
    output = model(data)
    print(output.shape)
