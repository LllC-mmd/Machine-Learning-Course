import torch
import torch.nn as nn
from basicModule import *


# ------Block is chosen from: ResBottleneck, SEResBottleneck
class SEResAttentionNet(nn.Module):
    def __init__(self, pic_length, output_stride, block, in_plane=64, expansion=4, num_classes=5):
        super(SEResAttentionNet, self).__init__()
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        self.in_plane = in_plane
        self.expansion = expansion

        self.conv_ini = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.in_plane, kernel_size=7, stride=2, padding=3, bias=False),  # [224, 224] -> [128, 128]
            nn.BatchNorm2d(self.in_plane),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv1 = block(self.in_plane, 64, stride=strides[0], dilation=dilations[0],
                           downsample=downSamplingChoice(self.in_plane, 64*self.expansion, strides[0]))
        # self, size, in_plane, num_plane, block, expansion=4
        self.attL3 = AttentionConvSEBlockL3(64*self.expansion, 32, block, self.expansion)
        self.conv2 = block(32*self.expansion, 32*self.expansion, stride=strides[1], dilation=dilations[1],
                           downsample=downSamplingChoice(32*self.expansion, 32*self.expansion**2, strides[1]))
        self.attL2 = AttentionConvSEBlockL2(32*self.expansion**2, 16*self.expansion, block, self.expansion)
        self.conv3 = block(16*self.expansion**2, 16*self.expansion**2, stride=strides[2], dilation=dilations[2],
                           downsample=downSamplingChoice(16*self.expansion**2, 16*self.expansion**3, strides[2]))
        self.attL1 = AttentionConvSEBlockL1(16*self.expansion**3, 8*self.expansion**2, block, self.expansion)
        self.conv4 = block(8*self.expansion**3, 8*self.expansion**3, stride=strides[3], dilation=dilations[3],
                           downsample=downSamplingChoice(8*self.expansion**3, 8*self.expansion**4, strides[3]))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(512*self.expansion, num_classes)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv_ini(x)
        out = self.conv1(out)
        out = self.attL3(out)
        out = self.conv2(out)
        out = self.attL2(out)
        out = self.conv3(out)
        out = self.attL1(out)
        out = self.conv4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


class ResNet101(nn.Module):
    def __init__(self, output_stride, block, in_plane=64, expansion=4, num_classes=5):
        super(ResNet101, self).__init__()
        layers = [3, 4, 23, 3]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.in_plane = in_plane
        self.expansion = expansion
        self.conv1 = nn.Conv2d(3, self.in_plane, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_plane)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], expansion=self.expansion)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], expansion=self.expansion)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], expansion=self.expansion)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], expansion=self.expansion)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(512*self.expansion, num_classes)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, num_plane, num_block, stride=1, dilation=1, expansion=4):
        layers = []
        layers.append(block(self.in_plane, num_plane, stride, dilation, expansion, downsample=downSamplingChoice(self.in_plane, num_plane*expansion, stride)))
        self.in_plane = num_plane * expansion
        for i in range(1, num_block):
            layers.append(block(self.in_plane, num_plane, dilation=dilation, downsample=downSamplingChoice(self.in_plane, num_plane*expansion, 1)))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out