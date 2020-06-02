import torch
import torch.nn as nn
import torch.nn.functional as F
from basicModule import *
from backbone import *


class Decoder(nn.Module):
    def __init__(self, in_plane, num_classes=5):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_plane, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x


class LUSegDeepLab(nn.Module):
    def __init__(self, backbone, num_plane=2048, output_stride=8, num_classes=5, pretrained_backbone=None):
        super(LUSegDeepLab, self).__init__()
        self.backbone = build_backbone(backbone, output_stride, pretrained_backbone)
        self.aspp = AtrousSPP(in_plane=num_plane, output_stride=output_stride, num_plane=256)
        if backbone == "AttResNet":
            self.decoder = Decoder(in_plane=128, num_classes=num_classes)
        elif backbone == "ResNet50":
            self.decoder = Decoder(in_plane=256, num_classes=num_classes)
        else:
            raise NotImplementedError

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        out = self.aspp(x)
        out = self.decoder(out, low_level_feat)
        out = F.interpolate(out, size=input.size()[2:], mode="bilinear", align_corners=True)
        return out


class LUSegUNet(nn.Module):
    def __init__(self, num_channel=512, num_classes=5):
        super(LUSegUNet, self).__init__()
        self.conv_ini = DownConv(3, int(num_channel/16))
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.down_conv1 = DownConv(int(num_channel/16), int(num_channel/8))
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.down_conv2 = DownConv(int(num_channel/8), int(num_channel/4))
        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.down_conv3 = DownConv(int(num_channel/4), int(num_channel/2))
        self.mpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.down_conv4 = DownConv(int(num_channel/2), num_channel)

        self.up_conv4 = UpConv(in_plane=num_channel, skip_plane=int(num_channel/2), num_plane=int(num_channel/2))
        self.up_conv3 = UpConv(in_plane=int(num_channel/2), skip_plane=int(num_channel/4), num_plane=int(num_channel/4))
        self.up_conv2 = UpConv(in_plane=int(num_channel/4), skip_plane=int(num_channel/8), num_plane=int(num_channel/8))
        self.up_conv1 = UpConv(in_plane=int(num_channel/8), skip_plane=int(num_channel/16), num_plane=int(num_channel/16))

        self.out_conv = nn.Conv2d(int(num_channel/16), num_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.conv_ini(x)  # 224

        x1 = self.mpool1(x0)  # 112
        x1 = self.down_conv1(x1)

        x2 = self.mpool2(x1)  # 56
        x2 = self.down_conv2(x2)

        x3 = self.mpool3(x2)  # 28
        x3 = self.down_conv3(x3)

        x4 = self.mpool4(x3)  # 14
        x4 = self.down_conv4(x4)

        out = self.up_conv4(x4, x3)
        out = self.up_conv3(out, x2)
        out = self.up_conv2(out, x1)
        out = self.up_conv1(out, x0)

        out = self.out_conv(out)
        
        return out


class LUSegUNet_IN(nn.Module):
    def __init__(self, num_channel=512, num_classes=5):
        super(LUSegUNet_IN, self).__init__()
        self.conv_ini = DownConvIN(3, int(num_channel / 16), IN=True)
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.down_conv1 = DownConvIN(int(num_channel / 16), int(num_channel / 8), IN=True)
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.down_conv2 = DownConvIN(int(num_channel / 8), int(num_channel / 4), IN=False)
        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.down_conv3 = DownConvIN(int(num_channel / 4), int(num_channel / 2), IN=False)
        self.mpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.down_conv4 = DownConvIN(int(num_channel / 2), num_channel, IN=False)

        self.up_conv4 = UpConv(in_plane=num_channel, skip_plane=int(num_channel / 2), num_plane=int(num_channel / 2))
        self.up_conv3 = UpConv(in_plane=int(num_channel / 2), skip_plane=int(num_channel / 4),
                               num_plane=int(num_channel / 4))
        self.up_conv2 = UpConv(in_plane=int(num_channel / 4), skip_plane=int(num_channel / 8),
                               num_plane=int(num_channel / 8))
        self.up_conv1 = UpConv(in_plane=int(num_channel / 8), skip_plane=int(num_channel / 16),
                               num_plane=int(num_channel / 16))

        self.out_conv = nn.Conv2d(int(num_channel / 16), num_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.conv_ini(x)  # 224

        x1 = self.mpool1(x0)  # 112
        x1 = self.down_conv1(x1)

        x2 = self.mpool2(x1)  # 56
        x2 = self.down_conv2(x2)

        x3 = self.mpool3(x2)  # 28
        x3 = self.down_conv3(x3)

        x4 = self.mpool4(x3)  # 14
        x4 = self.down_conv4(x4)

        out = self.up_conv4(x4, x3)
        out = self.up_conv3(out, x2)
        out = self.up_conv2(out, x1)
        out = self.up_conv1(out, x0)

        out = self.out_conv(out)

        return out
