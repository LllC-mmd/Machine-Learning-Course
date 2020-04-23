from torchvision import models
import torch
import torch.nn as nn
from basicModule import *


class LabelSmoothingCEloss(nn.Module):

    def __init__(self, epsilon=0.01, num_class=20):
        super(LabelSmoothingCEloss, self).__init__()
        self.confidence = 1.0 - epsilon
        self.epsilon = epsilon
        self.num_class = num_class

    def forward(self, x, target):
        N = x.size(0)
        smoothed_labels = torch.full(size=(N, self.num_class), fill_value=self.epsilon/(self.num_class-1))
        smoothed_labels.scatter_(dim=1, index=target.unsqueeze(1), value=self.confidence)
        log_prob = nn.functional.log_softmax(x, dim=1)
        loss = -torch.sum(log_prob*smoothed_labels)/N
        return loss


class ResNet18(nn.Module):

    def __init__(self, in_plane=64, num_classes=20):
        super(ResNet18, self).__init__()
        self.inplane = in_plane
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.inplane, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, num_classes)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, num_plane, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.inplane != num_plane):
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.inplane, out_channels=num_plane, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_plane)
            )
        layers = [BasicResBlock(self.inplane, num_plane, stride, downsample)]
        self.inplane = num_plane
        for _ in range(1, blocks):
            layers.append(BasicResBlock(self.inplane, num_plane))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x


class ResAttentionNet(nn.Module):

    def __init__(self, pic_length, in_plane=64, num_classes=20):
        super(ResAttentionNet, self).__init__()
        self.inplane = in_plane
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.inplane, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.inplane),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BasicResBlock(self.inplane, 64, stride=1, downsample=downSamplingChoice(self.inplane, 64, 1)),
            AttentionConvBlockL3(int(pic_length/4), 64, 64, stride=2),
            BasicResBlock(64, 128, stride=2, downsample=downSamplingChoice(64, 128, 2)),
            AttentionConvBlockL2(int(pic_length/8), 128, 128, stride=2),
            BasicResBlock(128, 256, stride=2, downsample=downSamplingChoice(128, 256, 2)),
            AttentionConvBlockL1(int(pic_length/16), 256, 256, stride=2),
            BasicResBlock(256, 512, stride=2, downsample=downSamplingChoice(256, 512, 2))
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(512, num_classes)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.features(x)
        # [128, 7, 7] -> [128, 1, 1]
        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.classifier(out)
        return out


class SEResAttentionNet(nn.Module):

    def __init__(self, pic_length, in_plane=64, num_classes=20):
        super(SEResAttentionNet, self).__init__()
        self.inplane = in_plane
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.inplane, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.inplane),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            SEResBlock(self.inplane, 64, stride=1, downsample=downSamplingChoice(self.inplane, 64, 1)),
            AttentionConvSEBlockL3(int(pic_length/4), 64, 64, stride=2),
            SEResBlock(64, 128, stride=2, downsample=downSamplingChoice(64, 128, 2)),
            AttentionConvSEBlockL2(int(pic_length/8), 128, 128, stride=2),
            SEResBlock(128, 256, stride=2, downsample=downSamplingChoice(128, 256, 2)),
            AttentionConvSEBlockL1(int(pic_length/16), 256, 256, stride=2),
            SEResBlock(256, 512, stride=2, downsample=downSamplingChoice(256, 512, 2))
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(512, num_classes)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.classifier(out)
        return out


def model_A(num_classes, pretrained=True):
    model_resnet = models.resnet18(pretrained=pretrained)
    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, num_classes)
    return model_resnet


def model_B(num_classes, pretrained=False):
    ## your code here
    model_resnet18 = ResNet18(in_plane=64, num_classes=num_classes)
    total_num = sum(p.numel() for p in model_resnet18.parameters())
    trainable_num = sum(p.numel() for p in model_resnet18.parameters() if p.requires_grad)
    print("Total parameter of model B: ", total_num, " Trainable parameter of model B: ", trainable_num)
    return model_resnet18


def model_C(num_classes, pretrained=False):
    ## your code here
    model_ResAttNet = ResAttentionNet(pic_length=224, in_plane=64, num_classes=num_classes)
    total_num = sum(p.numel() for p in model_ResAttNet.parameters())
    trainable_num = sum(p.numel() for p in model_ResAttNet.parameters() if p.requires_grad)
    print("Total parameter of model C: ", total_num, " Trainable parameter of model C: ", trainable_num)
    return model_ResAttNet
