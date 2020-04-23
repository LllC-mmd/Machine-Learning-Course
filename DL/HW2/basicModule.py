import torch
import torch.nn as nn


class BasicResBlock(nn.Module):

    def __init__(self, in_plane, num_plane, stride=1, downsample=None):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_plane, out_channels=num_plane, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_plane)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_plane, out_channels=num_plane, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_plane)
        self.downsample = downsample

    def forward(self, x):
        # [H, W] -> [H/s, W/s]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # [H/s, W/s] -> [H/s, W/s]
        out = self.conv2(out)
        out = self.bn2(out)

        # The residual and the output must be of the same dimensions
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out += identity
        out = self.relu(out)

        return out


class SEResBlock(nn.Module):

    def __init__(self, in_plane, num_plane, stride=1, reduction=16, downsample=None):
        super(SEResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_plane, out_channels=num_plane, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_plane)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_plane, out_channels=num_plane, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_plane)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.SEfc = nn.Sequential(
            nn.Linear(num_plane, int(num_plane/reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(num_plane/reduction), num_plane, bias=False),
            nn.Sigmoid()
        )
        self.downsample = downsample

    def forward(self, x):
        # [H, W] -> [H/s, W/s]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # [H/s, W/s] -> [H/s, W/s]
        out = self.conv2(out)
        out = self.bn2(out)
        channel_att = self.avg_pool(out)
        channel_att = torch.flatten(channel_att, start_dim=1)
        channel_att = self.SEfc(channel_att)
        channel_att = channel_att.view([channel_att.size(0), channel_att.size(1), 1, 1])
        out = out * channel_att.expand_as(out)
        # The residual and the output must be of the same dimensions
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out += identity
        out = self.relu(out)

        return out


def downSamplingChoice(in_plane, out_plane, stride):
    if (stride != 1) or (in_plane != out_plane):
        downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_plane, out_channels=out_plane, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_plane)
        )
    else:
        downsample = None
    return downsample


class AttentionConvBlockL3(nn.Module):

    def __init__(self, size, in_plane, num_plane, stride=2):
        super(AttentionConvBlockL3, self).__init__()
        self.inplane = in_plane
        self.plane_l1 = int(num_plane)
        self.plane_l2 = int(num_plane)
        self.plane_l3 = int(num_plane)
        # Feature Extraction Part
        self.trunk = nn.Sequential(
            BasicResBlock(in_plane, self.plane_l1, downsample=downSamplingChoice(in_plane, self.plane_l1, 1)),
            BasicResBlock(self.plane_l1, self.plane_l2, downsample=downSamplingChoice(self.plane_l1, self.plane_l2, 1))
        )
        # Attention Masking Part using Fully Convolutional operation
        # -----downsampling operation and skip-connection operation
        self.downsample1 = BasicResBlock(self.inplane, self.plane_l1, stride=1, downsample=downSamplingChoice(self.inplane, self.plane_l1, 1))
        self.skip_connect1 = BasicResBlock(self.plane_l1, self.plane_l3, stride=1, downsample=downSamplingChoice(self.plane_l1, self.plane_l3, 1))
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.downsample2 = BasicResBlock(self.plane_l1, self.plane_l2, stride=1, downsample=downSamplingChoice(self.plane_l1, self.plane_l2, 1))
        # self.skip_connect2 = BasicResBlock(self.plane_l2, self.plane_l3, stride=1, downsample=downSamplingChoice(self.plane_l2, self.plane_l3, 1))
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.downsample3 = BasicResBlock(self.plane_l2, self.plane_l3, stride=1, downsample=downSamplingChoice(self.plane_l2, self.plane_l3, 1))
        self.skip_connect3 = BasicResBlock(self.plane_l3, self.plane_l3, stride=1, downsample=downSamplingChoice(self.plane_l3, self.plane_l3, 1))
        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # -----upsampling operation
        self.upsample3 = nn.UpsamplingBilinear2d(size=(int(size/4), int(size/4)))
        self.upsample2 = nn.UpsamplingBilinear2d(size=(int(size/2), int(size/2)))
        self.upsample1 = nn.UpsamplingBilinear2d(size=(size, size))
        # -----normalization operation
        self.out_mask = nn.Sequential(
            nn.BatchNorm2d(self.plane_l3),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.plane_l3, self.plane_l3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.plane_l3),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.plane_l3, self.plane_l3, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()   # using sigmoid function for mixed attention
        )

    def forward(self, x):
        out_trunk = self.trunk(x)
        # [ip, 56, 56] -> [p1, 56, 56]
        out = self.downsample1(x)
        skip_c1 = self.skip_connect1(out)
        # [p1, 56, 56] -> [p1, 28, 28]
        out = self.mpool1(out)
        # [p1, 28, 28] -> [p2, 28, 28]
        out = self.downsample2(out)
        # skip_c2 = self.skip_connect2(out)
        # [p2, 28, 28] -> [p2, 14, 14]
        out = self.mpool2(out)
        # [p2, 14, 14] -> [p3, 14, 14]
        out = self.downsample3(out)
        skip_c3 = self.skip_connect3(out)
        # [p3, 14, 14] -> [p3, 7, 7]
        out = self.mpool3(out)
        # [p3, 7, 7] -> [p3, 14, 14]
        out = self.upsample3(out) + skip_c3
        # [p3, 14, 14] -> [p3, 28, 28]
        out = self.upsample2(out)
        # [p3, 28, 28] -> [p3, 56, 56]
        out = self.upsample1(out) + skip_c1
        feature_mask = self.out_mask(out)
        out = (1 + feature_mask) * out_trunk
        return out


class AttentionConvSEBlockL3(nn.Module):

    def __init__(self, size, in_plane, num_plane, stride=2):
        super(AttentionConvSEBlockL3, self).__init__()
        self.inplane = in_plane
        self.plane_l1 = int(num_plane)
        self.plane_l2 = int(num_plane)
        self.plane_l3 = int(num_plane)
        # Feature Extraction Part
        self.trunk = nn.Sequential(
            SEResBlock(in_plane, self.plane_l1, downsample=downSamplingChoice(in_plane, self.plane_l1, 1)),
            SEResBlock(self.plane_l1, self.plane_l2, downsample=downSamplingChoice(self.plane_l1, self.plane_l2, 1))
        )
        # Attention Masking Part using Fully Convolutional operation
        # -----downsampling operation and skip-connection operation
        self.downsample1 = SEResBlock(self.inplane, self.plane_l1, stride=1, downsample=downSamplingChoice(self.inplane, self.plane_l1, 1))
        self.skip_connect1 = SEResBlock(self.plane_l1, self.plane_l3, stride=1, downsample=downSamplingChoice(self.plane_l1, self.plane_l3, 1))
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.downsample2 = SEResBlock(self.plane_l1, self.plane_l2, stride=1, downsample=downSamplingChoice(self.plane_l1, self.plane_l2, 1))
        # self.skip_connect2 = SEResBlock(self.plane_l2, self.plane_l3, stride=1, downsample=downSamplingChoice(self.plane_l2, self.plane_l3, 1))
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.downsample3 = SEResBlock(self.plane_l2, self.plane_l3, stride=1, downsample=downSamplingChoice(self.plane_l2, self.plane_l3, 1))
        self.skip_connect3 = SEResBlock(self.plane_l3, self.plane_l3, stride=1, downsample=downSamplingChoice(self.plane_l3, self.plane_l3, 1))
        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # -----upsampling operation
        self.upsample3 = nn.UpsamplingBilinear2d(size=(int(size/4), int(size/4)))
        self.upsample2 = nn.UpsamplingBilinear2d(size=(int(size/2), int(size/2)))
        self.upsample1 = nn.UpsamplingBilinear2d(size=(size, size))
        # -----normalization operation
        self.out_mask = nn.Sequential(
            nn.BatchNorm2d(self.plane_l3),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.plane_l3, self.plane_l3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.plane_l3),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.plane_l3, self.plane_l3, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()   # using sigmoid function for mixed attention
        )

    def forward(self, x):
        out_trunk = self.trunk(x)
        # [ip, 56, 56] -> [p1, 56, 56]
        out = self.downsample1(x)
        skip_c1 = self.skip_connect1(out)
        # [p1, 56, 56] -> [p1, 28, 28]
        out = self.mpool1(out)
        # [p1, 28, 28] -> [p2, 28, 28]
        out = self.downsample2(out)
        # skip_c2 = self.skip_connect2(out)
        # [p2, 28, 28] -> [p2, 14, 14]
        out = self.mpool2(out)
        # [p2, 14, 14] -> [p3, 14, 14]
        out = self.downsample3(out)
        skip_c3 = self.skip_connect3(out)
        # [p3, 14, 14] -> [p3, 7, 7]
        out = self.mpool3(out)
        # [p3, 7, 7] -> [p3, 14, 14]
        out = self.upsample3(out) + skip_c3
        # [p3, 14, 14] -> [p3, 28, 28]
        out = self.upsample2(out)
        # [p3, 28, 28] -> [p3, 56, 56]
        out = self.upsample1(out) + skip_c1
        feature_mask = self.out_mask(out)
        out = (1 + feature_mask) * out_trunk
        return out


class AttentionConvBlockL2(nn.Module):

    def __init__(self, size, in_plane, num_plane, stride=2):
        super(AttentionConvBlockL2, self).__init__()
        self.inplane = in_plane
        self.plane_l1 = int(num_plane)
        self.plane_l2 = int(num_plane)
        # Feature Extraction Part
        self.trunk = nn.Sequential(
            BasicResBlock(in_plane, self.plane_l1, downsample=downSamplingChoice(in_plane, self.plane_l1, 1)),
            BasicResBlock(self.plane_l1, self.plane_l2, downsample=downSamplingChoice(self.plane_l1, self.plane_l2, 1))
        )
        # Attention Masking Part using Fully Convolutional operation
        # -----downsampling operation and skip-connection operation
        self.downsample1 = BasicResBlock(self.inplane, self.plane_l1, stride=1, downsample=downSamplingChoice(self.inplane, self.plane_l1, 1))
        self.skip_connect1 = BasicResBlock(self.plane_l1, self.plane_l2, stride=1, downsample=downSamplingChoice(self.plane_l1, self.plane_l2, 1))
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.downsample2 = BasicResBlock(self.plane_l1, self.plane_l2, stride=1, downsample=downSamplingChoice(self.plane_l1, self.plane_l2, 1))
        #self.skip_connect2 = BasicResBlock(self.plane_l2, self.plane_l2, stride=1, downsample=downSamplingChoice(self.plane_l2, self.plane_l2, 1))
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # -----upsampling operation
        self.upsample2 = nn.UpsamplingBilinear2d(size=(int(size/2), int(size/2)))
        self.upsample1 = nn.UpsamplingBilinear2d(size=(size, size))
        # -----normalization operation
        self.out_mask = nn.Sequential(
            nn.BatchNorm2d(self.plane_l2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.plane_l2, self.plane_l2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.plane_l2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.plane_l2, self.plane_l2, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()   # using sigmoid function for mixed attention
        )

    def forward(self, x):
        out_trunk = self.trunk(x)
        # [ip, 28, 28] -> [p1, 28, 28]
        out = self.downsample1(x)
        skip_c1 = self.skip_connect1(out)
        # [p1, 28, 28] -> [p1, 14, 14]
        out = self.mpool1(out)
        # [p1, 14, 14] -> [p2, 14, 14]
        out = self.downsample2(out)
        # skip_c2 = self.skip_connect2(out)
        # [p2, 14, 14] -> [p2, 7, 7]
        out = self.mpool2(out)
        # [p2, 7, 7] -> [p2, 14, 14]
        out = self.upsample2(out)
        # [p2, 14, 14] -> [p2, 28, 28]
        out = self.upsample1(out) + skip_c1
        feature_mask = self.out_mask(out)
        out = (1 + feature_mask) * out_trunk
        return out


class AttentionConvSEBlockL2(nn.Module):

    def __init__(self, size, in_plane, num_plane, stride=2):
        super(AttentionConvSEBlockL2, self).__init__()
        self.inplane = in_plane
        self.plane_l1 = int(num_plane)
        self.plane_l2 = int(num_plane)
        # Feature Extraction Part
        self.trunk = nn.Sequential(
            SEResBlock(in_plane, self.plane_l1, downsample=downSamplingChoice(in_plane, self.plane_l1, 1)),
            SEResBlock(self.plane_l1, self.plane_l2, downsample=downSamplingChoice(self.plane_l1, self.plane_l2, 1))
        )
        # Attention Masking Part using Fully Convolutional operation
        # -----downsampling operation and skip-connection operation
        self.downsample1 = SEResBlock(self.inplane, self.plane_l1, stride=1, downsample=downSamplingChoice(self.inplane, self.plane_l1, 1))
        self.skip_connect1 = SEResBlock(self.plane_l1, self.plane_l2, stride=1, downsample=downSamplingChoice(self.plane_l1, self.plane_l2, 1))
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.downsample2 = SEResBlock(self.plane_l1, self.plane_l2, stride=1, downsample=downSamplingChoice(self.plane_l1, self.plane_l2, 1))
        # self.skip_connect2 = SEResBlock(self.plane_l2, self.plane_l2, stride=1, downsample=downSamplingChoice(self.plane_l2, self.plane_l2, 1))
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # -----upsampling operation
        self.upsample2 = nn.UpsamplingBilinear2d(size=(int(size/2), int(size/2)))
        self.upsample1 = nn.UpsamplingBilinear2d(size=(size, size))
        # -----normalization operation
        self.out_mask = nn.Sequential(
            nn.BatchNorm2d(self.plane_l2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.plane_l2, self.plane_l2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.plane_l2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.plane_l2, self.plane_l2, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()   # using sigmoid function for mixed attention
        )

    def forward(self, x):
        out_trunk = self.trunk(x)
        # [ip, 28, 28] -> [p1, 28, 28]
        out = self.downsample1(x)
        skip_c1 = self.skip_connect1(out)
        # [p1, 28, 28] -> [p1, 14, 14]
        out = self.mpool1(out)
        # [p1, 14, 14] -> [p2, 14, 14]
        out = self.downsample2(out)
        # skip_c2 = self.skip_connect2(out)
        # [p2, 14, 14] -> [p2, 7, 7]
        out = self.mpool2(out)
        # [p2, 7, 7] -> [p2, 14, 14]
        out = self.upsample2(out)
        # [p2, 14, 14] -> [p2, 28, 28]
        out = self.upsample1(out) + skip_c1
        feature_mask = self.out_mask(out)
        out = (1 + feature_mask) * out_trunk
        return out


class AttentionConvBlockL1(nn.Module):

    def __init__(self, size, in_plane, num_plane, stride=2):
        super(AttentionConvBlockL1, self).__init__()
        self.inplane = in_plane
        self.plane_l1 = int(num_plane)
        # Feature Extraction Part
        self.trunk = nn.Sequential(
            BasicResBlock(in_plane, self.plane_l1, downsample=downSamplingChoice(in_plane, self.plane_l1, 1)),
            BasicResBlock(self.plane_l1, self.plane_l1, downsample=downSamplingChoice(self.plane_l1, self.plane_l1, 1))
        )
        # self.trunk = BasicResBlock(in_plane, self.plane_l1, downsample=downSamplingChoice(in_plane, self.plane_l1, 1))
        # Attention Masking Part using Fully Convolutional operation
        # -----downsampling operation and skip-connection operation
        self.downsample1 = BasicResBlock(self.inplane, self.plane_l1, stride=1, downsample=downSamplingChoice(self.inplane, self.plane_l1, 1))
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # -----upsampling operation
        self.upsample1 = nn.UpsamplingBilinear2d(size=(size, size))
        # -----normalization operation
        self.out_mask = nn.Sequential(
            nn.BatchNorm2d(self.plane_l1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.plane_l1, self.plane_l1, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()   # using sigmoid function for mixed attention
        )

    def forward(self, x):
        out_trunk = self.trunk(x)
        # [ip, 14, 14] -> [p1, 14, 14]
        out = self.downsample1(x)
        # [p1, 14, 14] -> [p1, 7, 7]
        out = self.mpool1(out)
        # [p1, 7, 7] -> [p1, 14, 14]
        out = self.upsample1(out)
        feature_mask = self.out_mask(out)
        out = (1 + feature_mask) * out_trunk
        return out


class AttentionConvSEBlockL1(nn.Module):

    def __init__(self, size, in_plane, num_plane, stride=2):
        super(AttentionConvSEBlockL1, self).__init__()
        self.inplane = in_plane
        self.plane_l1 = int(num_plane)
        # Feature Extraction Part
        self.trunk = nn.Sequential(
            SEResBlock(in_plane, self.plane_l1, downsample=downSamplingChoice(in_plane, self.plane_l1, 1)),
            SEResBlock(self.plane_l1, self.plane_l1, downsample=downSamplingChoice(self.plane_l1, self.plane_l1, 1))
        )
        # self.trunk = SEResBlock(in_plane, self.plane_l1, downsample=downSamplingChoice(in_plane, self.plane_l1, 1))
        # Attention Masking Part using Fully Convolutional operation
        # -----downsampling operation and skip-connection operation
        self.downsample1 = SEResBlock(self.inplane, self.plane_l1, stride=1, downsample=downSamplingChoice(self.inplane, self.plane_l1, 1))
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # -----upsampling operation
        self.upsample1 = nn.UpsamplingBilinear2d(size=(size, size))
        # -----normalization operation
        self.out_mask = nn.Sequential(
            nn.BatchNorm2d(self.plane_l1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.plane_l1, self.plane_l1, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()   # using sigmoid function for mixed attention
        )

    def forward(self, x):
        out_trunk = self.trunk(x)
        # [ip, 14, 14] -> [p1, 14, 14]
        out = self.downsample1(x)
        # [p1, 14, 14] -> [p1, 7, 7]
        out = self.mpool1(out)
        # [p1, 7, 7] -> [p1, 14, 14]
        out = self.upsample1(out)
        feature_mask = self.out_mask(out)
        out = (1 + feature_mask) * out_trunk
        return out
