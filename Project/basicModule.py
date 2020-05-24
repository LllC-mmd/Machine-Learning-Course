import torch
import torch.nn as nn
import torch.nn.functional as F


def downSamplingChoice(in_plane, out_plane, stride):
    if (stride != 1) or (in_plane != out_plane):
        downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_plane, out_channels=out_plane, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_plane)
        )
    else:
        downsample = None
    return downsample


# ---ResNet block & bottleneck
class ResBlock(nn.Module):
    def __init__(self, in_plane, num_plane, stride=1, downsample=None):
        super(ResBlock, self).__init__()
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


class ResBottleneck(nn.Module):

    def __init__(self, in_plane, num_plane, stride=1, dilation=1, expansion=4, downsample=None):
        super(ResBottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels=in_plane, out_channels=num_plane, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_plane)
        self.conv2 = nn.Conv2d(in_channels=num_plane, out_channels=num_plane, kernel_size=3,
                               stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(num_plane)
        self.conv3 = nn.Conv2d(in_channels=num_plane, out_channels=num_plane*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_plane*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        # [H, W] -> [H, W]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # [H, W] -> [H/s, W/s]
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # [H/s, W/s] -> [H/s, W/s]
        out = self.conv3(out)
        out = self.bn3(out)

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


class SEResBottleneck(nn.Module):
    def __init__(self, in_plane, num_plane, stride=1, dilation=1, expansion=4, reduction=16, downsample=None):
        super(SEResBottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels=in_plane, out_channels=num_plane, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_plane)
        self.conv2 = nn.Conv2d(in_channels=num_plane, out_channels=num_plane, kernel_size=3,
                               stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(num_plane)
        self.conv3 = nn.Conv2d(in_channels=num_plane, out_channels=num_plane*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_plane*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.SEfc = nn.Sequential(
            nn.Linear(num_plane*self.expansion, int(num_plane*self.expansion/reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(num_plane*self.expansion/reduction), num_plane*self.expansion, bias=False),
            nn.Sigmoid()
        )
        self.downsample = downsample

    def forward(self, x):
        # [H, W] -> [H, W]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # [H, W] -> [H/s, W/s]
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # [H/s, W/s] -> [H/s, W/s]
        out = self.conv3(out)
        out = self.bn3(out)

        channel_att = self.avg_pool(out)
        channel_att = torch.flatten(channel_att, start_dim=1)
        channel_att = self.SEfc(channel_att)
        channel_att = channel_att.view([channel_att.size(0), channel_att.size(1), 1, 1])
        out = out*channel_att.expand_as(out)
        # The residual and the output must be of the same dimensions
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out += identity
        out = self.relu(out)

        return out


# ---Residual Attention Net with Squeeze-and-Excitation module
# ------Block is chosen from: ResBottleneck, SEResBottleneck
class AttentionConvSEBlockL3(nn.Module):
    def __init__(self, in_plane, num_plane, block, expansion=4):
        super(AttentionConvSEBlockL3, self).__init__()
        self.in_plane = in_plane
        self.middle_plane = int(num_plane)
        self.out_plane = int(num_plane*expansion)
        self.expansion = expansion
        # Feature Extraction Part
        self.trunk = nn.Sequential(
            block(self.in_plane, self.middle_plane, downsample=downSamplingChoice(self.in_plane, self.out_plane, 1)),
            block(self.out_plane, self.middle_plane, downsample=downSamplingChoice(self.out_plane, self.out_plane, 1))
        )
        # Attention Masking Part using Fully Convolutional operation
        # -----downsampling operation and skip-connection operation
        self.downsample1 = block(self.in_plane, self.middle_plane, downsample=downSamplingChoice(self.in_plane, self.out_plane, 1))
        self.skip_connect1 = block(self.out_plane, self.middle_plane, downsample=downSamplingChoice(self.out_plane, self.out_plane, 1))
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.downsample2 = block(self.out_plane, self.middle_plane, downsample=downSamplingChoice(self.out_plane, self.out_plane, 1))
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.downsample3 = block(self.out_plane, self.middle_plane, downsample=downSamplingChoice(self.out_plane, self.out_plane, 1))
        self.skip_connect3 = block(self.out_plane, self.middle_plane, downsample=downSamplingChoice(self.out_plane, self.out_plane, 1))
        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # -----upsampling operation
        self.upsample3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # -----normalization operation
        self.out_mask = nn.Sequential(
            nn.BatchNorm2d(self.out_plane),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_plane, self.out_plane, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.out_plane),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_plane, self.out_plane, kernel_size=1, stride=1, bias=False),
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


class AttentionConvSEBlockL2(nn.Module):
    def __init__(self, in_plane, num_plane, block, expansion=4):
        super(AttentionConvSEBlockL2, self).__init__()
        self.in_plane = in_plane
        self.middle_plane = int(num_plane)
        self.out_plane = int(num_plane*expansion)
        # Feature Extraction Part
        self.trunk = nn.Sequential(
            block(self.in_plane, self.middle_plane, downsample=downSamplingChoice(self.in_plane, self.out_plane, 1)),
            block(self.out_plane, self.middle_plane, downsample=downSamplingChoice(self.out_plane, self.out_plane, 1))
        )
        # Attention Masking Part using Fully Convolutional operation
        # -----downsampling operation and skip-connection operation
        self.downsample1 = block(self.in_plane, self.middle_plane, downsample=downSamplingChoice(self.in_plane, self.out_plane, 1))
        self.skip_connect1 = block(self.out_plane, self.middle_plane, downsample=downSamplingChoice(self.out_plane, self.out_plane, 1))
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.downsample2 = block(self.out_plane, self.middle_plane, downsample=downSamplingChoice(self.out_plane, self.out_plane, 1))
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # -----upsampling operation
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # -----normalization operation
        self.out_mask = nn.Sequential(
            nn.BatchNorm2d(self.out_plane),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_plane, self.out_plane, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.out_plane),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_plane, self.out_plane, kernel_size=1, stride=1, bias=False),
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


class AttentionConvSEBlockL1(nn.Module):
    def __init__(self, in_plane, num_plane, block, expansion=4):
        super(AttentionConvSEBlockL1, self).__init__()
        self.in_plane = in_plane
        self.middle_plane = int(num_plane)
        self.out_plane = int(num_plane * expansion)
        # Feature Extraction Part
        self.trunk = nn.Sequential(
            block(self.in_plane, self.middle_plane, downsample=downSamplingChoice(self.in_plane, self.out_plane, 1)),
            block(self.out_plane, self.middle_plane, downsample=downSamplingChoice(self.out_plane, self.out_plane, 1))
        )
        # Attention Masking Part using Fully Convolutional operation
        # -----downsampling operation and skip-connection operation
        self.downsample1 = block(self.in_plane, self.middle_plane, downsample=downSamplingChoice(self.in_plane, self.out_plane, 1))
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # -----upsampling operation
        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # -----normalization operation
        self.out_mask = nn.Sequential(
            nn.BatchNorm2d(self.out_plane),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_plane, self.out_plane, kernel_size=1, stride=1, bias=False),
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


# ---Atrous Spatial Pyramid Pooling in DeepLab
class AtrousSPPBranch(nn.Module):
    def __init__(self, in_plane, num_plane, kernel_size, padding, dilation):
        super(AtrousSPPBranch, self).__init__()
        self.aConv = nn.Conv2d(in_plane, num_plane, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(num_plane)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.aConv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out


class AtrousSPP(nn.Module):
    def __init__(self, in_plane, output_stride, num_plane=256):
        super(AtrousSPP, self).__init__()
        if output_stride == 16:
            dilation_rates = [6, 12, 18]
        elif output_stride == 8:
            dilation_rates = [12, 24, 36]
        else:
            raise Exception("Output_stride should be 8 or 16")

        self.aspp_branch1 = AtrousSPPBranch(in_plane, num_plane=num_plane, kernel_size=1, padding=0, dilation=1)
        self.aspp_branch2 = AtrousSPPBranch(in_plane, num_plane=num_plane, kernel_size=3, padding=dilation_rates[0], dilation=dilation_rates[0])
        self.aspp_branch3 = AtrousSPPBranch(in_plane, num_plane=num_plane, kernel_size=3, padding=dilation_rates[1], dilation=dilation_rates[1])
        self.aspp_branch4 = AtrousSPPBranch(in_plane, num_plane=num_plane, kernel_size=3, padding=dilation_rates[2], dilation=dilation_rates[2])

        self.avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                      nn.Conv2d(in_plane, num_plane, kernel_size=1, stride=1, bias=False),
                                      nn.BatchNorm2d(num_plane),
                                      nn.ReLU(inplace=True))
        self.outConv = nn.Conv2d(5*num_plane, num_plane, kernel_size=1, bias=False)
        self.outbn = nn.BatchNorm2d(num_plane)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.init_weight()

    def forward(self, x):
        x1 = self.aspp_branch1(x)
        x2 = self.aspp_branch2(x)
        x3 = self.aspp_branch3(x)
        x4 = self.aspp_branch4(x)
        x5 = self.avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)
        out = torch.cat((x1, x2, x3, x4, x5), dim=1)

        out = self.outConv(out)
        out = self.outbn(out)
        out = self.relu(out)
        out = self.dropout(out)

        return out

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# ---Position Attention Module in DA-Net
class posAtt(nn.Module):
    def __init__(self, in_plane):
        super(posAtt, self).__init__()
        self.num_plane = int(in_plane/8)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.qConv = nn.Conv2d(in_plane, self.num_plane, kernel_size=1)
        self.kConv = nn.Conv2d(in_plane, self.num_plane, kernel_size=1)
        self.vConv = nn.Conv2d(in_plane, in_plane, kernel_size=1)

    def forward(self, x):
        batch_size, c, h, w = x.size()
        # query matrix
        qMat = self.qConv(x)
        qMat = qMat.view(batch_size, self.num_plane, h*w)
        qMat = qMat.transpose(2, 1)   # qMat: [batch_size, h*w, num_plane]
        # key matrix
        kMat = self.kConv(x)
        kMat = kMat.view(batch_size, self.num_plane, h*w)
        # value matrix
        vMat = self.vConv(x)
        vMat = vMat.view(batch_size, c, h*w)
        # score matrix for similarity metric
        score = torch.bmm(qMat, kMat)
        score = torch.softmax(score, dim=-1)

        out = torch.bmm(vMat, score.transpose(2, 1))
        out = out.view(batch_size, c, h, w)
        out = self.gamma * out + x

        return out


# ---DownConv Module & UpConv in U-Net
class DownConv(nn.Module):
    def __init__(self, in_plane, num_plane):
        super(DownConv, self).__init__()
        self.dconv = nn.Sequential(
            nn.Conv2d(in_plane, num_plane, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_plane),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_plane, num_plane, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_plane),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.dconv(x)


class UpConv(nn.Module):
    def __init__(self, in_plane, skip_plane, num_plane):
        super(UpConv, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.skip_connect = nn.Conv2d(in_channels=skip_plane, out_channels=in_plane, kernel_size=1, bias=False)
        self.conv = DownConv(2*in_plane, num_plane)

    def forward(self, x, skip_x):
        out = self.upsample(x)
        out = torch.cat([out, self.skip_connect(skip_x)], dim=1)
        out = self.conv(out)
        return out
