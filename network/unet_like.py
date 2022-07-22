import torch
import torch.nn as nn
import torchvision


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )


class DeConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(DeConvBNReLU, self).__init__(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )


class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x1)
        return x1


class Unet_like(nn.Module):
    # class Unet_like_back(nn.Module):
    def __init__(self, out_channels=32):
        super().__init__()
        self.base_model = torchvision.models.resnet18(True)
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2])
        self.layer2 = ConvBNReLU(in_planes=64, out_planes=128, kernel_size=4, stride=2)
        self.layer3 = ConvBNReLU(in_planes=128, out_planes=256, kernel_size=4, stride=2)
        self.resblock = nn.Sequential(
            *[BasicBlock(inplanes=256, planes=256, stride=2, downsample=nn.MaxPool2d(kernel_size=2)),
              BasicBlock(inplanes=256, planes=256),
              BasicBlock(inplanes=256, planes=256),
              BasicBlock(inplanes=256, planes=256),
              BasicBlock(inplanes=256, planes=256),
              BasicBlock(inplanes=256, planes=256), ])
        self.layer4 = Decoder(in_channels=256, middle_channels=384, out_channels=128)
        self.layer5 = Decoder(in_channels=128, middle_channels=192, out_channels=64)
        self.layer6 = ConvBNReLU(in_planes=64, out_planes=out_channels, kernel_size=1, stride=1)
        self.decode1 = Decoder(32, 96, 32)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        )
        self.conv_last = nn.Conv2d(32, 2, 1)

    def forward(self, input):
        e1 = self.layer1(input)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        f = self.resblock(e3)
        d3 = self.layer4(f, e3)
        d2 = self.layer5(d3, e2)
        x_a = self.layer6(d2)
        if self.training:
            d1 = self.decode1(x_a, e1)
            d0 = self.decode0(d1)
            seg_out = self.conv_last(d0)
            return x_a, seg_out
        else:
            return x_a


class Unet(nn.Module):
    def __init__(self, out_channels, n_class=1):
        super().__init__()

        self.base_model = torchvision.models.resnet18(True)
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2])
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.layer4 = self.base_layers[6]
        self.layer5 = self.base_layers[7]
        self.decode4 = Decoder(512, 256 + 256, 256)
        self.decode3 = Decoder(256, 256 + 128, 256)
        self.decode2 = Decoder(256, 128 + 64, 128)
        self.decode1 = Decoder(128, 64 + 64, 64)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        )
        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        print('input.shape', input.shape)
        e1 = self.layer1(input)  # 64,128,128
        print('e1.shape', e1.shape)
        e2 = self.layer2(e1)  # 64,64,64
        print('e2.shape', e2.shape)
        e3 = self.layer3(e2)  # 128,32,32
        print('e3.shape', e3.shape)
        e4 = self.layer4(e3)  # 256,16,16
        print('e3.shape', e4.shape)
        f = self.layer5(e4)  # 512,8,8
        print('f.shape', f.shape)
        d4 = self.decode4(f, e4)  # 256,16,16
        print('d4.shape', d4.shape)
        d3 = self.decode3(d4, e3)  # 256,32,32
        print('d3.shape', d3.shape)
        d2 = self.decode2(d3, e2)  # 128,64,64
        print('d2.shape', d2.shape)
        d1 = self.decode1(d2, e1)  # 64,128,128
        print('d1.shape', d1.shape)
        d0 = self.decode0(d1)  # 64,256,256
        print('d0.shape', d0.shape)
        out = self.conv_last(d0)  # 1,256,256
        print('out.shape', out.shape)
        exit(3)
        return out