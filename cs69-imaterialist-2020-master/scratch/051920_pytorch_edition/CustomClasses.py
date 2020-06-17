import torch
import torch.nn as nn
import torch.nn.functional as F
print("PyTorch Version: %s" % torch.__version__)
print("Number of GPUs detected: %d" % torch.cuda.device_count())

class Mish(nn.Module):
    @staticmethod
    def mish(input):
        return input * torch.tanh(F.softplus(input));

    def __init__(self):
        super().__init__();

    def forward(self, input):
        return Mish.mish(input);

class SeparableConv2d(nn.Module):
    """ Custom Sub Building Block for AttributesNetwork """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__();

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias);
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias);

    def forward(self, x):
        x = self.conv1(x);
        x = self.pointwise(x);
        return x;

class Block(nn.Module):
    """ Convolutional Blocks for Attribute Network """
    def __init__(self, in_filters, out_filters, reps, strides=1, activation=None):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False);
            self.skipbn = nn.BatchNorm2d(out_filters);
        else:
            self.skip = None;

        activation = nn.ReLU() if activation is None else activation;
        rep = [];
        rep.append(activation);
        rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False));
        rep.append(nn.BatchNorm2d(out_filters));
        filters = out_filters;

        for i in range(reps - 1):
            rep.append(activation);
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False));
            rep.append(nn.BatchNorm2d(filters));

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1));
        self.rep = nn.Sequential(*rep);

    def forward(self, inp):
        x = self.rep(inp);
        if self.skip is not None:
            skip = self.skip(inp);
            skip = self.skipbn(skip);
        else:
            skip = inp;
        x += skip;
        return x;


class AttributesNetwork(nn.Module):
    def __init__(self, num_classes):
        super(AttributesNetwork, self).__init__();
        self.num_classes = num_classes;

        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.mish = Mish()

        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(128)

        self.block1 = Block(128, 256, 2, 2)
        self.block2 = Block(256, 256, 3, 1)
        self.block3 = Block(256, 256, 3, 1)
        self.block4 = Block(256, 256, 3, 1)
        self.block5 = Block(256, 256, 3, 1)
        self.block6 = Block(256, 256, 3, 1)
        self.block7 = Block(256, 384, 2, 2)

        self.conv3 = SeparableConv2d(384, 512, 3, stride=1, padding=0, bias=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.mish(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        x = self.mish(x)
        x = self.conv3(x)

        x = self.mish(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        result = self.fc(x)

        return torch.sigmoid(result);


class Block4Hourglass(nn.Module):
    """ Custom building block """
    def __init__(self, depth, channel):
        super(Block4Hourglass, self).__init__()
        self.depth = depth
        hg = []
        for _ in range(self.depth):
            hg.append([
                Block(channel, channel, 3, 1, activation=Mish()),
                Block(channel, channel, 2, 2, activation=Mish()),
                Block(channel, channel, 3, 1, activation=Mish())
            ])
        hg[0].append(Block(channel, channel, 3, 1, activation=Mish()))
        hg = [nn.ModuleList(h) for h in hg]
        self.hg = nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = self.hg[n - 1][1](up1)

        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)

        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x);


class StackedHourglass(nn.Module):
    """ Complete CNN Architecture 1 """
    def __init__(self, num_classes):
        super(StackedHourglass, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 128, 3, 2, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(128)
        self.mish = Mish()

        self.conv2 = nn.Conv2d(128, 256, 3, 1, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(256)

        self.block1 = Block4Hourglass(4, 256)
        self.bn3 = nn.BatchNorm2d(256)
        self.block2 = Block4Hourglass(4, 256)

        self.sigmoid = nn.Sigmoid()

        self.conv3 = nn.Conv2d(256, num_classes, 1, bias=True)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.mish(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.mish(x)

        out1 = self.block1(x)
        x = self.bn3(out1)
        x = self.mish(x)
        out2 = self.block2(x)

        r = self.sigmoid(out1 + out2)
        r = F.interpolate(r, scale_factor=2)

        return self.conv3(r)

