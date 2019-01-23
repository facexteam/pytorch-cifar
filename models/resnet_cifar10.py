'''ResNet for cifar10 in PyTorch.
@author: zhaoyafei

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import BasicBlock


class ResNet_cifar10(nn.Module):
    def __init__(self, block, num_layers, num_classes=10):
        super(ResNet_cifar10, self).__init__()
        self.in_planes = 16

        if (num_layers-2) % 6 == 0:
            n = (num_layers-2)//6
            num_blocks = [2*n, 2*n, 2*n]
        else:
            raise ValueError(
                "no experiments done on num_layers {}, you can do it yourself".format(num_layers))

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet20_cifar10():
    return ResNet_cifar10(BasicBlock, 20)


def ResNet32_cifar10():
    return ResNet_cifar10(BasicBlock, 32)


def ResNet44_cifar10():
    return ResNet_cifar10(BasicBlock, 44)


def ResNet56_cifar10():
    return ResNet_cifar10(BasicBlock, 56)


def ResNet110_cifar10():
    return ResNet_cifar10(BasicBlock, 110)


def ResNet1202_cifar10():
    return ResNet_cifar10(BasicBlock, 1202)


def test():
    net = ResNet20_cifar10()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
