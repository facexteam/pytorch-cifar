'''ResNet for cifar10_fc in PyTorch.
@author: zhaoyafei

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import BasicBlock


class ResNet_cifar10_fc(nn.Module):
    def __init__(self, block, num_layers, output_size=10):
        super(ResNet_cifar10_fc, self).__init__()
        self.in_planes = 16
        self.num_layers = num_layers
        self.output_size = output_size

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
        self.linear = nn.Linear(64*block.expansion, output_size)
        self.output_shape = (output_size, )

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
        # out = out / (out.norm() + self.eps) * self.scale
        out = self.linear(out)
        return out


def ResNet20_cifar10_fc(output_size=32):
    return ResNet_cifar10_fc(BasicBlock, 20, output_size)


def ResNet32_cifar10_fc(output_size=32):
    return ResNet_cifar10_fc(BasicBlock, 32, output_size)


def ResNet44_cifar10_fc(output_size=32):
    return ResNet_cifar10_fc(BasicBlock, 44, output_size)


def ResNet56_cifar10_fc(output_size=32):
    return ResNet_cifar10_fc(BasicBlock, 56, output_size)


def ResNet110_cifar10_fc(output_size=32):
    return ResNet_cifar10_fc(BasicBlock, 110, output_size)


def ResNet1202_cifar10_fc(output_size=32):
    return ResNet_cifar10_fc(BasicBlock, 1202, output_size)


def test(output_size=32):
    net = ResNet20_cifar10_fc(output_size)
    print('===> net: \n', net)

    print('===> test net with random data:')
    y = net(torch.randn(1, 3, 32, 32))
    print('    net output:\n', y)
    print('    net output.shape:\n', y.shape)


# test(32)
