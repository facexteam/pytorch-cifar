'''ResNet for cifar10_nofc in PyTorch.
@author: zhaoyafei

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import BasicBlock


class LargeMarginModule_CosineLoss(nn.Module):
    def __init__(self, embedding_net, num_classes=10, scale=32, m=0.5):
        super(LargeMarginModule_CosineLoss, self).__init__()
        self.in_planes = sum(embedding_net.output_shape)
        self.eps = eps
        self.scale = scale
        self.m = m*scale

        self.embedding_net = embedding_net
        self.linear = nn.Linear(self.in_planes, num_classes)

    def forward(self, x, targets):
        out = self.embedding_net(x)

        # do L2 normalization
        # for ele in out:
        #     ele = ele / (ele.norm() + self.eps) * self.scale
        out = F.normalize(out, dim=1) * self.scale
        out_for_predict = self.linear(out)

        for i in range(x.shape[0]):
            out[i][targets[i]] -= self.m

        out_for_loss = self.linear(out)

        return out_for_loss, out_for_predict
