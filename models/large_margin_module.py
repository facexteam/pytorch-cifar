'''ResNet for cifar10_nofc in PyTorch.
@author: zhaoyafei

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import BasicBlock


class LargeMarginModule_CosineLoss(nn.Module):
    def __init__(self, embedding_net, num_classes=10, scale=32, m=0.5, eps=1e-12):
        super(LargeMarginModule_CosineLoss, self).__init__()
        self.in_planes = sum(embedding_net.output_shape)
        self.eps = eps
        self.scale = scale
        self.m = m

        self.embedding_net = embedding_net
        self.linear = nn.Linear(self.in_planes, num_classes, bias=False)

    def forward(self, x, targets, device):
        out = self.embedding_net(x)

        # print('\n===> In LargeMarginModule_CosineLoss.forward()\n')
        # print('---> emb (before norm): ', out)
        # print('---> emb[j].norm (before norm): ', out.norm(dim=1))
        # print('---> weight (before norm): ', self.linear.weight)
        # print('---> weight[j].norm (before norm): ',
        #       self.linear.weight.norm(dim=1))

        # do L2 normalization
        out = F.normalize(out, dim=1)
        # print('---> emb (after norm): ', out)
        # print('---> emb[j].norm (after norm): ', out.norm(dim=1))

        weight = F.normalize(self.linear.weight, dim=1)
        # print('---> weight (after norm): ', weight)
        # print('---> weight[j].norm (after norm): ',
        #       weight.norm(dim=1))

        out_for_predict = F.linear(out, weight) * self.scale
        # print('---> out_for_predict (fc_output): ', out_for_predict)
        # print('---> out_for_predict[j].norm (fc_output): ',
        #       out_for_predict.norm(dim=1))

        shift_mat = torch.zeros_like(out_for_predict)

        for i in range(x.shape[0]):
            shift_mat[i][targets[i]] = self.m*self.scale
        # print('---> shift_mat for twister: ', shift_mat)

        # shift_mat.to(device)

        out_for_loss = out_for_predict - shift_mat
        # print('---> out_for_loss (out_for_predict after twister): ', out_for_loss)
        # print(
        #     '---> out_for_loss[j].norm (out_for_predict after twister): ',
        #     out_for_loss.norm(dim=1))

        return out_for_loss, out_for_predict
