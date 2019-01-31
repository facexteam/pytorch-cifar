'''Large margin softmax in PyTorch.
@author: zhaoyafei
'''
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import BasicBlock
import numpy as np


class SpaSoftmax_v3(nn.Module):
    def __init__(self, embedding_net, output_size=10, scale=32, m=30):
        super(SpaSoftmax_v3, self).__init__()
        self.input_size = sum(embedding_net.output_shape)
        self.scale = scale
        # self.m = m * np.pi / 180.0 / np.pi
        self.m = m / 180.0

        assert(self.m >= 0 and self.m<=1.0)

        self.embedding_net = embedding_net
        self.linear = nn.Linear(self.input_size, output_size, bias=False)

    def forward(self, x, targets):
        embedding = self.embedding_net(x)

        # print('\n===> In LargeMarginModule_cosface.forward()\n')
        # print('---> emb (before norm): ', embedding)
        # print('---> emb[j].norm (before norm): ', embedding.norm(dim=1))
        # print('---> weight (before norm): ', self.linear.weight)
        # print('---> weight[j].norm (before norm): ',
        #       self.linear.weight.norm(dim=1))

        # do L2 normalization
        embedding = F.normalize(embedding, dim=1)
        # print('---> emb (after norm): ', embedding)
        # print('---> emb[j].norm (after norm): ', embedding.norm(dim=1))

        weight = F.normalize(self.linear.weight, dim=1)
        # print('---> weight (after norm): ', weight)
        # print('---> weight[j].norm (after norm): ',
        #       weight.norm(dim=1))

        cos_theta = F.linear(embedding, weight)

        # print('---> cos_theta (fc_output): ', cos_theta)
        # print('---> cos_theta[j].norm (fc_output): ',
        #       cos_theta.norm(dim=1))

        theta = cos_theta.acos() / np.pi
        
        minus_theta = 1.0 - theta.clamp(-1,1)

        one_hot = torch.zeros_like(cos_theta)

        # for i in range(x.shape[0]):
        #     one_hot[i][targets[i]] = self.m*self.scale
        one_hot.scatter_(1, targets.view(-1, 1), 1)

        biased_minus_theta = minus_theta - one_hot * self.m

        output = biased_minus_theta * self.scale

        # print('---> output (biased_cos_theta): ', output)
        # print(
        #     '---> output[j].norm (biased_cos_theta): ',
        #     output.norm(dim=1))

        return output, cos_theta

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'input_size=' + str(self.input_size) \
            + ', output_size=' + str(self.output_size) \
            + ', m=' + str(self.m) + ')'
