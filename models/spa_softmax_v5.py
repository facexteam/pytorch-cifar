'''Large margin softmax in PyTorch.
@author: zhaoyafei
'''
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class SpaSoftmax_v5(nn.Module):
    def __init__(self, embedding_net, output_size=10, scale=16, m=30):
        super(SpaSoftmax_v5, self).__init__()
        assert (output_size > 1 and
                scale >= 1.0 and
                m >= 0 and m <= 180)

        self.input_size = sum(embedding_net.output_shape)
        assert(self.input_size > 1)

        self.scale = scale
        # self.m = m * np.pi / 180.0 / np.pi
        self.m = m
        self.nm = m / 180.0
        self.output_size = output_size

        self.embedding_net = embedding_net
        self.linear = nn.Linear(self.input_size, self.output_size, bias=False)

    def get_fc_weights(self):
        wt = self.linear.weight.clone().detach()
        return wt

    def set_fc_weights(self, wt):
        self.linear.weight.data.copy_(wt)

    def set_fc_weights_to_ones(self):
        self.linear.weight.data.fill_(1.0)

    def forward(self, x, targets):
        embedding = self.embedding_net(x)

        # print('\n===> In LargeMarginModule_cosface.forward()\n')
        # print('---> emb (before norm):\n', embedding)
        # print('---> emb[j].norm (before norm):\n', embedding.norm(dim=1))
        # print('---> weight (before norm):\n', self.linear.weight)
        # print('---> weight[j].norm (before norm):\n',
        #       self.linear.weight.norm(dim=1))

        # do L2 normalization
        embedding = F.normalize(embedding, dim=1)
        # print('---> emb (after norm):\n', embedding)
        # print('---> emb[j].norm (after norm):\n', embedding.norm(dim=1))

        weight = F.normalize(self.linear.weight, dim=1)
        # print('---> weight (after norm):\n', weight)
        # print('---> weight[j].norm (after norm):\n',
        #       weight.norm(dim=1))

        cos_theta = F.linear(embedding, weight).clamp(-1, 1)
        # cos_theta = cos_theta.clamp(-1, 1)
        # print('---> cos_theta:\n', cos_theta)

        theta = cos_theta.acos() / np.pi
        theta = theta.clamp(0, 1)
        # print('---> theta:\n', theta*180)

        if self.nm > 0:
            one_hot = torch.zeros_like(cos_theta)
            one_hot.scatter_(1, targets.view(-1, 1), 1)
            theta = theta + one_hot * self.nm
            # print('---> biased theta:\n', theta*180)

        output = theta * (-self.scale)
        # print('---> output (s*(-biased_norm_theta)):\n', output)

        # minus_theta = - theta.clamp(0, 1)

        # if self.nm > 0:
        #     one_hot = torch.zeros_like(cos_theta)

        #     # for i in range(x.shape[0]):
        #     #     one_hot[i][targets[i]] = self.nm*self.scale
        #     one_hot.scatter_(1, targets.view(-1, 1), 1)

        #     biased_minus_theta = minus_theta - one_hot * self.nm

        #     output = biased_minus_theta * self.scale

        #     # print('---> output (biased_cos_theta):\n', output)
        #     # print(
        #     #     '---> output[j].norm (biased_cos_theta):\n',
        #     #     output.norm(dim=1))
        # else:
        #     output = minus_theta * self.scale

        return output, cos_theta

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'input_size=' + str(self.input_size) \
            + ', output_size=' + str(self.output_size) \
            + ', scale=' + str(self.scale) \
            + ', m=' + str(self.m) \
            + ')'


if __name__ == '__main__':

    class IdentityModule(nn.Module):
        def __init__(self, output_size=64):
            super(IdentityModule, self).__init__()
            self.output_shape = (output_size,)

        def forward(self, x):
            return x

    def infer(net, data, targets):
        print('===> input data: \n', data)
        print('===> targets: \n', targets)

        pred, cos_theta = net(data, targets)
        print('===> output of net(data):\n')
        print('---> pred (s*biased_norm_theta): \n', pred)
        print('---> cos_theta: \n', cos_theta)

    emb_size = 20
    output_size = 10
    scale = 8

    #dummpy_data = torch.ones(3, emb_size)
    dummpy_data = torch.zeros(3, emb_size)
    for i, row in enumerate(dummpy_data):
        # print('row[{}]: {}'.format(i, row))
        for j in range(len(row)):
            if j % 2 == 0:
                row[j] = 1

        # print('row[{}]: {}'.format(i, row))
    dummpy_targets = torch.tensor([0, 1, 2])

    print('\n#=============================')
    print('\n===> Testing SpaSoftmax_v5 net with dummy data')
    net = IdentityModule(emb_size)
    net = SpaSoftmax_v5(net, output_size, scale)
    net.set_fc_weights_to_ones()
    print('net:\n', net)

    infer(net, dummpy_data, dummpy_targets)

