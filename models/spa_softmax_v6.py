'''Large margin softmax in PyTorch.
@author: zhaoyafei
'''
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


eps = 1e-4


class SpaSoftmax_v6(nn.Module):
    def __init__(self, embedding_net, output_size=10, scale=16, m=0.5, n=1):
        super(SpaSoftmax_v6, self).__init__()
        assert (output_size > 1 and
                scale >= 1.0 and
                m > 0 and
                n >= m)

        self.input_size = sum(embedding_net.output_shape)
        assert(self.input_size > 1)

        self.scale = scale
        self.m = m
        self.n = n
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
        # print('---> emb (before norm):\n', embedding)
        # print('---> emb[j].norm (before norm):\n', embedding.norm(dim=1))
        # print('---> weight (before norm):\n', self.linear.weight)
        # print('---> weight[j].norm (before norm):\n',
        #       self.linear.weight.norm(dim=1))

        # print('---> targets:\n', targets)

        # do L2 normalization
        embedding = F.normalize(embedding, dim=1)
        # print('---> emb (after norm):\n', embedding)
        # print('---> emb[j].norm (after norm):\n', embedding.norm(dim=1))

        weight = F.normalize(self.linear.weight, dim=1)
        # print('---> weight (after norm):\n', weight)
        # print('---> weight[j].norm (after norm):\n',
        #       weight.norm(dim=1))

        cos_theta = F.linear(embedding, weight).clamp(-1, 1)
        # print('---> cos_theta:\n', cos_theta)

        theta = cos_theta.acos() / np.pi
        theta = theta.clamp(0, 1)
        # print('---> theta:\n', theta*180)

        if self.n > 1.0+eps or self.n < 1.0-eps:
            theta_1 = theta.pow(self.n)
        else:
            theta_1 = theta

        # print('---> theta_1:\n', theta_1*180)

        #print(self.m > eps and self.m < self.n-eps)
        if self.m > eps and self.m < self.n-eps:
            one_hot = torch.zeros_like(cos_theta)

            # for i in range(x.shape[0]):
            #     one_hot[i][targets[i]] = self.m*self.scale
            one_hot.scatter_(1, targets.view(-1, 1), 1)

            one_hot_comp = torch.ones_like(cos_theta) - one_hot

            theta_2 = theta.pow(self.m)
            # print('---> theta_2:\n', theta_2*180)

            theta_3 = torch.mul(theta_1, one_hot_comp) + \
                torch.mul(theta_2, one_hot)

            # print('---> biased_theta:\n', theta_3*180)

            output = (1.0 - theta_3) * self.scale

            # print('---> output (s*(1-biased_norm_theta)):\n', output)
            # print(
            #     '---> output[j].norm (s*(1-biased_theta)):\n',
            # output.norm(dim=1))
        else:
            output = (1.0 - theta_1) * self.scale

        return output, cos_theta

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'input_size=' + str(self.input_size) \
            + ', output_size=' + str(self.output_size) \
            + ', scale=' + str(self.scale) \
            + ', m=' + str(self.m) \
            + ', n=' + str(self.n) \
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
    m_n_list = [(0.33, 1), (0.5, 1), (0.67, 1), (0.8, 1), (0.9, 1)]
    # m_n_list = [(1, 1.25), (1, 1.5), (1, 2), (1, 2.5), (1, 3)]
    # m_n_list = [(0.33, 3), (0.5, 2), (0.67, 1.5), (0.8, 1.25), ]
    # m_n_list = [(0.33, 1), (0.5, 1), (0.67, 1), (0.8, 1), (0.9, 1),
    #             (1, 1.25), (1, 1.5), (1, 2), (1, 2.5), (1, 3),
    #             (0.33, 3), (0.5, 2), (0.67, 1.5), (0.8, 1.25), ]

    #dummpy_data = torch.ones(3, emb_size)
    dummpy_data = torch.zeros(3, emb_size)
    # for i, row in enumerate(dummpy_data):
    #     # print('row[{}]: {}'.format(i, row))
    #     for j in range(len(row)):
    #         if j % 2 == 0:
    #             row[j] = 1

    # print('row[{}]: {}'.format(i, row))
    dummpy_targets = torch.tensor([0, 1, 2])

    for (m, n) in m_n_list:
        print('\n#=============================')
        print('\n===> Testing SpaSoftmax_v6 net with dummy data')
        net = IdentityModule(emb_size)
        net = SpaSoftmax_v6(net, output_size, scale, m, n)
        net.set_fc_weights_to_ones()
        print('net:\n', net)

        infer(net, dummpy_data, dummpy_targets)

