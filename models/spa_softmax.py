'''Large margin softmax in PyTorch.
@author: zhaoyafei
'''
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


eps = 1e-4


class SpaSoftmax(nn.Module):
    def __init__(self, embedding_net, output_size=10, scale=4, m=2, b=0):
        super(SpaSoftmax, self).__init__()
        assert (output_size > 1 and
                scale >= 1 and
                m >= 1.0 and
                b >= 0)

        self.input_size = sum(embedding_net.output_shape)
        assert(self.input_size > 1)

        self.scale = scale
        self.m = m
        self.b = b
        self.output_size = output_size

        self.embedding_net = embedding_net
        self.linear = nn.Linear(self.input_size, output_size, bias=False)

    def get_fc_weights(self):
        wt = self.linear.weight.clone().detach()
        return wt

    def set_fc_weights(self, wt):
        self.linear.weight.data.copy_(wt)

    def set_fc_weights_to_ones(self):
        self.linear.weight.data.fill_(1.0)

    def forward(self, x, targets):
        embedding = self.embedding_net(x)

        # print('---> emb (before norm): ', embedding)
        # print('---> emb[j].norm (before norm): ', embedding.norm(dim=1))
        # print('---> weight (before norm): ', self.linear.weight)
        # print('---> weight[j].norm (before norm): ',
        #       self.linear.weight.norm(dim=1))

        # print('---> targets:\n', targets)

        # do L2 normalization
        embedding = F.normalize(embedding, dim=1)
        # print('---> emb (after norm): ', embedding)
        # print('---> emb[j].norm (after norm): ', embedding.norm(dim=1))

        weight = F.normalize(self.linear.weight, dim=1)
        # print('---> weight (after norm): ', weight)
        # print('---> weight[j].norm (after norm): ',
        #       weight.norm(dim=1))

        cos_theta = F.linear(embedding, weight).clamp(-1, 1)
        # print('---> cos_theta:\n', cos_theta)

        if self.m > 1.0:
            one_hot = torch.zeros_like(cos_theta)
            one_hot.scatter_(1, targets.view(-1, 1), 1)
            # print('---> one_hot: ', one_hot)

            # bias = torch.mul(cos_theta, one_hot) * \
            #     self.m - one_hot * (self.m - 1) - \
            #     torch.mul(cos_theta, one_hot)
            # bias = torch.mul(cos_theta, one_hot) * \
            #     (self.m - 1) - one_hot * (self.m - 1)
            # bias = torch.mul(cos_theta - 1, one_hot) * \
            #     (self.m - 1)

            biased_cos_theta = cos_theta + torch.mul(cos_theta - 1, one_hot) * \
                (self.m - 1)

            if self.b > 0:
                biased_cos_theta = biased_cos_theta - one_hot * self.b

            # print('---> biased_cos_theta:\n', biased_cos_theta)
            # one_hot.to(device)

            output = biased_cos_theta * self.scale
        else:
            output = cos_theta * self.scale

        # print('---> output (s*biased_cos_theta):\n', output)
        # print(
        #     '---> output[j].norm (s*biased_cos_theta):\n',
        #     output.norm(dim=1))

        return output, cos_theta

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'input_size=' + str(self.input_size) \
            + ', output_size=' + str(self.output_size) \
            + ', scale=' + str(self.scale) \
            + ', m=' + str(self.m) \
            + ', b=' + str(self.b) \
            + ')'


class SpaSoftmax_v1_ext(nn.Module):
    def __init__(self, embedding_net, output_size=10, scale=4, m=2, n=1, b=0):
        super(SpaSoftmax_v1_ext, self).__init__()
        assert (output_size > 1 and
                scale >= 1 and
                m >= n and n > 0 and
                b >= 0)
        self.input_size = sum(embedding_net.output_shape)
        assert(self.input_size > 1)

        self.scale = scale
        self.m = m
        self.n = n
        self.b = b
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
        # print('---> emb (before norm): ', embedding)
        # print('---> emb[j].norm (before norm): ', embedding.norm(dim=1))
        # print('---> weight (before norm): ', self.linear.weight)
        # print('---> weight[j].norm (before norm): ',
        #       self.linear.weight.norm(dim=1))

        # print('---> targets:\n', targets)

        # do L2 normalization
        embedding = F.normalize(embedding, dim=1)
        # print('---> emb (after norm): ', embedding)
        # print('---> emb[j].norm (after norm): ', embedding.norm(dim=1))

        weight = F.normalize(self.linear.weight, dim=1)
        # print('---> weight (after norm): ', weight)
        # print('---> weight[j].norm (after norm): ',
        #       weight.norm(dim=1))

        cos_theta = F.linear(embedding, weight).clamp(-1, 1)
        # print('---> cos_theta:\n', cos_theta)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, targets.view(-1, 1), 1)
        # print('---> one_hot: ', one_hot)

        if (self.m - self.n) > 0:
            cos_theta_1 = cos_theta - 1
            # print('---> cos_theta_1:\n', cos_theta_1)

            # biased_cos_theta = cos_theta_1 * self.n + torch.mul(cos_theta_1, one_hot) * \
            #     (self.m - self.n) + (1 + self.b)
            biased_cos_theta = cos_theta_1 * self.n + torch.mul(cos_theta_1, one_hot) * \
                (self.m - self.n) + 1
        else:
            biased_cos_theta = cos_theta

        if self.b > 0:
            biased_cos_theta = biased_cos_theta - one_hot * self.b

        # print('---> biased_cos_theta:\n', biased_cos_theta)

        output = biased_cos_theta * self.scale

        # print('---> output (s*biased_cos_theta):\n', output)
        # print(
        #     '---> output[j].norm (s*biased_cos_theta): ',
        #     output.norm(dim=1))

        return output, cos_theta

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'input_size=' + str(self.input_size) \
            + ', output_size=' + str(self.output_size) \
            + ', scale=' + str(self.scale) \
            + ', m=' + str(self.m) \
            + ', n=' + str(self.n) \
            + ', b=' + str(self.b) \
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
        print('===> output of net(data): ')
        print('pred (s*biased_cos_theta): \n', pred)
        print('cos_theta: \n', cos_theta)

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
    dummpy_targets = torch.tensor([0, 1, 2])

    print('\n#=============================')
    print('\n===> Testing SpaSoftmax net with dummy data')
    net = IdentityModule(emb_size)
    net = SpaSoftmax(net, output_size, scale)
    net.set_fc_weights_to_ones()
    print('net: ', net)

    infer(net, dummpy_data, dummpy_targets)

    print('\n#=============================')
    print('\n===> Testing SpaSoftmax_v2_ext net with dummy data')
    net = IdentityModule(emb_size)
    net = SpaSoftmax_v1_ext(net, output_size, scale)
    net.set_fc_weights_to_ones()
    print('net: ', net)

    infer(net, dummpy_data, dummpy_targets)

