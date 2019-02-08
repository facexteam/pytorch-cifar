'''Large margin softmax in PyTorch.
@author: zhaoyafei
'''
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import BasicBlock
import numpy as np


class LargeMarginModule_cosface(nn.Module):
    def __init__(self, embedding_net, output_size=10, scale=4, m=0.5):
        super(LargeMarginModule_cosface, self).__init__()
        assert (output_size > 1 and
                scale >= 1 and
                m >= 0)

        self.input_size = sum(embedding_net.output_shape)
        assert(self.input_size > 1)

        self.scale = scale
        self.m = m

        self.embedding_net = embedding_net
        self.linear = nn.Linear(self.input_size, output_size, bias=False)

    def get_fc_weights(self):
        wt = self.linear.weight.clone().detach()
        return wt

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

        if self.m > 0:
            one_hot = torch.zeros_like(cos_theta)

            # for i in range(x.shape[0]):
            #     one_hot[i][targets[i]] = 1
            one_hot.scatter_(1, targets.view(-1, 1), 1)

            # print('---> one_hot after twister: ', one_hot)

            # one_hot.to(device)

            output = (cos_theta - one_hot * self.m) * self.scale
        else:
            output = cos_theta * self.scale

        # print('---> output (cos_theta after twister): ', output)
        # print(
        #     '---> output[j].norm (cos_theta after twister): ',
        #     output.norm(dim=1))

        return output, cos_theta

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'input_size=' + str(self.input_size) \
            + ', output_size=' + str(self.output_size) \
            + ', scale=' + str(self.scale) \
            + ', m=' + str(self.m) \
            + ')'


class LargeMarginModule_arcface(nn.Module):
    def __init__(self, embedding_net, output_size=10, scale=4, m=0.2):
        super(LargeMarginModule_arcface, self).__init__()
        assert (output_size > 1 and
                scale >= 1 and
                m >= 0)

        self.input_size = sum(embedding_net.output_shape)
        assert(self.input_size > 1)

        self.output_size = output_size

        self.scale = scale
        self.m = m
        # self.sin_m = np.sin(self.m)
        # self.cos_m = np.cos(self.m)

        self.embedding_net = embedding_net

        # self.weight = nn.Parameter(torch.Tensor(
        #     self.output_size, self.input_size))
        # nn.init.xavier_uniform_(self.weight)

        self.linear = nn.Linear(self.input_size, output_size, bias=False)
        # self.weight = self.linear.weight

    def get_fc_weights(self):
        wt = self.linear.weight.clone().detach()
        return wt

    def forward(self, x, targets):
        # print('===> In LargeMarginModule_arcface.forward()')
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        embedding = self.embedding_net(x)

        # --------------------------- calculate cos(theta) & theta ---------------------------
        # # ebd_norm = torch.norm(embedding, 2, 1)

        # normalized_ebd = F.normalize(embedding, dim=1)
        # normalized_wt = F.normalize(self.linear.weight, dim=1)

        # cos_theta = F.linear(normalized_ebd, normalized_wt)
        # cos_theta = cos_theta.clamp(-1, 1)

        # print('---> cos_theta:', cos_theta)
        # print('---> cos_theta.requires_grad:', cos_theta.requires_grad)
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

        if self.m > 0:
            cos_theta = cos_theta.clamp(-1, 1)

            # theta = cos_theta.data.acos()  # no grad here
            theta = cos_theta.acos()  # theta requires grad here
            # theta.requires_grad_()
            # print('---> theta:', theta)
            # print('---> theta.requires_grad:', theta.requires_grad)

            # --------------------------- convert targets to one-hot ---------------------------
            one_hot = torch.zeros_like(cos_theta)
            one_hot.scatter_(1, targets.view(-1, 1), 1)

            # --------------------------- Calculate output ---------------------------
            new_theta = theta + one_hot * self.m
            # print('---> theta add one_hot:', new_theta)
            # print('---> new_theta:', new_theta)
            # print('---> new_theta.requires_grad:', new_theta.requires_grad)

            output = new_theta.cos() * self.scale
        else:
            output = cos_theta * self.scale

        # print('---> output:', output)
        # print('---> output.requires_grad:',
        #       output.requires_grad)

        return output, cos_theta

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'input_size=' + str(self.input_size) \
            + ', output_size=' + str(self.output_size) \
            + ', scale=' + str(self.scale) \
            + ', m=' + str(self.m) \
            + ')'


class LargeMarginModule_ScaledASoftmax(nn.Module):
    def __init__(self, embedding_net, output_size=10, scale=4, m=3, min_lambda=.0):
        super(LargeMarginModule_ScaledASoftmax, self).__init__()
        assert (output_size > 1 and
                scale >= 1 and
                m >= 0 and
                min_lambda >= 0)

        self.input_size = sum(embedding_net.output_shape)
        assert(self.input_size > 1)

        self.output_size = output_size

        self.scale = scale
        self.m = int(m)

        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.min_lambda = min_lambda

        self.embedding_net = embedding_net

        self.iter = 0

        # self.weight = nn.Parameter(torch.Tensor(
        #     self.output_size, self.input_size))
        # nn.init.xavier_uniform_(self.weight)
        self.linear = nn.Linear(self.input_size, output_size, bias=False)

        # duplication formula
        # refer to:  Double-angle, triple-angle, and half-angle formulae
        # in https://en.wikipedia.org/wiki/List_of_trigonometric_identities
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def get_fc_weights(self):
        wt = self.linear.weight.clone().detach()
        return wt

    def forward(self, x, targets):
        # print('===> In LargeMarginModule_arcface.forward()')

        embedding = self.embedding_net(x)

        _lambda = 0
        if self.min_lambda > 0:
            # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
            self.iter += 1
            _lambda = max(self.min_lambda, self.base *
                          (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # ebd_norm = torch.norm(embedding, 2, 1)

        normalized_ebd = F.normalize(embedding, dim=1)
        normalized_wt = F.normalize(self.linear.weight, dim=1)

        cos_theta = F.linear(normalized_ebd, normalized_wt)
        cos_theta = cos_theta.clamp(-1, 1)

        cos_m_theta = self.mlambda[self.m](cos_theta)
        # print('---> cos_theta:', cos_theta)
        # print('---> cos_theta.requires_grad:', cos_theta.requires_grad)
        # print('---> cos_m_theta:', cos_m_theta)
        # print('---> cos_m_theta.requires_grad:', cos_m_theta.requires_grad)

        theta = cos_theta.data.acos()  # theta does not require grad() here
        # print('---> theta:', theta)
        # print('---> theta.requires_grad:', theta.requires_grad)

        k = (self.m * theta / np.pi).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        # print('---> phi_theta:', phi_theta)

        # --------------------------- convert targets to one-hot ---------------------------
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, targets.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        if _lambda > 0:
            output = one_hot * (phi_theta - cos_theta) * \
                (1.0 / (1 + _lambda)) + cos_theta
        else:
            output = one_hot * (phi_theta - cos_theta) + cos_theta

        # output *= ebd_norm.view(-1, 1)
        output *= self.scale
        # print('---> output:', output)

        return output, cos_theta

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'input_size=' + str(self.input_size) \
            + ', output_size=' + str(self.output_size) \
            + ', scale=' + str(self.scale) \
            + ', m=' + str(self.m) \
            + ', min_lambda=' + str(self.min_lambda) \
            + ')'


class LargeMarginModule_ASoftmaxLoss(nn.Module):
    def __init__(self, embedding_net, input_size, output_size=10, m=4, min_lambda=5.0):
        super(LargeMarginModule_ASoftmaxLoss, self).__init__()
        assert (output_size > 1 and
                input_size > 1 and
                m >= 0 and
                min_lambda >= 0)

        self.input_size = input_size
        self.output_size = output_size

        # self.scale = scale
        self.m = int(m)

        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.min_lambda = min_lambda

        assert (self.scale >= 1 and
                self.m >= 0 and
                self.min_lambda >= 0)

        self.embedding_net = embedding_net

        self.iter = 0

        # self.weight = nn.Parameter(torch.Tensor(
        #     self.output_size, self.input_size))
        # nn.init.xavier_uniform_(self.weight)
        self.linear = nn.Linear(self.input_size, output_size, bias=False)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def get_fc_weights(self):
        wt = self.linear.weight.clone().detach()
        return wt

    def forward(self, x, targets):
        embedding = self.embedding_net(x)

        _lambda = 0
        if self.min_lambda > 0:
            self.iter += 1
            # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
            _lambda = max(self.min_lambda, self.base *
                          (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        normalized_ebd = F.normalize(embedding, dim=1)
        normalized_wt = F.normalize(self.linear.weight, dim=1)

        # cos_theta = F.linear(F.normalize(embedding),
        #                      F.normalize(self.linear.weight))
        # cos_theta = cos_theta.clamp(-1, 1)
        cos_theta = F.linear(normalized_ebd, normalized_wt)
        cos_theta = cos_theta.clamp(-1, 1)

        cos_m_theta = self.mlambda[self.m](cos_theta)

        theta = cos_theta.data.acos()
        k = (self.m * theta / np.pi).floor()
        phi_theta = ((-1.0)**k) * cos_m_theta - 2 * k

        ebd_norm = torch.norm(embedding, 2, 1)

        # --------------------------- convert targets to one-hot ---------------------------
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, targets.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        if _lambda > 0:
            output = one_hot * (phi_theta - cos_theta) * \
                (1.0 / (1 + _lambda)) + cos_theta
        else:
            output = one_hot * (phi_theta - cos_theta) + cos_theta

        output *= ebd_norm.view(-1, 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'input_size=' + str(self.input_size) \
            + ', output_size=' + str(self.output_size) \
            + ', m=' + str(self.m) \
            + ', min_lambda=' + str(self.min_lambda) \
            + ')'
