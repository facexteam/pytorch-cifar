'''Large margin softmax in PyTorch.
@author: zhaoyafei
'''
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import BasicBlock
import numpy as np


class LargeMarginModule_Cosineface(nn.Module):
    def __init__(self, embedding_net, output_size=10, scale=32, m=0.5, eps=1e-12):
        super(LargeMarginModule_Cosineface, self).__init__()
        self.input_size = sum(embedding_net.output_shape)
        self.eps = eps
        self.scale = scale
        self.m = m

        self.embedding_net = embedding_net
        self.linear = nn.Linear(self.input_size, output_size, bias=False)

    def forward(self, x, targets, device):
        embedding = self.embedding_net(x)

        # print('\n===> In LargeMarginModule_Cosineface.forward()\n')
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

        output_for_predict = F.linear(embedding, weight) * self.scale
        # print('---> output_for_predict (fc_output): ', output_for_predict)
        # print('---> output_for_predict[j].norm (fc_output): ',
        #       output_for_predict.norm(dim=1))

        if self.m > 0:
            one_hot = torch.zeros_like(output_for_predict)

            # for i in range(x.shape[0]):
            #     one_hot[i][targets[i]] = self.m*self.scale
            one_hot.scatter_(1, targets.view(-1, 1), self.m*self.scale)

            # print('---> one_hot after twister: ', one_hot)

            # one_hot.to(device)

            output_for_loss = output_for_predict - one_hot
        else:
            output_for_loss = output_for_predict

        # print('---> output_for_loss (output_for_predict after twister): ', output_for_loss)
        # print(
        #     '---> output_for_loss[j].norm (output_for_predict after twister): ',
        #     output_for_loss.norm(dim=1))

        return output_for_loss, output_for_predict

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'input_size=' + str(self.input_size) \
            + ', output_size=' + str(self.output_size) \
            + ', m=' + str(self.m) + ')'


class LargeMarginModule_Arcface(nn.Module):
    def __init__(self, embedding_net, output_size=10, scale=32, m=0.5):
        super(LargeMarginModule_Arcface, self).__init__()
        self.input_size = sum(embedding_net.output_shape)
        self.output_size = output_size

        self.scale = scale
        self.m = m
        # self.sin_m = np.sin(self.m)
        # self.cos_m = np.cos(self.m)

        self.embedding_net = embedding_net

        self.weight = nn.Parameter(torch.Tensor(
            self.output_size, self.input_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, targets, device):
        # print('===> In LargeMarginModule_Arcface.forward()')
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        embedding = self.embedding_net(x)

        # --------------------------- calculate cos(theta) & theta ---------------------------
        # ebd_norm = torch.norm(embedding, 2, 1)

        normalized_ebd = F.normalize(embedding, dim=1)
        normalized_wt = F.normalize(self.weight, dim=1)

        cos_theta = F.linear(normalized_ebd, normalized_wt)
        # cos_theta = cos_theta.clamp(-1, 1)
        cos_theta.requires_grad_()
        # print('---> cos_theta:', cos_theta)

        theta = cos_theta.data.acos()
        theta.requires_grad_()
        # print('---> theta:', theta)

        # --------------------------- convert targets to one-hot ---------------------------
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, targets.view(-1, 1), self.m)

        # --------------------------- Calculate output ---------------------------
        theta = theta + one_hot
        # print('---> theta add one_hot:', theta)

        output_for_loss = theta.data.cos() * self.scale
        output_for_loss.requires_grad_()

        # output_for_loss *= self.scale
        # print('---> output_for_loss:', output_for_loss)

        output_for_predict = cos_theta * self.scale
        # print('---> output_for_predict:', output_for_predict)

        return output_for_loss, output_for_predict

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'input_size=' + str(self.input_size) \
            + ', output_size=' + str(self.output_size) \
            + ', m=' + str(self.m) + ')'


class LargeMarginModule_ScaledASoftmax(nn.Module):
    def __init__(self, embedding_net, output_size=10, scale=32, m=4, min_lambda=5.0):
        super(LargeMarginModule_ScaledASoftmax, self).__init__()
        self.input_size = sum(embedding_net.output_shape)
        self.output_size = output_size

        self.scale = scale
        self.m = int(m)

        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.min_lambda = min_lambda

        self.embedding_net = embedding_net

        self.iter = 0

        self.weight = nn.Parameter(torch.Tensor(
            self.output_size, self.input_size))
        nn.init.xavier_uniform_(self.weight)

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

    def forward(self, x, targets, device):
        # print('===> In LargeMarginModule_Arcface.forward()')

        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        embedding = self.embedding_net(x)

        self.lamb = max(self.min_lambda, self.base *
                        (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # ebd_norm = torch.norm(embedding, 2, 1)

        normalized_ebd = F.normalize(embedding, dim=1)
        normalized_wt = F.normalize(self.weight, dim=1)

        cos_theta = F.linear(normalized_ebd, normalized_wt)
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        # print('---> cos_theta:', cos_theta)
        # print('---> cos_m_theta:', cos_m_theta)

        theta = cos_theta.data.acos()
        # print('---> theta:', theta)

        k = (self.m * theta / np.pi).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        # print('---> phi_theta:', phi_theta)

        # --------------------------- convert targets to one-hot ---------------------------
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, targets.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output_for_loss = (one_hot * (phi_theta - cos_theta) /
                           (1 + self.lamb)) + cos_theta
        # output_for_loss *= ebd_norm.view(-1, 1)
        output_for_loss *= self.scale
        # print('---> output_for_loss:', output_for_loss)

        output_for_predict = cos_theta*self.scale
        # print('---> output_for_predict:', output_for_predict)

        return output_for_loss, output_for_predict

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'input_size=' + str(self.input_size) \
            + ', output_size=' + str(self.output_size) \
            + ', m=' + str(self.m) + ')'


class LargeMarginModule_ASoftmaxLoss(nn.Module):
    def __init__(self, embedding_net, input_size, output_size=10, m=4, min_lambda=5.0):
        super(LargeMarginModule_ASoftmaxLoss, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # self.scale = scale
        self.m = int(m)

        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.min_lambda = min_lambda

        self.embedding_net = embedding_net

        self.iter = 0

        self.weight = nn.Parameter(torch.Tensor(
            self.output_size, self.input_size))
        nn.init.xavier_uniform_(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, x, targets, device):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        embedding = self.embedding_net(x)

        self.lamb = max(self.min_lambda, self.base *
                        (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(embedding), F.normalize(self.weight))
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
        output = (one_hot * (phi_theta - cos_theta) /
                  (1 + self.lamb)) + cos_theta
        output *= ebd_norm.view(-1, 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'input_size=' + str(self.input_size) \
            + ', output_size=' + str(self.output_size) \
            + ', m=' + str(self.m) + ')'
