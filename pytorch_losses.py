import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable


# L1Loss AKA absolute loss
loss = nn.L1Loss()
input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
target = autograd.Variable(torch.randn(3, 5))
output = loss(input, target)
output.backward()

# MSE Loss AKA AKA AKA
loss = nn.MSELoss()
input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
target = autograd.Variable(torch.randn(3, 5))
output = loss(input, target)
output.backward()

# CrossEntropyLoss
loss = nn.CrossEntropyLoss()
input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
target = autograd.Variable(torch.LongTensor(3).random_(5))
output = loss(input, target)
output.backward()

# NLLLoss
m = nn.LogSoftmax()
loss = nn.NLLLoss()
# input is of size N x C = 3 x 5
input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
# each element in target has to have 0 <= value < C
target = autograd.Variable(torch.LongTensor([1, 0, 4]))
output = loss(m(input), target)
output.backward()

# PoissonNLLLoss # Negative log likelihood loss with Poisson distribution of target.
loss = nn.PoissonNLLLoss()
log_input = autograd.Variable(torch.randn(5, 2), requires_grad=True)
target = autograd.Variable(torch.randn(5, 2))
output = loss(log_input, target)
output.backward()

# NLLLoss2d # negative log likehood loss, but for image inputs. It computes NLL loss per-pixel.
m = nn.Conv2d(16, 32, (3, 3)).float()
loss = nn.NLLLoss2d()
# input is of size N x C x height x width
input = autograd.Variable(torch.randn(3, 16, 10, 10))
# each element in target has to have 0 <= value < C
target = autograd.Variable(torch.LongTensor(3, 8, 8).random_(0, 4))
output = loss(m(input), target)
output.backward()

# KLDivLoss # The Kullback-Leibler divergence Loss

# BCELoss # Binary Cross Entropy
m = nn.Sigmoid()
loss = nn.BCELoss()
input = autograd.Variable(torch.randn(3), requires_grad=True)
target = autograd.Variable(torch.FloatTensor(3).random_(2))
output = loss(m(input), target)
output.backward()

# BCEWithLogitsLoss # This loss combines a Sigmoid layer and the BCELoss in one single class

# MarginRankingLoss # Creates a criterion that measures the loss given inputs x1, x2, two 1D mini-batch Tensor`s, and a label 1D mini-batch tensor `y with values (1 or -1).

# HingeEmbeddingLoss
'''                 { x_i,                  if y_i ==  1
loss(x, y) = 1/n {
                    { max(0, margin - x_i), if y_i == -1'''
                    
#MultiLabelMarginLoss # multi-class multi-classification hinge loss (margin-based loss) 
# loss(x, y) = sum_ij(max(0, 1 - (x[y[j]] - x[i]))) / x.size(0)

# SmoothL1Loss AKA Huber loss # Creates a criterion that uses a squared term if the absolute element-wise error falls below 1 and an L1 term otherwise.

# SoftMarginLoss # two-class classification logistic loss

# MultiLabelSoftMarginLoss # multi-label one-versus-all loss based on max-entropy

# CosineEmbeddingLoss

# MultiMarginLoss

# TripletMarginLoss
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
input1 = autograd.Variable(torch.randn(100, 128))
input2 = autograd.Variable(torch.randn(100, 128))
input3 = autograd.Variable(torch.randn(100, 128))
output = triplet_loss(input1, input2, input3)
output.backward()