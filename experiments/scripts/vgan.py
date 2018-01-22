from __future__ import print_function
import torch
import data.mnist as mnist
from utils.helper import *
import numpy as np
import os

#see https://raw.githubusercontent.com/caogang/wgan-gp/master/gan_mnist.py
use_cuda = False
DIM = 64
OUTPUT_DIM = 784
LAMBDA = 10
train_loader = mnist.get_mnist_train_loader()
e = EnsGAN()
one = torch.FloatTensor([1])
mone = one * -1
zdim = 128
bs = 64 #batch size
n_epochs = 2
result_path = 'results/vgan/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
for i in xrange(n_epochs):
    for batch_idx, (x, y) in enumerate(train_loader):
        xv = Variable(x)
        z = torch.FloatTensor(bs, zdim).normal_(0,1)
        zv = Variable(z)
        eps = torch.FloatTensor(bs, 1,1,1).uniform_()
        ls = []
        for g in e.gs:
            e.c.zero_grad()
            g.zero_grad()
            gen = Variable(g(zv).data)
            loss = e.c(gen).mean() - e.c(xv).mean() + calc_gradient_penalty(e.c, xv.data, gen.data, eps)
            loss.backward()
            e.copt.step()
            ls.append(loss.data.numpy()) 
        best_idx = np.argmin(ls) 
        e.c.zero_grad()
        e.gs[best_idx].zero_grad()
        save_result(e.gs[best_idx](zv), result_path+'iter_'+str(i)+'g_'+str(best_idx))
        loss = -e.c(e.gs[best_idx](zv)).mean()
        loss.backward()
        e.gopts[best_idx].step()
