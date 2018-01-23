from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.autograd as autograd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import itertools

use_cuda = False
def calc_gradient_penalty(C, x, gen, eps, LAMBDA):
    xhat = eps*x + (1-eps)*gen
    xhat = Variable(xhat, requires_grad=True)
    d_int = C(xhat)
    gradients = autograd.grad(outputs=d_int, inputs=xhat,
                              grad_outputs=torch.ones(d_int.size()).cuda(gpu) if use_cuda else torch.ones(
                                  d_int.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def save_result(images, path = 'result.png'):
    size_figure_grid = 10
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(8*8):
        i = k // 8
        j = k % 8
        ax[i, j].cla()
        ax[i, j].imshow(images[k].cpu().data.view(64, 64).numpy(), cmap='gray')

    #fig.text(0.5, 0.04, ha='center')
    plt.savefig(path)
