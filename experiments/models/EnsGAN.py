from __future__ import print_function
import torch.nn as nn
import torch.optim as optim

#see https://raw.githubusercontent.com/caogang/wgan-gp/master/gan_mnist.py
DIM = 64
OUTPUT_DIM = 784
class C(nn.Module):
    def __init__(self):
        super(C, self).__init__()

        main = nn.Sequential(
            nn.Conv2d(1, DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(DIM, 2*DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(2*DIM, 4*DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
        )
        self.main = main
        self.output = nn.Linear(4*4*4*DIM, 1)

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4*4*4*DIM)
        out = self.output(out)
        return out.view(-1)

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()

        preprocess = nn.Sequential(
            nn.Linear(128, 4*4*4*DIM),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4*DIM, 2*DIM, 5),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2*DIM, DIM, 5),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 1, 8, stride=2)

        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4*DIM, 4, 4)
        output = self.block1(output)
        output = output[:, :, :7, :7]
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        return output
        #return output.view(-1, OUTPUT_DIM)

class EnsGAN:
    def __init__(self,n=4):
        self.gs = []
        self.gopts = []
        for i in xrange(n):
            self.gs.append(G()) 
            self.gopts.append( optim.Adam(self.gs[i].parameters(), lr=1e-4, betas=(0.5,0.9)) )
        self.c = C() 
        self.copt = optim.Adam(self.c.parameters(), lr=1e-4, betas=(0.5, 0.9))

