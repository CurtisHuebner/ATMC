#!/usr/bin/env python3
import torch as t
import torch.nn.functional as f
import torch.optim as opt

class ATMC(opt.Optimizer):
    def __init__(self, params,h=0.001,d=1,m=1):
        defaults = dict(h=h, m=m,d=d)
        super(ATMC, self).__init__(params, defaults)

    def step(self,closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            h = group['h']
            m = group['m']
            d = group['d']

            for param in group['params']:
                if param.grad is None:
                    continue
                g = param.grad
                n = t.randn_like(g)

                param_state = self.state[param]

                if 'momentum' not in param_state:
                        param_state['momentum'] = t.zeros_like(g).detach()
                p =  param_state['momentum']
                if 'regulator' not in param_state:
                        param_state['regulator'] = t.zeros_like(g).detach()
                xi = param_state['regulator']

                alpha = f.relu(d-xi)
                beta = (alpha + xi)

                #TODO:Replace with exact integration of the underlying DE
                term = -g*h-beta*p*h+t.sqrt(2*h*alpha*m)*n

                param.data += h*(p/m)
                param_state['regulator'] += h*(p**2/m-1)
                param_state['momentum'] += term

        return loss

def test():
    x = t.zeros((1,))
    x.requires_grad = True

    def f(x):
        loss = t.sum(x**2/2)
        loss.backward()
        return loss

    samps = []
    sampler = ATMC([x],h=0.01)
    for i in range(80000):
        sampler.zero_grad()
        sampler.step(lambda: f(x))
        samps.append(x.detach().numpy()[0])

    plt.hist(samps,bins=50,density=True, alpha=0.5,label="samps")
    plt.hist(np.random.normal(size=20000), bins=50, density=True,
    alpha=0.5,label="normal")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    test()





