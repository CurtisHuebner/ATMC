import numpy as np
import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
import torch.nn.functional as f
import atmc

import copy


N_SAMPLES = 2000
D_SAMPLES = 20
BATCH_SIZE = 200

class MLP(nn.Module):
    def __init__(self, d_in, h, d_out):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(d_in, h)
        self.l2 = nn.Linear(h, h)
        self.l3 = nn.Linear(h, d_out)

    def forward(self, x):
        h1 = f.elu(self.l1(x))
        h2 = f.elu(self.l2(h1))
        y_pred = self.l3(h2)
        return y_pred

def main():
    x = np.random.normal(size=(N_SAMPLES,D_SAMPLES))
    y = np.random.normal(size=(N_SAMPLES,))

    mlp1 = MLP(D_SAMPLES,60,1)
    mlp2 = copy.deepcopy(mlp1)


    opt1 = t.optim.Adam(mlp1.parameters())
    opt2 = atmc.ATMC(mlp2.parameters())

    def train(opt,mlp):
        losses = []
        for i in range(20000):
            index = np.random.choice(range(N_SAMPLES),size=(BATCH_SIZE,),replace=False)
            b_x = t.FloatTensor(x[index])
            b_y = t.FloatTensor(y[index])

            mlp.zero_grad()
            log_p = t.sum((b_y-t.squeeze(mlp(b_x)))**2)/2
            prior = sum([t.sum(x**2) for x in mlp.parameters()])
            loss = (N_SAMPLES/BATCH_SIZE)*log_p+prior
            loss.backward()
            opt.step()

            if i % 20 == 0:
                print(i)
                with t.no_grad():
                    b_x = t.FloatTensor(x)
                    b_y = t.FloatTensor(y)
                    loss = (t.sum((b_y-t.squeeze(mlp(b_x)))**2)/2)
                    losses.append(loss.cpu().numpy())

        return losses

    l1 = train(opt1,mlp1)
    l2 = train(opt2,mlp2)

    l3 = [np.sum(y**2)/2]*len(l1)

    plt.plot(l1,label="Adam")
    plt.plot(l2,label="ATMC")
    plt.plot(l3,label="Baseline")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
