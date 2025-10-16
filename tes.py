
import ale_py
import shimmy
import gymnasium as gym
import numpy as np
from numpy import random
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
import math
from numpy.random import gumbel
import timeit

from sympy.physics.units import second

CROSSOVER_RATE = 0.7
DISTRIBUTION_INDEX = 2
BOUNDS_DICT = {"conv.0.weight": [-5e-02, 5e-02], "conv.0.bias": [-5e-02, 5e-02], "conv.2.weight": [-5e-02, 5e-02], "conv.2.bias": [-5e-02, 5e-02], "conv.4.weight": [-5e-02, 5e-02], "conv.4.bias": [-5e-02, 5e-02], "policy.0.weight": [-5e-02, 5e-02], "policy.0.bias": [-5e-01, 5e-01], "policy.2.weight": [-5e-02, 5e-02], "policy.2.bias": [-5e-01, 5e-01]}


def calc_betaq(beta, rand):
    alpha = 2.0 - np.power(beta, -(DISTRIBUTION_INDEX + 1.0))

    mask, mask_not = (rand <= (1.0 / alpha)), (rand > (1.0 / alpha))

    betaq = np.zeros(mask.shape)
    betaq[mask] = np.power((rand * alpha), (1.0 / (DISTRIBUTION_INDEX + 1.0)))[mask]
    betaq[mask_not] = np.power((1.0 / (2.0 - rand * alpha)), (1.0 / (DISTRIBUTION_INDEX + 1.0)))[mask_not]

    return betaq
class Actor(nn.Module):
    # mobilenet colv algorithmmay be faster
    def __init__(self, input_shape, actions):
        super(Actor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[1], 36, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(36, 72, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(72, 72, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.policy = nn.Sequential(
            nn.Linear(self.get_conv_size(input_shape), 512),
            nn.ReLU(),
            nn.Linear(512, actions)
        )

    def get_conv_size(self, input_shape):
        o = self.conv(torch.zeros(*input_shape))
        return int(np.prod(o.size()))



    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.policy(conv_out)


env = gym.make("PongNoFrameskip-v4", render_mode="human")
# for getting state size and state Tensor initialization
state = env.reset()[0]
state = torch.FloatTensor(state)
state = torch.zeros_like(state)
CollectiveState = torch.cat((state, state, state, state), 2)
CollectiveState.unsqueeze_(0)
CollectiveState = CollectiveState.transpose(1, 3)


pongActor = Actor(CollectiveState.shape, env.action_space.n)
secondPongActor = Actor(CollectiveState.shape, env.action_space.n)

a = 0
b = []
c = []
d = []

for name, param in pongActor.named_parameters():
    if param.requires_grad:
        #print(name, param.data)
        a += 1
        b.append(param.data.size())
        c.append(param.data.size().numel())
        d.append(name)

#print(a)
#print(b)
#print(c)
#print(d)

child1 = pongActor.state_dict()
child2 = secondPongActor.state_dict()

print(child1)

import datetime
starttime = datetime.datetime.now()
f = 0
for name in d:
    #flatten so you can index properly for SBX shenanegins
    origsize = child1[name].size()
    Pancaked1 = torch.flatten(child1[name])
    Pancaked2 = torch.flatten(child2[name])
    print(torch.numel(Pancaked1))

    #make sure that this is VERY optimized, we are going to iterate through this BILLIONS of times
    for i in range(torch.numel(Pancaked1)):
        # crossover algo
        if random.rand() < CROSSOVER_RATE:
            temp1 = Pancaked1[i]
            temp2 = Pancaked2[i]

            # y1 is smaller val for computation
            sm = temp1 < temp2
            y1 = np.where(sm, temp1, temp2)
            y2 = np.where(sm, temp2, temp1)



            # stolen from pymoo repo
            # eta is distribution index, hyperparameter LOL



            # difference between all variables
            delta = (y2 - y1)

            # xl and xu were lower and upper bounds of problem (aka tensor value)

            randinit = random.rand()
            xl, xu = BOUNDS_DICT[name]
            beta = 1.0 + (2.0 * (y1 - xl) / delta)
            betaq = calc_betaq(beta, randinit)
            c1 = 0.5 * ((y1 + y2) - betaq * delta)

            beta = 1.0 + (2.0 * (xu - y2) / delta)
            betaq = calc_betaq(beta, randinit)
            c2 = 0.5 * ((y1 + y2) + betaq * delta)


            if random.rand() < 0.5:
                Pancaked1[i] = torch.from_numpy(np.where(sm, c1, c2)).float()
                Pancaked2[i] = torch.from_numpy(np.where(sm, c2, c1)).float()
            else:
                Pancaked2[i] = torch.from_numpy(np.where(sm, c1, c2)).float()
                Pancaked1[i] = torch.from_numpy(np.where(sm, c2, c1)).float()

    # after Pancaked done
    child1[name] = Pancaked1
    child2[name] = Pancaked2
    print("phase" + name + "done")









print(child1)
print(f)
print(sum(c))
endtime = datetime.datetime.now()
diff = endtime - starttime
print('Job took: ', diff.days, diff.seconds, diff.microseconds)
#for i in range(12000000):

# code outline:
# copy state_dict to sb
# loop through names of state_dict
# perform SBX on each section of state_dict with probabilities and allat
# perform SBX TWICE for 2 OFFSPRING
# assign each section of modified state_dict to sb
# model.load_state_dict(sb)

