import math
import ale_py
import shimmy
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
from numpy.random import gumbel

from torch.distributions import Categorical

import sys
# hyperparameters
LEARNING_RATE = 0.01
ENTROPY_BETA = 0.01
FRAMESKIP = 4
EPISODES = 10000

REWARD_STEPS = 4
GAMMA = 0.99


#PATHA = "betterpongactor.pth"
#PATHC = "betterpongcritic.pth"


def queueTensor(tensor, x):
    return torch.cat((tensor[:, 3:, :, :], x), dim=1)
    # dim=2 means concatenating along rgb axis, imagine an overlap of FRAMESKIP amt of frames
def sample_gumbel(shape, eps=1e-20):
  """Sample from Gumbel(0, 1)"""
  U = torch.rand(shape)
  return torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature=0.5):
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(logits.shape[0])
  return F.softmax( y / temperature)

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


class Critic(nn.Module):
    def __init__(self, input_shape):
        super(Critic, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[1], 36, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(36, 72, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(72, 72, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.value = nn.Sequential(
            nn.Linear(self.get_conv_size(input_shape), 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def get_conv_size(self, input_shape):
        o = self.conv(torch.zeros(*input_shape))
        return int(np.prod(o.size()))



    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.value(conv_out)




#env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("PongNoFrameskip-v4", render_mode = "human")

# for getting state size and state Tensor initialization
state = env.reset()[0]
state = torch.FloatTensor(state)
state = torch.zeros_like(state)
CollectiveState = torch.cat((state, state, state, state), 2)
CollectiveState.unsqueeze_(0)
CollectiveState = CollectiveState.transpose(1, 3)


pongActor = Actor(CollectiveState.shape, env.action_space.n)

#pongCritic = Critic(shared = pongActor.net) #shared layers might be wrong
pongCritic = Critic(CollectiveState.shape)


# optimizers
Actoroptimizer = optim.SGD(pongActor.parameters(), lr=LEARNING_RATE, momentum=0.9)
Criticoptimizer = optim.SGD(pongCritic.parameters(), lr=LEARNING_RATE, momentum=0.9)


done = False
episode_reward = 0
log_prob = 0
pongActor.train()
pongCritic.train()
term = False
for i in range(EPISODES):
    state = env.reset()[0] # fix?
    env.render()
    episode_reward = 0
    # main episode loop



    # memory buffer thing
    done = False
    counter = 0
    actions_logprob = []
    rewards = []
    CollectiveState = torch.zeros_like(CollectiveState)
    advantages = []
    values = []
    Qvals = []
    entropies = []

    # do first 4 frames here to not take into account reward before agent could do anything, then do normal cycle of 4
    #env.reset() but for 4 frames for acceleration and stuff
    for j in range(FRAMESKIP): # first frame already done at env.reset
        state, _, _, _, _ = env.step(0) # 0 = noop
        env.render()
        state = torch.FloatTensor(state)
        state.unsqueeze_(0)
        state = state.transpose(1,3)
        CollectiveState = queueTensor(CollectiveState, state)



    while not done:
        action_probs = pongActor(CollectiveState)

        #gumbel things
        action_distribution = gumbel_softmax_sample(action_probs)
        # multimodal does not have grad_fn
        chosen_action = action_distribution.multinomial(num_samples=1, replacement=True)
        value = pongCritic(CollectiveState)



        # entropy?
        # save values, both should have REWARD_STEPS variables by the end
        actions_logprob.append(torch.log(action_distribution[0][chosen_action.item()]))
        values.append(value)

        print(action_probs)
        print(chosen_action)
        print(value)
        # get ready for next step
        counter += 1
        # massive issue with counting logic for some reason
        # frameskipping part
        for k in range(FRAMESKIP):
            #states.append(state)
            state, reward, done, _, _ = env.step(chosen_action)
            state = torch.FloatTensor(state)
            state.unsqueeze_(0)
            state = state.transpose(1, 3)
            CollectiveState = queueTensor(CollectiveState, state)

            rewards.append(reward) # should have FRAMESKIP * REWARD_STEPS elements by the end
            episode_reward += reward
            # terminate and process learnings
            if done:
                break

        # frameskip over, think about next action REWARDSTEPS amt of times

        # assessment period
        if counter == REWARD_STEPS or done:
            if done:
                term = True
            n = len(rewards)
            # reset vars for loss
            Qval = pongCritic(CollectiveState)
            Qval = Qval.detach().numpy()[0][0]
            Qvals = np.zeros(math.ceil(n / FRAMESKIP))
            for t in reversed(range(n)):
                Qval = rewards[t] + GAMMA * Qval # going from last to first, but why he did it like this I have no fucking clue
                # removing this if causes Qval and Value to explode for some reason
                if t % FRAMESKIP == 0:
                    Qvals[t // FRAMESKIP] = Qval
                elif term:
                    Qval = 0
                    term = False

            # loss processing
            values = torch.cat(values)
            values = torch.flatten(values)
            Qvals = torch.FloatTensor(Qvals)
            Qvals = torch.flatten(Qvals)
            advantages = Qvals - values
            advantages = torch.FloatTensor(advantages)

            #entropies = torch.stack(entropies)


            actions_logprob = torch.stack(actions_logprob)
            actor_loss = (-actions_logprob * advantages.detach()).mean() #.detach optional?

            advantages = advantages.pow(2).mean()
            critic_loss = 0.5 * advantages

            #entropy_loss = -ENTROPY_BETA * entropies.mean()

            #critic_loss = entropy_loss + critic_loss

            print("done phase")
            print(Qvals)
            print(values)
            print(actor_loss)
            print(critic_loss)
            print(rewards)
            print("done done")
            Actoroptimizer.zero_grad()
            actor_loss.backward(retain_graph=True)


            Criticoptimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(pongActor.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(pongCritic.parameters(), max_norm=0.5)


            #print(critic_loss)
            #print(critic_loss.grad)
            Actoroptimizer.step()
            Criticoptimizer.step()


            #print(Qvals)
            #print(values)

            # memory batching goes here

            # resetting variables after updating model



            counter = 0
            actions_logprob = []
            rewards = []
            advantages = []
            values = []
            Qvals = []
            entropies = []


    print("episode: " + str(i))
    print("reward: " + str(episode_reward))
    print("Actor loss: " + str(actor_loss))
    print("Critic loss: " + str(critic_loss))





