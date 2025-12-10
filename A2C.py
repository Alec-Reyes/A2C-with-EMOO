from argparse import Action
from collections import deque


import numpy as np
from numpy import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import socket
import struct
import copy


import sys


import math

daindexfordicts = 0


# hyperparameters
LEARNING_RATE = 1e-4
ENTROPY_BETA = 0.001


FRAMESTACK = 4 #has to be the same as FRAMESKIP in client
REWARD_STEPS = 10
GAMMA = 0.99
ACTION_SPACE = 180 # 20 * 9 guidelines for buttons

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)


PATHA = "pongactorPONG.pth"
PATHC = "pongcriticPONG.pth"


gameScores = []


CROSSOVER_RATE = 0.7
DISTRIBUTION_INDEX = 2
POPULATION_SIZE = 20
Q_TRAINING_EPISODES = 10
# bounds arbitrarily placed based on observing a successful model (premature stop after a high rally)
BOUNDS_DICT = {"net.0.weight": [-3e-01, 3e-01], "net.0.bias": [-5e-02, 5e-02], "net.2.weight": [-5e-02, 5e-02], "net.2.bias": [-5e-01, 5e-01], "actor.0.weight": [-5e-01, 5e-01], "actor.0.bias": [-5e-02, 5e-02], "critic.0.weight": [-5e-01, 5e-01], "critic.0.bias": [-5e-02, 5e-02]}


"""
TODO:
- test only DNN
- tweak reward system maybe idk have fun working that's for sure
Log of hyperparameters that worked (AI won a game after a given time:
- Reward Steps: 4
- Frame stacks: 4, Frames Ignored: 2 (include every other frame)
- LEARNING_RATE = 1e-4
- ENTROPY_BETA = 0.001
- REWARD_STEPS = 4
- GAMMA = 0.99
- Reward for going towards ball: 0.005 / -0.005


- erasing reward for going towards ball still converges, with premature stop at episode 267 w 5 models running concurrently (so functionally ep 50-smth)
"""
def calc_betaq(beta, rand):
   alpha = 2.0 - np.power(beta, -(DISTRIBUTION_INDEX + 1.0))


   mask, mask_not = (rand <= (1.0 / alpha)), (rand > (1.0 / alpha))


   betaq = np.zeros(mask.shape)
   betaq[mask] = np.power((rand * alpha), (1.0 / (DISTRIBUTION_INDEX + 1.0)))[mask]
   betaq[mask_not] = np.power((1.0 / (2.0 - rand * alpha)), (1.0 / (DISTRIBUTION_INDEX + 1.0)))[mask_not]


   return betaq


def dominate_max(p1, p2): # checking if p2 dominated by p1
   assert len(p1) == len(p2), (len(p1), len(p2))
   dominated = False
   for i, j in zip(p1, p2):
       if j > i: # if p2 is better than p1 in at least 1 aspect
           return False
       if i > j:
           dominated = True
   # only triggers if p1 is better than p2 in both aspects, meaning p2 is dominated by p1
   return dominated


def SBX(parent1, parent2, n):
   child1 = parent1
   child2 = parent2
   param_names = []
   if n == 0:
       param_names = param_names_actor
   elif n == 1:
       param_names = param_names_critic
   for layer in param_names:
       # tofix: separate param_names to actor and critic; iterate based on parent1 params like for param name in parent1
       origsize = parent1[layer].size()
       childpart1 = torch.flatten(parent1[layer])
       childpart2 = torch.flatten(parent2[layer])
       for i in range(torch.numel(childpart1)):
           if random.rand() < CROSSOVER_RATE:
               bit1 = childpart1[i]
               bit2 = childpart2[i]


               # y1 is smaller val for computation
               sm = bit1 < bit2
               y1 = np.where(sm, bit1, bit2)
               y2 = np.where(sm, bit2, bit1)


               # stolen from pymoo repo
               # eta is distribution index, hyperparameter LOL


               # difference between all variables
               delta = (y2 - y1)


               # xl and xu were lower and upper bounds of problem (aka tensor value)


               randinit = random.rand()
               xl, xu = BOUNDS_DICT[layer]
               beta = 1.0 + (2.0 * (y1 - xl) / delta)
               betaq = calc_betaq(beta, randinit)
               c1 = 0.5 * ((y1 + y2) - betaq * delta)


               beta = 1.0 + (2.0 * (xu - y2) / delta)
               betaq = calc_betaq(beta, randinit)
               c2 = 0.5 * ((y1 + y2) + betaq * delta)


               if random.rand() < 0.5:
                   childpart1[i] = torch.from_numpy(np.where(sm, c1, c2)).float()
                   childpart2[i] = torch.from_numpy(np.where(sm, c2, c1)).float()
               else:
                   childpart1[i] = torch.from_numpy(np.where(sm, c1, c2)).float()
                   childpart2[i] = torch.from_numpy(np.where(sm, c2, c1)).float()


               # REMEMBER TO ADD MUTATION SOMETIME


       child1[layer] = torch.reshape(childpart1, origsize)
       child2[layer] = torch.reshape(childpart2, origsize)
   return child1, child2


def OLDSPX(parent1, parent2, n):
   child1 = parent1
   child2 = parent2
   param_names = []
   if n == 0:
       param_names = param_names_actor
   elif n == 1:
       param_names = param_names_critic
   for layer in param_names:
       # tofix: separate param_names to actor and critic; iterate based on parent1 params like for param name in parent1
       origsize = parent1[layer].size()
       childpart1 = torch.flatten(parent1[layer])
       childpart2 = torch.flatten(parent2[layer])
       layersize = torch.numel(childpart1)


       index = int(layersize * random.rand(1)[0])
       temp1 = torch.cat((childpart1[0:index], childpart2[index:layersize]))
       temp2 = torch.cat((childpart2[0:index], childpart1[index:layersize]))
       child1[layer] = torch.reshape(temp1, origsize)
       child2[layer] = torch.reshape(temp2, origsize)
   return child1, child2

def SPX(parent1, parent2, n):
    child1 = parent1
    child2 = parent2
    param_names = []
    totalsize = 0
    if n == 0:
        param_names = param_names_actor
    elif n == 1:
        param_names = param_names_critic
    for layer in param_names:
        origsize = torch.numel(parent1[layer])
        totalsize += origsize

    crossover_size = random.randint(0, totalsize)
    for layers in param_names:
         a = torch.numel(parent1[layers])
         if crossover_size < a:
             break
         else:
             crossover_size -= a
             child1[layers] = parent2[layers]
             child2[layers] = parent1[layers]

    return child1, child2


   



def init_weights(m):
   if isinstance(m, nn.Linear):
       torch.nn.init.xavier_uniform(m.weight)
       m.bias.data.fill_(0.01)


class Actor(nn.Module):
   def __init__(self, input, actions):
       super(Actor, self).__init__()


       self.net = nn.Sequential(
           nn.Linear(input, 128),
           nn.ReLU(),
           nn.Linear(128, 64))


       self.actor = nn.Sequential(
           nn.Linear(64, actions),
           nn.Softmax(dim=0),
       )


       self.net.apply(init_weights)


       self.actor.apply(init_weights)


   def forward(self, x):
       x = self.net(x)
       return self.actor(x)




class Critic(nn.Module):
   def __init__(self, input):
       super(Critic, self).__init__()


       self.net = nn.Sequential(
           nn.Linear(input, 128),
           nn.ReLU(),
           nn.Linear(128, 64))


       self.critic = nn.Sequential(
           nn.Linear(64, 1),
       )


       self.net.apply(init_weights)


       self.critic.apply(init_weights)


   def forward(self, x):
       x = self.net(x)
       return self.critic(x)




def Test_Agent():
   winningScores = []
   explorationScores = []
   for game in range(3):
       explorationScore = 0
       data = conn.recv(4096)
       temp = np.frombuffer(data, dtype="float64")
       state = torch.FloatTensor(temp)
       finished = False
       while not finished:
           action_probs = TestPongActor(state)
           action_distribution = Categorical(action_probs)
           chosen_action = action_distribution.sample()
           #value = TestPongCritic(state)


           # next_state, reward, done, _, _ = env.step(chosen_action.item())
           conn.sendall(struct.pack('i', chosen_action.item()))
           data = conn.recv(4096)
           data = torch.FloatTensor(np.frombuffer(data, dtype="float64"))
           finished = data[-1].item()
           #reward = data[-5:-1].numpy()
           state = data[0:state_final_index]
           explorationScore += 1
       winningScores.append(state[-1].item())
       explorationScores.append(explorationScore)
   return sum(explorationScores), sum(winningScores)




# bind socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen()
conn, addr = s.accept()




# pongCritic = Critic(shared = pongActor.net) #shared layers might be wrong
current_state_dict = 0
P_state_dicts = []
Prime_state_dicts = []
Q_state_dicts = []
P_scores = []
Prime_scores = []
Q_scores = []


# values determined by hyperparameters
params = int((FRAMESTACK * 4) + 2)
state_final_index = int(-FRAMESTACK - 1)




pongActor = Actor(params, ACTION_SPACE) # <- CHANGE LATER BRO
pongCritic = Critic(params)


TestPongActor = Actor(params, ACTION_SPACE) # <- CHANGE LATER BRO
TestPongCritic = Critic(params)




# initial P, both scores and dicts
print("initializing first generation P")
for i in range(POPULATION_SIZE):
   temp = Actor(params,  ACTION_SPACE)
   tempc = Critic(params)
   state_dict_actor = temp.state_dict()
   state_dict_critic = tempc.state_dict()
   P_state_dicts.append([state_dict_actor, state_dict_critic])
   TestPongActor.load_state_dict(state_dict_actor)
   TestPongCritic.load_state_dict(state_dict_critic)
   print(i)
   a, b = Test_Agent()
   P_scores.append([a, b])
print("finished initializing first generation P")




param_names_actor = []
param_sizes_actor = []
for name, param in pongActor.named_parameters():
   if param.requires_grad:
       param_sizes_actor.append(param.data.size().numel())
       param_names_actor.append(name)


param_names_critic = []
param_sizes_critic = []
for name, param in pongCritic.named_parameters():
   if param.requires_grad:
       param_sizes_critic.append(param.data.size().numel())
       param_names_critic.append(name)




# initial Prime dicts
P_state_dicts_copy = copy.deepcopy(P_state_dicts)
P_scores_copy = copy.deepcopy(P_scores)
for numbers in range(POPULATION_SIZE // 4):
   rand1 = 0
   rand2 = 0
   while rand1 == rand2:
       rand1 = random.randint(0, len(P_state_dicts_copy))
       rand2 = random.randint(0, len(P_state_dicts_copy))
   # rand2 has to be bigger bc it's popped first
   if rand1 > rand2:
       rand1, rand2 = rand2, rand1


   if dominate_max(P_scores_copy[rand1], P_scores_copy[rand2]): # rand1 is better than rand2
       actor1, critic1 = P_state_dicts_copy[rand1]
   elif dominate_max(P_scores_copy[rand2], P_scores_copy[rand1]): # rand1 is better than rand2
       actor1, critic1 = P_state_dicts_copy[rand2]
   elif random.randint(0, 2) == 0:
       actor1, critic1 = P_state_dicts_copy[rand1]
   else:
       actor1, critic1 = P_state_dicts_copy[rand2]
   P_state_dicts_copy.pop(rand2)
   P_state_dicts_copy.pop(rand1)
   P_scores_copy.pop(rand2)
   P_scores_copy.pop(rand1)


   rand1 = 0
   rand2 = 0
   while rand1 == rand2:
       rand1 = random.randint(0, len(P_state_dicts_copy))
       rand2 = random.randint(0, len(P_state_dicts_copy))
   # rand2 has to be bigger bc it's popped first
   if rand1 > rand2:
       rand1, rand2 = rand2, rand1
   if dominate_max(P_scores_copy[rand1], P_scores_copy[rand2]):  # rand1 is better than rand2
       actor2, critic2 = P_state_dicts_copy[rand1]
   elif dominate_max(P_scores_copy[rand2], P_scores_copy[rand1]):  # rand1 is better than rand2
       actor2, critic2 = P_state_dicts_copy[rand1]
   elif random.randint(0, 2) == 0:
       actor2, critic2 = P_state_dicts_copy[rand1]
   else:
       actor2, critic2 = P_state_dicts_copy[rand1]




   P_state_dicts_copy.pop(rand2)
   P_state_dicts_copy.pop(rand1)
   P_scores_copy.pop(rand2)
   P_scores_copy.pop(rand1)


   actor1, actor2 = SPX(actor1, actor2, 0)
   critic1, critic2 = SPX(critic1, critic2, 1)
   Prime_state_dicts.append([actor1, critic1])
   Prime_state_dicts.append([actor2, critic2])








pongActor.load_state_dict(Prime_state_dicts[0][0])
pongCritic.load_state_dict(Prime_state_dicts[0][1])


# optimizers
Actoroptimizer = optim.Adam(pongActor.parameters(), lr=LEARNING_RATE)
Criticoptimizer = optim.Adam(pongCritic.parameters(), lr=LEARNING_RATE, eps=1e-3)


done = False
term = False


pongActor.train()
pongCritic.train()
TestPongActor.eval()
TestPongCritic.eval()




for i in range(10000):  # episodes
   # initialize state
   data = conn.recv(4096)
   temp = np.frombuffer(data, dtype="float64")
   state = torch.FloatTensor(temp)
   episode_reward = 0
   done = False




   # values for doing A2C
   counter = 0
   Qval = 0
   actions_logprob = []
   rewards = []
   advantages = []
   values = []
   Qvals = []
   entropies = []




   while not done:
       # save state before next step bc turning to float tensor
       #state = torch.FloatTensor(state)


       # moves
       action_probs = pongActor(state)
       action_distribution = Categorical(action_probs)
       chosen_action = action_distribution.sample()
       value = pongCritic(state)


       #next_state, reward, done, _, _ = env.step(chosen_action.item())
       conn.sendall(struct.pack('i', chosen_action.item()))
       data = conn.recv(4096)
       data = torch.FloatTensor(np.frombuffer(data, dtype="float64"))
       done = data[-1].item()
       reward = data[state_final_index:-1].numpy()
       state = data[0:state_final_index]




       # save values
       actions_logprob.append(action_distribution.log_prob(chosen_action))
       rewards = np.append(rewards, reward)
       values.append(value)
       entropies.append(action_distribution.entropy())
       # get ready for next step
       counter += 1
       #state = next_state
       # env.render()
       # assessment period
       if counter == REWARD_STEPS or done:
           if done:
               lastscore = state[-1].item()
               gameScores.append(lastscore)
               Qval = 0 # -1 makes it MUCH better fsr
               term = True
           else:
               Qval = pongCritic(state)
               Qval = Qval.detach().numpy()[0]

           n = len(rewards)
           Qvals = np.zeros(math.ceil(n / FRAMESTACK))

           for t in reversed(range(n)):
               Qval = rewards[t] + GAMMA * Qval  # going from last to first, but why he did it like this I have no fucking clue
               if t % FRAMESTACK == 0:
                   Qvals[t // FRAMESTACK] = Qval

           if np.size(Qvals) > len(values): # fix later; idk how to fix ts
               print("some errors in npsize still present")
               Qvals = np.delete(Qvals, -1)


           values = torch.cat(values)
           Qvals = torch.FloatTensor(Qvals)
           entropies = torch.stack(entropies)


           advantages = Qvals - values
           advantages = torch.FloatTensor(advantages)


           actions_logprob = torch.stack(actions_logprob)
           actor_loss = (-actions_logprob * advantages.detach()).mean()  # .detach optional?


           advantages = advantages.pow(2).mean()
           critic_loss = 0.5 * advantages


           entropy_loss = -ENTROPY_BETA * entropies.mean()


           critic_loss = critic_loss + entropy_loss


           if episode_reward > 6767676767676767: # fix later
               torch.save(pongActor.state_dict(), PATHA)
               torch.save(pongCritic.state_dict(), PATHC)
               print("saved model!")
               sys.exit(0)







           Actoroptimizer.zero_grad()
           actor_loss.backward(retain_graph=True)


           Criticoptimizer.zero_grad()
           critic_loss.backward()


           torch.nn.utils.clip_grad_norm_(pongActor.parameters(), 0.5)
           torch.nn.utils.clip_grad_norm_(pongCritic.parameters(), 0.5)


           Actoroptimizer.step()
           Criticoptimizer.step()

           new_ep_reward = episode_reward
           for item in rewards:
               new_ep_reward += item
           episode_reward = new_ep_reward
           # resetting variables after updating model
           counter = 0
           Qval = 0
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

   """if i == 1000:
       Q_TRAINING_EPISODES = 10
       temp = Actor(params, ACTION_SPACE)
       tempc = Critic(params)
       state_dict_actor = temp.state_dict()
       state_dict_critic = tempc.state_dict()
       Prime_state_dicts[0][0] = state_dict_actor
       Prime_state_dicts[0][1] = state_dict_critic
       pongActor.load_state_dict(Prime_state_dicts[0][0])
       pongCritic.load_state_dict(Prime_state_dicts[0][1])
       print("should be done")"""

   # debugging
   if (i + 1) % Q_TRAINING_EPISODES == 0:
       Q_state_dicts.append([pongActor.state_dict(), pongCritic.state_dict()])
       current_state_dict = (current_state_dict + 1) % (len(Prime_state_dicts))


       if current_state_dict == 0 and i != 0:




           # testing scores of each algorithm (NOT DONE)
           # supposedly, all P dicts can get their results from the previous generation, since they're unmodified


           # TODO: get initial scores for P_dicts, then assign the final pop scores to the scores of P
           # testing Prime and Q
           Prime_scores = []
           Q_scores = []
           for individual in range(len(Prime_state_dicts)):
               a, b = Prime_state_dicts[individual]
               TestPongActor.load_state_dict(a)
               TestPongCritic.load_state_dict(b)
               a, b = Test_Agent()
               Prime_scores.append([a, b])
           for individual in range(len(Q_state_dicts)):
               a, b = Q_state_dicts[individual]
               TestPongActor.load_state_dict(a)
               TestPongCritic.load_state_dict(b)
               a, b = Test_Agent()
               Q_scores.append([a, b])


           # step 1: finding paeto frontiers
           pop_indexes = []
           pop_scores = []
           for element in P_scores:
               pop_scores.append(element)
           for element in Prime_scores:
               pop_scores.append(element)
           for element in Q_scores:
               pop_scores.append(element)


           # finding set of paeto frontiers; algo 2 in wuji paper
           pop_scores_copy = copy.deepcopy(pop_scores)  # replacing values already put in frontier with [-1, -1]
           pop_indexes = []
           done_sorting = False
           current_frontier = []
           while not done_sorting:
               for thing in range(len(pop_scores_copy)):
                   not_dominated = True
                   if pop_scores_copy[thing] == [-1, -1]:
                       continue
                   for comparator in range(len(pop_scores_copy)):
                       if dominate_max(pop_scores_copy[comparator], pop_scores_copy[thing]):
                           not_dominated = False


                   if not_dominated:
                       current_frontier.append(thing)


               for item in current_frontier:
                   pop_scores_copy[item] = [-1, -1]
               pop_indexes.append(current_frontier)
               current_frontier = []
               allnegs = True
               for thing in pop_scores_copy:
                   if thing != [-1, -1]:
                       allnegs = False
                   done_sorting = allnegs


           #algorithm 3 of Wuji Paper


           final_pop_indexes = []
           for frontier in pop_indexes:
               if len(final_pop_indexes) == POPULATION_SIZE:
                   break
               elif len(final_pop_indexes) + len(frontier) < POPULATION_SIZE:
                   for item in frontier:
                       final_pop_indexes.append(item)
               elif len(final_pop_indexes) + len(frontier) == POPULATION_SIZE:
                   for item in frontier:
                       final_pop_indexes.append(item)
                   break
               # algorithm 4 of Wuji Paper
               else:
                   WS = 0
                   ES = 0
                   WSindex = 0
                   ESindex = 0
                   # search for highest winning and exploration score and add it to the list
                   for index in frontier:
                       if pop_scores[index][0] > ES:
                           ES = pop_scores[index][0]
                           ESindex = index
                       if pop_scores[index][1] > WS:
                           WS = pop_scores[index][0]
                           WSindex = index


                   if len(final_pop_indexes) + 2 > POPULATION_SIZE:
                       final_pop_indexes.append(ESindex)
                   elif len(final_pop_indexes) + 2 == POPULATION_SIZE:
                       final_pop_indexes.append(ESindex)
                       final_pop_indexes.append(WSindex)
                   else:
                       # the CD sort proper
                       distances_list = []
                       for item in frontier:
                           distance = 0
                           if item == WSindex or item == ESindex:
                               distances_list.append(float('-inf'))
                               continue
                           else:
                               closest_low_WS_index = 0
                               closest_low_ES_index = 0
                               closest_high_WS_index = 0
                               closest_high_ES_index = 0
                               for comparator in frontier:
                                   if pop_scores[item][1] - pop_scores[comparator][1] > 0 and pop_scores[closest_low_WS_index][1] - pop_scores[comparator][1] < 0:
                                       closest_low_WS_index = comparator
                                   elif pop_scores[item][0] - pop_scores[comparator][0] > 0 and pop_scores[closest_low_ES_index][0] - pop_scores[comparator][0] > 0:
                                       closest_low_ES_index = comparator
                                   if pop_scores[item][1] - pop_scores[comparator][1] < 0 and pop_scores[closest_high_WS_index][1] - pop_scores[comparator][1] > 0:
                                       closest_high_WS_index = comparator
                                   elif pop_scores[item][0] - pop_scores[comparator][0] < 0 and pop_scores[closest_high_ES_index][0] - pop_scores[comparator][0] > 0:
                                       closest_high_ES_index = comparator
                               print("sample in CD sort:")
                               print(pop_scores[closest_low_ES_index])
                               print(pop_scores[closest_high_ES_index])
                               print(pop_scores[item])
                               distance += abs(pop_scores[item][1] - pop_scores[closest_low_WS_index][1])
                               distance += abs(pop_scores[item][1] - pop_scores[closest_high_WS_index][1])
                               distance += abs(pop_scores[item][0] - pop_scores[closest_low_ES_index][0])
                               distance += abs(pop_scores[item][0] - pop_scores[closest_high_ES_index][0])
                               distances_list.append(distance)








                       # sorting distances


                       sorted_distances_indexes = []
                       for temp in range(len(distances_list)):
                           max_distance = 0
                           max_distance_index = 0
                           for temp2 in range(len(distances_list)):
                               if distances_list[temp2] > max_distance:
                                   max_distance = distances_list[temp2]
                                   max_distance_index = temp2


                           sorted_distances_indexes.append(max_distance_index)
                           distances_list[max_distance_index] = float('-inf')




                       temp = 0
                       while len(final_pop_indexes) < POPULATION_SIZE:
                           final_pop_indexes.append(sorted_distances_indexes[temp])
                           temp += 1








           #TODO: figure out why these vlaues are empty
           P_scores = []
           P_state_dicts_copy = []
           if (i + 1) % 100 == 0:
               f = open("frontiers.txt", "a")
               f.write(str(final_pop_indexes))
               f.write(str(pop_scores))
               f.close()
           for item in final_pop_indexes:
               P_scores.append(pop_scores[item])
               if item < POPULATION_SIZE :
                   P_state_dicts_copy.append(P_state_dicts[item])
               elif item >= int((3/2) * POPULATION_SIZE):
                   item -= int(3/2 * POPULATION_SIZE)
                   P_state_dicts_copy.append(Q_state_dicts[item])
               elif item >= POPULATION_SIZE:
                   item -= POPULATION_SIZE
                   P_state_dicts_copy.append(Prime_state_dicts[item])
               else:
                   print("SOMETHING WRONG WITH THE FINAL STEP")
                   print(item)
                   print(POPULATION_SIZE)


           P_state_dicts = copy.deepcopy(P_state_dicts_copy)
           Prime_state_dicts = []
           Q_state_dicts = []




           # start of Prime dict generation
           # first do binary selection


           #SPX + mutation thing to get Prime from P


           P_state_dicts_copy = copy.deepcopy(P_state_dicts)
           P_scores_copy = copy.deepcopy(P_scores)
           for numbers in range(POPULATION_SIZE // 4):
               rand1 = 0
               rand2 = 0
               while rand1 == rand2:
                   rand1 = random.randint(0, len(P_state_dicts_copy))
                   rand2 = random.randint(0, len(P_state_dicts_copy))
               # rand2 has to be bigger bc it's popped first
               if rand1 > rand2:
                   rand1, rand2 = rand2, rand1


               if dominate_max(P_scores_copy[rand1], P_scores_copy[rand2]):  # rand1 is better than rand2
                   actor1, critic1 = P_state_dicts_copy[rand1]
               elif dominate_max(P_scores_copy[rand2], P_scores_copy[rand1]):  # rand1 is better than rand2
                   actor1, critic1 = P_state_dicts_copy[rand2]
               elif random.randint(0, 2) == 0:
                   actor1, critic1 = P_state_dicts_copy[rand1]
               else:
                   actor1, critic1 = P_state_dicts_copy[rand2]
               P_state_dicts_copy.pop(rand2)
               P_state_dicts_copy.pop(rand1)
               P_scores_copy.pop(rand2)
               P_scores_copy.pop(rand1)


               rand1 = 0
               rand2 = 0
               while rand1 == rand2:
                   rand1 = random.randint(0, len(P_state_dicts_copy))
                   rand2 = random.randint(0, len(P_state_dicts_copy))
               # rand2 has to be bigger bc it's popped first
               if rand1 > rand2:
                   rand1, rand2 = rand2, rand1
               if dominate_max(P_scores_copy[rand1], P_scores_copy[rand2]):  # rand1 is better than rand2
                   actor2, critic2 = P_state_dicts_copy[rand1]
               elif dominate_max(P_scores_copy[rand2], P_scores_copy[rand1]):  # rand1 is better than rand2
                   actor2, critic2 = P_state_dicts_copy[rand1]
               elif random.randint(0, 2) == 0:
                   actor2, critic2 = P_state_dicts_copy[rand1]
               else:
                   actor2, critic2 = P_state_dicts_copy[rand1]


               P_state_dicts_copy.pop(rand2)
               P_state_dicts_copy.pop(rand1)
               P_scores_copy.pop(rand2)
               P_scores_copy.pop(rand1)


               actor1, actor2 = SPX(actor1, actor2, 0)
               critic1, critic2 = SPX(critic1, critic2, 1)
               Prime_state_dicts.append([actor1, critic1])
               Prime_state_dicts.append([actor2, critic2])


           # now, P should be POP_SIZE, Prime should be half P, and Q should be Prime

       if (i + 1) % 100 == 0:
           daindexfordicts += 1
           for lol in range(POPULATION_SIZE):
               PATHA = "A" + str(lol) + str(daindexfordicts) + ".pth"
               PATHC = "C" + str(lol) + str(daindexfordicts) + ".pth"
               torch.save(P_state_dicts[lol][0], PATHA)
               torch.save(P_state_dicts[lol][1], PATHC)

       pongActor.load_state_dict(Prime_state_dicts[current_state_dict][0])
       pongCritic.load_state_dict(Prime_state_dicts[current_state_dict][1])



