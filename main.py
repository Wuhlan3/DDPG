import gym
from itertools import count
from tqdm import tqdm
import os
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils_memory import Replay_memory 
from model import Actor, Critic

#超参

tau = 0.005
gamma = 0.99
render = False
log_interval = 50
capacity = 100000
batch_size = 100
test_iter = 10
load = False
exploration_noise = 0.1
target_learn_interval = 1
learn_iteration = 200

env_name = "LunarLanderContinuous-v2"   #选择不同的游戏环境
mode = "train"                          #可以选择train模式，或者test模式
max_episode = 1000                      #episode总次数
#环境配置
env = gym.make(env_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dim_s = env.observation_space.shape[0]  #获取状态s的维度
dim_a = env.action_space.shape[0]       #获取dim_a的维度
max_action = float(env.action_space.high[0])    #动作的幅度，如Pendulum-v0游戏动作的最大值为2.0
save_modelname = './model_' + env_name   #model的名称

class DDPG(object):
    def __init__(self, dim_s, dim_a, max_action):
        #policy网络作为actor，并进行初始化
        self.policy = Actor(dim_s, dim_a, max_action).to(device)
        self.policy_target = Actor(dim_s, dim_a, max_action).to(device)
        self.policy_target.load_state_dict(self.policy.state_dict())
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.actor_counter = 0
        
        #Q网络作为critic，并进行初始化
        self.Q = Critic(dim_s, dim_a).to(device)
        self.Q_target = Critic(dim_s, dim_a).to(device)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.Q_optimizer = optim.Adam(self.Q.parameters(), lr=1e-3)
        self.critic_counter = 0
        
        #初始化内存池，大小为capacity
        self.Replay_memory = Replay_memory(capacity)


    def learn(self):

        for it in range(learn_iteration):
            #从内存池中取出一组数据,取出[s,s',a,r]
            s, next_s, a, r, d = self.Replay_memory.sample(batch_size)
            state = torch.FloatTensor(s).to(device)
            next_state = torch.FloatTensor(next_s).to(device)
            action = torch.FloatTensor(a).to(device)
            reward = torch.FloatTensor(r).to(device)
            done = torch.FloatTensor(1-d).to(device)

            #计算当前的Q
            current_Q = self.Q(state, action)
            #经过Qtarget网络，计算target_Q
            target_Q = self.Q_target(next_state, self.policy_target(next_state))
            target_Q = reward + (done * gamma * target_Q).detach()

            #计算critic的Loss,取均方差,优化critic
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.Q_optimizer.zero_grad()
            critic_loss.backward()
            self.Q_optimizer.step()

            #计算actor的Loss，只需要取 -Q, #优化actor
            actor_loss = -self.Q(state, self.policy(state)).mean()
            self.policy_optimizer.zero_grad()
            actor_loss.backward()
            self.policy_optimizer.step()

            #更新固定后的target网络
            for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.policy.parameters(), self.policy_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def sample(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy(state).cpu().data.numpy().flatten()


    def save(self):
        torch.save(self.policy.state_dict(), save_modelname + '_actor')
        torch.save(self.Q.state_dict(), save_modelname + '_critic')

    def load(self):
        self.policy.load_state_dict(torch.load(save_modelname + '_actor'))
        self.Q.load_state_dict(torch.load(save_modelname + '_critic'))

def main():
    agent = DDPG(dim_s, dim_a, max_action)
    temp_reward = 0
    if mode == 'train':
        if load: 
            agent.load()
        total_step = 0
        for i in tqdm(range(max_episode),desc = "training"):
            total_reward = 0
            step =0
            state = env.reset()
            for t in count():
                action = agent.sample(state)
                action = (action + np.random.normal(0, exploration_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)
                next_state, reward, done, _ = env.step(action)
                agent.Replay_memory.push((state, next_state, action, reward, np.float(done)))
                state = next_state
                if done:
                    break
                step += 1
                total_reward += reward
            total_step += (step+1)
            with open("rewards.txt", "a") as fp:
                fp.write("{} {} {:0.2f}\n".format(total_step, i, total_reward))
            
            agent.learn()
            if i % log_interval == 0:
                agent.save()

    elif mode == 'test':
        agent.load()
        for i in tqdm(range(test_iter)):
            state = env.reset()
            for t in count():
                action = agent.sample(state)
                next_state, reward, done, _ = env.step(np.float32(action))
                temp_reward += reward
                env.render()
                if done :
                    print("{} {:0.2f} {}".format(i, t, temp_reward))
                    temp_reward = 0
                    break
                state = next_state
    

if __name__ == '__main__':
    main()