"""
Classes for different RL agents
"""
import numpy as np
import torch
from models import DQN
from memory import ReplayBuffer
import torch.optim as optim
import torch.nn as nn

class DQNAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=0.9999,
                 replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn', device='cuda:0'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.device = device

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        # Create policy and target DQN models
        self.policy = DQN(self.n_actions, input_dims=self.input_dims, name=self.env_name+'_'+'policy', chkpt_dir=self.chkpt_dir)
        self.target = DQN(self.n_actions, input_dims=self.input_dims, name=self.env_name+'_'+'target', chkpt_dir=self.chkpt_dir)

        # put on correct device (GPU or CPU)
        self.policy.to(device)
        self.target.to(device)

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        # Loss
        self.loss = nn.MSELoss()
        
    def choose_action(self, observation):
        # Choose an action
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation],dtype=torch.float).to(self.device)
            actions = self.policy.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = \
                                self.memory.sample_buffer(self.batch_size)

        states = torch.tensor(state).to(self.device)
        rewards = torch.tensor(reward).to(self.device)
        dones = torch.tensor(done).to(self.device)
        actions = torch.tensor(action).to(self.device)
        states_ = torch.tensor(new_state).to(self.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.target.load_state_dict(self.policy.state_dict())

    def decrement_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_dec
        
    def save_models(self):
        self.policy.save_checkpoint()

    def load_models(self):
        self.policy.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.policy.forward(states)[indices, actions]
        q_next = self.target.forward(states_).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        loss = self.loss(q_target, q_pred).to(self.device)
        loss.backward()
        self.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()