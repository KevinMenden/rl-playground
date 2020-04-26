"""
Model classes for different RL algorithms
"""
from pathlib import Path
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):
    """
    DQN model according to https://deepmind.com/research/publications/playing-atari-deep-reinforcement-learning
    The model is slightly modified
    """
    def __init__(self, n_actions, name, input_dims, chkpt_dir):
        super(DQN, self).__init__()

        self.checkpoint_dir = Path(chkpt_dir)
        self.checkpoint_file = self.checkpoint_dir / name

        # Model layers
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        fc_input_dims = self.calculate_conv_output_dims(input_dims)
        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, n_actions)


    def calculate_conv_output_dims(self, input_dims):
        # Get size of output conv layer
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=T.device(self.device)))
