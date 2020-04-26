import numpy as np
from agents import DQNAgent
from env_wrapper import make_env
from pathlib import Path
import sys
import warnings
import timeit
from torch.utils import tensorboard

# Parameters
env = make_env('PongNoFrameskip-v4')
load_checkpoint = False
n_games = 1000
do_training = True
save_freq = 10

# File paths
ckpt_dir = Path("C:/Users/kevin/OneDrive/Dokumente/Coding/reinforcement_learning/models")
log_dir = Path("C:/Users/kevin/OneDrive/Dokumente/Coding/reinforcement_learning/logs")

# Tensorboard summar writer for logging
writer = tensorboard.SummaryWriter(log_dir=log_dir)

# Create DQN Agent
agent = DQNAgent(gamma=0.99, epsilon=1, lr=0.0001,
                 input_dims=(env.observation_space.shape),
                 n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
                 batch_size=32, replace=1000, eps_dec=0.99999,
                 chkpt_dir=ckpt_dir, algo='DQNAgent',
                 env_name='PongNoFrameskip-v4' )


# load models if already saved
if load_checkpoint:
    agent.load_models()


n_steps = 0
scores, eps_history, steps_array = [], [], []

# Play games
for game in range(n_games):
    start = timeit.default_timer()
    done = False
    observation = env.reset()

    score = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward

        env.render()

        if do_training:
            agent.store_transition(observation, action, reward, observation_, int(done))
            agent.learn()
        
        observation = observation_
        n_steps += 1

    scores.append(score)
    steps_array.append(n_steps)

    avg_score = np.mean(scores[-50:])
    end = timeit.default_timer()
    game_time = end - start
    writer.add_scalar("Score", score, n_steps)
    writer.add_scalar("Epsilon", agent.epsilon, n_steps)

    print(f"Game: {game}, score: {score}, avg score: {avg_score:.2f}, time: {game_time:.1f}, epsilon: {agent.epsilon:.2f}, steps: {n_steps}")

    if game % save_freq == 0:
        agent.save_models

    eps_history.append(agent.epsilon)
    
    
writer.close()