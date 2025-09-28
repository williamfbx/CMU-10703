#! python3

import argparse
import collections
import random
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np # NOTE only imported because https://github.com/pytorch/pytorch/issues/13918
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayMemory():
    def __init__(self, memory_size, batch_size):
        # define init params
        # use collections.deque
        # BEGIN STUDENT SOLUTION
        # https://docs.python.org/3/library/collections.html#collections.deque
        self.buffer = collections.deque(maxlen=memory_size)
        self.batch_size = batch_size
        # END STUDENT SOLUTION

    def sample_batch(self):
        # randomly chooses from the collections.deque
        # BEGIN STUDENT SOLUTION
        buffer_size = len(self.buffer)
        idxs = random.sample(range(buffer_size), self.batch_size)
        batch = [self.buffer[idx] for idx in idxs]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states, axis=0),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states, axis=0),
            np.array(dones, dtype=np.float32),
        )
        # END STUDENT SOLUTION

    def append(self, transition):
        # append to the collections.deque
        # BEGIN STUDENT SOLUTION
        self.buffer.append(transition)
        # END STUDENT SOLUTION


class DeepQNetwork(nn.Module):
    def __init__(self, state_size, action_size, double_dqn, lr_q_net=2e-4, gamma=0.99, epsilon=0.05, target_update=50, burn_in=10000, replay_buffer_size=50000, replay_buffer_batch_size=32, device='cpu'):
        super(DeepQNetwork, self).__init__()

        # define init params
        self.state_size = state_size
        self.action_size = action_size
        self.double_dqn = double_dqn

        self.gamma = gamma
        self.epsilon = epsilon

        self.target_update = target_update

        self.burn_in = burn_in

        self.device = device

        hidden_layer_size = 256

        # q network
        q_net_init = lambda: nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            # BEGIN STUDENT SOLUTION
            nn.Linear(hidden_layer_size, action_size)
            # END STUDENT SOLUTION
        )

        # initialize replay buffer, networks, optimizer, move networks to device
        # BEGIN STUDENT SOLUTION
        self.replay = ReplayMemory(replay_buffer_size, replay_buffer_batch_size)
        
        self.q_net = q_net_init().to(device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr_q_net)
        
        self.target_net = q_net_init().to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        # END STUDENT SOLUTION

    def forward(self, state, action, new_state):
        # calculate q value and target
        # use the correct network for the target based on self.double_dqn
        # BEGIN STUDENT SOLUTION
        Q_state = self.q_net(state)
        Q_curr = Q_state.gather(1, action.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            if self.double_dqn:
                Q_state_online = self.q_net(new_state)
                actions_online = Q_state_online.argmax(dim=1, keepdim=True)
                Q_state_next = self.target_net(new_state)
                Q_next = Q_state_next.gather(1, actions_online).squeeze(1)
            else:
                Q_state_next = self.q_net(new_state)
                Q_next = Q_state_next.max(dim=1).values

        return Q_curr, Q_next
        # END STUDENT SOLUTION

    def get_action(self, state, stochastic):
        # if stochastic, sample using epsilon greedy, else get the argmax
        # BEGIN STUDENT SOLUTION
        if stochastic and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return q_values.argmax().item()
        # END STUDENT SOLUTION
        
    def train(self, env, num_episodes, max_steps, test_frequency, num_test_episodes):
        
        self.burn_in_buffer(env)
        test_means = []
        
        for episode in range(1, num_episodes + 1):
            observation, _ = env.reset()
            
            for _ in range(max_steps):
                action = self.get_action(observation, stochastic=True)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                self.replay.append((observation, action, reward, next_observation, done))
                
                self.optimize_model()
                
                observation = next_observation
                if done:
                    break
            
            # Update target network
            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
                
            # Freeze for plotting
            if episode % test_frequency == 0:
                episode_returns = self.evaluate(env, max_steps, num_test_episodes)
                test_means.append(np.mean(episode_returns))
                print(f'Episode {episode}/{num_episodes}, Average Test Return: {test_means[-1]}')
                
        return test_means
    
    def evaluate(self, env, max_steps, num_test_episodes):
        episode_returns = []
        for _ in range(num_test_episodes):
            observation, _ = env.reset()
            total_reward = 0.0
            
            for _ in range(max_steps):
                action = self.get_action(observation, stochastic=False)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                observation = next_observation
                if done:
                    break
                
            episode_returns.append(total_reward)
        return episode_returns
                
    def optimize_model(self):
        states, actions, rewards, next_states, dones = self.replay.sample_batch()

        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        Q_curr, Q_next = self.forward(states_tensor, actions_tensor, next_states_tensor)
        y_curr = rewards_tensor + (1.0 - dones_tensor) * self.gamma * Q_next
        loss = F.mse_loss(Q_curr, y_curr)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def burn_in_buffer(self, env):
        observation, _ = env.reset()
        
        for _ in range(self.burn_in):
            action = env.action_space.sample()
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            self.replay.append((observation, action, reward, next_observation, done))
            
            if done:
                observation, _ = env.reset()
            else:
                observation = next_observation
        print(f'Burned in {self.burn_in} timesteps, replay buffer size: {len(self.replay.buffer)}')


def graph_agents(
    graph_name, mean_undiscounted_returns, test_frequency, max_steps, num_episodes
):
    print(f'Starting: {graph_name}')

    # graph the data mentioned in the homework pdf
    # BEGIN STUDENT SOLUTION
    D = np.asarray(mean_undiscounted_returns, dtype=np.float32)
    average_total_rewards = np.mean(D, axis=0)
    min_total_rewards = np.min(D, axis=0)
    max_total_rewards = np.max(D, axis=0)
    
    if not os.path.exists("./graphs"):
        os.makedirs("./graphs")
    # END STUDENT SOLUTION

    # plot the total rewards
    xs = [i * test_frequency for i in range(len(average_total_rewards))]
    fig, ax = plt.subplots()
    plt.fill_between(xs, min_total_rewards, max_total_rewards, alpha=0.1)
    ax.plot(xs, average_total_rewards)
    ax.set_ylim(-max_steps * 0.01, max_steps * 1.1)
    ax.set_title(graph_name, fontsize=10)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Total Reward')
    fig.savefig(f'./graphs/{graph_name}.png')
    plt.close(fig)
    print(f'Finished: {graph_name}')


def parse_args():
    parser = argparse.ArgumentParser(description='Train an agent.')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs to average over for graph')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes to train for')
    parser.add_argument('--max_steps', type=int, default=200, help='Maximum number of steps in the environment')
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Environment name')
    parser.add_argument(
        "--test_frequency",
        type=int,
        default=100,
        help="Number of training episodes between test episodes",
    )
    parser.add_argument("--double_dqn", action="store_true", help="Use Double DQN")
    return parser.parse_args()


def main():
    args = parse_args()

    # init args, agents, and call graph_agent on the initialized agents
    # BEGIN STUDENT SOLUTION
    env = gym.make(args.env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    agents = []
    for _ in range(args.num_runs):
        agents.append(
            DeepQNetwork(
                state_size=state_size,
                action_size=action_size,
                double_dqn=args.double_dqn,
                lr_q_net=2e-4,
                gamma=0.99,
                epsilon=0.05,
                target_update=50,
                burn_in=10000,
                replay_buffer_size=50000,
                replay_buffer_batch_size=32,
                device=device,
            )
        )
        
    mean_undiscounted_returns = []
    for i, agent in enumerate(agents, start=1):
        print(f"Run {i}/{args.num_runs}")
        
        test_means = agent.train(
            env=env,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            test_frequency=args.test_frequency,
            num_test_episodes=20,
        )
        mean_undiscounted_returns.append(test_means)
    
    graph_name = f"{'Double DQN' if args.double_dqn else 'DQN'}"
    graph_agents(
        graph_name=graph_name,
        mean_undiscounted_returns=mean_undiscounted_returns,
        test_frequency=args.test_frequency,
        max_steps=args.max_steps,
        num_episodes=args.num_episodes,
    )
    # END STUDENT SOLUTION


if '__main__' == __name__:
    main()

