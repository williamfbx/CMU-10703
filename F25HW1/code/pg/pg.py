#! python3

import argparse

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np  # NOTE only imported because https://github.com/pytorch/pytorch/issues/13918
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


class PolicyGradient(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        lr_actor=1e-3,
        lr_critic=1e-3,
        mode="REINFORCE",
        n=0,
        gamma=0.99,
        device="cpu",
    ):
        super(PolicyGradient, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        self.mode = mode
        self.n = n
        self.gamma = gamma
        self.device = device
        hidden_layer_size = 256

        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, action_size),
            # BEGIN STUDENT SOLUTION
            nn.Softmax(dim=-1)
            # END STUDENT SOLUTION
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            # BEGIN STUDENT SOLUTION
            nn.Linear(hidden_layer_size, 1)
            # END STUDENT SOLUTION
        )

        # initialize networks, optimizers, move networks to device
        # BEGIN STUDENT SOLUTION
        self.to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic)
        # END STUDENT SOLUTION

    def forward(self, state):
        return (self.actor(state), self.critic(state))

    def get_action(self, state, stochastic):
        # if stochastic, sample using the action probabilities, else get the argmax
        # BEGIN STUDENT SOLUTION
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            probs = self.forward(state)[0]
            
            if stochastic:
                dist = torch.distributions.Categorical(probs=probs)
                action = dist.sample()
            else:
                action = torch.argmax(probs, dim=-1)
                
        return int(action.item())
        # END STUDENT SOLUTION

    def calculate_n_step_bootstrap(self, rewards_tensor, values):
        # calculate n step bootstrap
        # BEGIN STUDENT SOLUTION
        gamma = self.gamma
        R = rewards_tensor
        T = R.shape[0]
        N = self.n

        if self.mode == "REINFORCE":
            returns = torch.zeros_like(R)
            G = 0.0
            for t in reversed(range(T)):
                G = R[t] + gamma * G
                returns[t] = G
            return returns
        
        if self.mode == "REINFORCE_WITH_BASELINE":
            returns = torch.zeros_like(R)
            G = 0.0
            for t in reversed(range(T)):
                G = R[t] + gamma * G
                returns[t] = G
            return returns
        
        if self.mode == "A2C":
            returns = torch.zeros_like(R)
            V_omega = values.squeeze(-1)
            for t in range(T):
                V_end = 0.0
                if t + N < T:
                    V_end = V_omega[t + N]

                G = 0.0
                for k in range(t, min(t + N - 1, T - 1) + 1):
                    G += (gamma ** (k - t)) * R[k]
                G += (gamma ** N) * V_end
                returns[t] = G
            return returns
        # END STUDENT SOLUTION

    def train(self, states, actions, rewards):
        # train the agent using states, actions, and rewards
        # BEGIN STUDENT SOLUTION
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        
        probs, values = self.forward(states_tensor)
        dist = torch.distributions.Categorical(probs=probs)
        logp = dist.log_prob(actions_tensor)
        
        if self.mode == "REINFORCE":
            returns = self.calculate_n_step_bootstrap(rewards_tensor, values.detach())
            advantage = returns

            actor_loss = -(logp * advantage.detach()).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            
        elif self.mode == "REINFORCE_WITH_BASELINE":
            returns = self.calculate_n_step_bootstrap(rewards_tensor, values.detach())
            advantage = returns - values.squeeze()

            actor_loss = -(logp * advantage.detach()).mean()
            critic_loss = F.mse_loss(values.squeeze(), returns.detach())

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
            
        elif self.mode == "A2C":
            returns = self.calculate_n_step_bootstrap(rewards_tensor, values.detach())
            advantage = returns - values.squeeze()

            actor_loss = -(logp * advantage.detach()).mean()
            critic_loss = F.mse_loss(values.squeeze(), returns.detach())

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
        
        # END STUDENT SOLUTION

    def run(self, env, max_steps, num_episodes, train, num_test_episodes, graph_every):
        total_rewards = []
        snapshot_means = []

        # run the agent through the environment num_episodes times for at most max steps
        # BEGIN STUDENT SOLUTION
        for episode in range(num_episodes):
            # env.reset() returns observation, info (https://gymnasium.farama.org/api/env/)
            # print(env.reset())
            state, _ = env.reset()
            episode_reward = 0
            states = []
            actions = []
            rewards = []

            for _ in range(max_steps):
                # sample using the action probabilities
                action = self.get_action(state, stochastic=train)
                # env.step(action) returns observation, reward, terminated, truncated, info (https://gymnasium.farama.org/api/env/)
                # print(env.step(action))
                next_state, reward, terminated, truncated, _ = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                episode_reward += reward
                state = next_state

                if terminated or truncated:
                    break
            total_rewards.append(episode_reward)

            if train:
                self.train(states, actions, rewards)
                
                # freeze for plotting
                if (episode + 1) % graph_every == 0:
                    test_rewards = []
                    
                    for _ in range(num_test_episodes):
                        state, _ = env.reset()
                        test_episode_reward = 0
                        
                        for _ in range(max_steps):
                            # get argmax
                            action = self.get_action(state, stochastic=False)
                            next_state, reward, terminated, truncated, _ = env.step(action)
                            
                            test_episode_reward += reward
                            state = next_state
                            
                            if terminated or truncated:
                                break
                        test_rewards.append(test_episode_reward)
                        
                    snapshot_means.append(np.mean(test_rewards))
                    print(f"Episode {episode+1}/{num_episodes}, Test Average Reward: {snapshot_means[-1]:.2f}")
        
        # END STUDENT SOLUTION
        return total_rewards, snapshot_means


def graph_agents(
    graph_name,
    agents,
    env,
    max_steps,
    num_episodes,
    num_test_episodes,
    graph_every,
):
    print(f"Starting: {graph_name}")

    if agents[0].n != 0:
        graph_name += "_" + str(agents[0].n)

    # graph the data mentioned in the homework pdf
    # BEGIN STUDENT SOLUTION
    num_runs = len(agents)
    num_snapshots = num_episodes // graph_every
    D = np.zeros((num_runs, num_snapshots), dtype=np.float32)
    
    for i, agent in enumerate(agents):
        print(f"Run {i+1}/{num_runs}")
        
        _, snapshot_means = agent.run(
            env=env,
            max_steps=max_steps,
            num_episodes=num_episodes,
            train=True,
            num_test_episodes=num_test_episodes,
            graph_every=graph_every,
        )
        D[i, :] = np.array(snapshot_means)
    
    average_total_rewards = np.mean(D, axis=0)
    max_total_rewards = np.max(D, axis=0)
    min_total_rewards = np.min(D, axis=0)
    
    if not os.path.exists("./graphs"):
        os.makedirs("./graphs")
    # END STUDENT SOLUTION

    # plot the total rewards
    xs = [i * graph_every for i in range(len(average_total_rewards))]
    fig, ax = plt.subplots()
    plt.fill_between(xs, min_total_rewards, max_total_rewards, alpha=0.1)
    ax.plot(xs, average_total_rewards)
    ax.set_ylim(-max_steps * 0.01, max_steps * 1.1)
    ax.set_title(graph_name, fontsize=10)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Total Reward")
    fig.savefig(f"./graphs/{graph_name}.png")
    plt.close(fig)
    print(f"Finished: {graph_name}")


def parse_args():
    mode_choices = ["REINFORCE", "REINFORCE_WITH_BASELINE", "A2C"]

    parser = argparse.ArgumentParser(description="Train an agent.")
    parser.add_argument(
        "--mode",
        type=str,
        default="REINFORCE",
        choices=mode_choices,
        help="Mode to run the agent in",
    )
    parser.add_argument("--n", type=int, default=0, help="The n to use for n step A2C")
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of runs to average over for graph",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=3500, help="Number of episodes to train for"
    )
    parser.add_argument(
        "--num_test_episodes",
        type=int,
        default=20,
        help="Number of episodes to test for every eval step",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Maximum number of steps in the environment",
    )
    parser.add_argument(
        "--env_name", type=str, default="CartPole-v1", help="Environment name"
    )
    parser.add_argument(
        "--graph_every", type=int, default=100, help="Graph every x episodes"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # init args, agents, and call graph_agents on the initialized agents
    # BEGIN STUDENT SOLUTION
    # https://gymnasium.farama.org/api/spaces/
    env = gym.make(args.env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    agents = []
    for _ in range(args.num_runs):
        agents.append(
            PolicyGradient(
                state_size=state_size,
                action_size=action_size,
                lr_actor=1e-3,
                lr_critic=1e-3,
                mode=args.mode,
                n=args.n,
                gamma=0.99,
                device=device,
            )
        )
        
    graph_agents(
        graph_name=args.mode,
        agents=agents,
        env=env,
        max_steps=args.max_steps,
        num_episodes=args.num_episodes,
        num_test_episodes=args.num_test_episodes,
        graph_every=args.graph_every,
    )
    # END STUDENT SOLUTION


if "__main__" == __name__:
    main()
