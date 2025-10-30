import numpy as np
import torch
import gym
import os
import td3
import replay_buffer
import envs  # This is needed to register the environment
import matplotlib.pyplot as plt

from model import PENN
from run import LR, STATE_DIM


class TrainerTD3:
    def __init__(
        self,
        env_name,
        seed,
        start_timesteps,
        eval_freq,
        max_timesteps,
        expl_noise,
        batch_size,
        num_nets,
        device,
    ):
        self.env = gym.make(env_name)
        state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        self.expl_noise = expl_noise

        self.model = PENN(num_nets, STATE_DIM, self.action_dim, LR, device=device)
        self.model.load_state_dict(torch.load("model.pt", map_location=device))

        # Set seeds
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.policy = td3.TD3(state_dim, self.action_dim, self.max_action, device)
        self.memory = replay_buffer.ReplayBuffer(state_dim, self.action_dim)
        self.real_memory = replay_buffer.ReplayBuffer(
            state_dim, self.action_dim, max_size=5000, save_timestep=True
        )
        self.start_timesteps = start_timesteps
        self.eval_freq = eval_freq
        self.expl_noise = expl_noise
        self.batch_size = batch_size
        self.max_timesteps = max_timesteps

    def eval_policy(self, eval_episodes=10):
        avg_reward = 0.0
        successes = 0
        for _ in range(eval_episodes):
            state, done = self.env.reset(), False
            while not done:
                action = self.policy.select_action(np.array(state))
                state, reward, done, info = self.env.step(action)
                avg_reward += reward
                if info.get("done") == "goal reached":
                    successes += 1
        avg_reward /= eval_episodes
        success_rate = successes / eval_episodes
        print("---------------------------------------")
        print(
            f"Evaluation over {eval_episodes} episodes: Reward {avg_reward:.3f}, Success Rate {success_rate:.2f}"
        )
        print("---------------------------------------")
        return avg_reward, success_rate

    def get_synthetic_transition(self, state, action):
        # TODO: write your code here
        # randomly choose a network from ensemble and get new state from it
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
        if not torch.is_tensor(action):
            action = torch.tensor(action, dtype=torch.float32, device=self.model.device)
        
        picked_net = torch.randint(self.model.num_nets, size=(1,)).item()
        input = torch.cat([state[:self.model.state_dim].unsqueeze(0), action.unsqueeze(0)], dim=1)
        mean, logvar = self.model.get_output(self.model.networks[picked_net](input))
        delta = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)

        new_state = state[:self.model.state_dim] + delta.squeeze()
        return self.env.step_state(
            new_state.tolist()
        )  # pass in new state to the env for the full transition

    def train_td3_with_mix(
        self,
        synthetic_ratio,
    ):
        state, done = self.env.reset(), False
        episode_reward = 0
        self.episode_timesteps = 0
        episode_num = 0

        # Logging lists
        eval_rewards = []
        eval_success_rates = []

        for t in range(1, int(self.max_timesteps) + 1):
            self.episode_timesteps += 1

            # Select action randomly or from policy
            if t < self.start_timesteps:
                action = self.env.action_space.sample()
            else:
                action = (
                    self.policy.select_action(np.array(state))
                    + np.random.normal(
                        0, self.max_action * self.expl_noise, size=self.action_dim
                    )
                ).clip(-self.max_action, self.max_action)

            # Perform action
            if np.random.rand() < synthetic_ratio:
                next_state, reward, done, _ = self.get_synthetic_transition(
                    state, action
                )
            else:
                next_state, reward, done, _ = self.env.step(action)

            # Store data in replay buffer
            self.memory.add(state, action, next_state, reward, done)

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= self.start_timesteps:
                self.policy.train(self.memory, batch_size)

            if done:
                print(
                    f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {self.episode_timesteps} Reward: {episode_reward:.3f}"
                )
                # Reset environment
                state, done = self.env.reset(), False
                episode_reward = 0
                self.episode_timesteps = 0
                episode_num += 1

            # Evaluate episode
            if t % self.eval_freq == 0:
                avg_rew, success_rate = self.eval_policy()
                eval_rewards.append(avg_rew)
                eval_success_rates.append(success_rate)

        print("Training finished.")
        print("Evaluation Rewards:", eval_rewards)
        print("Evaluation Success Rates:", eval_success_rates)

        return eval_rewards, eval_success_rates

    def train_with_model_rollouts(self, rollout_length):
        eval_rewards, eval_success_rates = [], []
        state, done = self.env.reset(), False

        for _ in range(self.start_timesteps * 2):
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)
            self.real_memory.add(
                state,
                action,
                next_state,
                reward,
                done,
                timestep=self.env.elapsed_steps - 1,
            )
            state = next_state
            if done:
                state, done = self.env.reset(), False

        t = self.start_timesteps
        while t < self.max_timesteps:
            state, done = self.env.reset(), False

            # Do a real environment rollout
            while not done:
                action = (
                    self.policy.select_action(np.array(state))
                    + np.random.normal(
                        0, self.max_action * self.expl_noise, size=self.action_dim
                    )
                ).clip(-self.max_action, self.max_action)
                next_state, reward, done, _ = self.env.step(action)
                self.real_memory.add(
                    state,
                    action,
                    next_state,
                    reward,
                    done,
                    timestep=self.env.elapsed_steps - 1,
                )
                state = next_state

            # Step 3: Generate model rollouts starting from random past states
            model_state, _, _, _, _, timestep = self.real_memory.sample(1)

            # Set current state to sampled state
            model_state = model_state.squeeze(dim=0).cpu().numpy()
            self.env.set_state(model_state.tolist(), elapsed_steps=timestep)

            rollout_reward = []
            model_done = False

            # Do a fully synthetic rollout
            for _ in range(rollout_length):
                model_action = (
                    self.policy.select_action(np.array(model_state))
                    + np.random.normal(
                        0, self.max_action * self.expl_noise, size=self.action_dim
                    )
                ).clip(-self.max_action, self.max_action)
                next_model_state, model_reward, model_done, _ = (
                    self.get_synthetic_transition(model_state, model_action)
                )
                rollout_reward.append(model_reward)

                # Add the synthetic transition to the training buffer
                self.memory.add(
                    model_state,
                    model_action,
                    next_model_state,
                    model_reward,
                    model_done,
                )
                model_state = next_model_state
                t += 1

                self.policy.train(self.memory, self.batch_size)

                if t % self.eval_freq == 0:
                    print(f"T: {t}")
                    avg_rew, success_rate = self.eval_policy()
                    eval_rewards.append(avg_rew)
                    eval_success_rates.append(success_rate)

                if model_done:
                    break

        return eval_rewards, eval_success_rates

    def compute_and_plot_model_error(self, max_rollout_length=20, num_trajectories=10):
        all_errors = []
        for _ in range(num_trajectories):
            real_state, done = self.env.reset(), False
            model_state = real_state.copy()
            trajectory_errors = []

            real_trajectory = []
            # Do real rollout
            while not done:
                action = self.policy.select_action(np.array(real_state))
                real_trajectory.append((real_state, action, self.env.elapsed_steps))
                next_real_state, _, done, _ = self.env.step(action)
                real_state = next_real_state

            # Set initial state
            start_idx = np.random.randint(
                low=0, high=len(real_trajectory) - max_rollout_length
            )
            model_state, action, timestep = real_trajectory[start_idx]
            self.env.elapsed_steps = timestep
            self.env.set_state(model_state.tolist(), elapsed_steps=timestep)

            # Rollout with model
            for step in range(max_rollout_length):
                next_model_state, _, _, _ = self.get_synthetic_transition(
                    model_state, action
                )
                real_state, _, _ = real_trajectory[start_idx + step]
                error = np.linalg.norm(real_state[:8] - next_model_state[:8])
                trajectory_errors.append(error)
                model_state = next_model_state

            all_errors.append(trajectory_errors)

        # Pad trajectories with NaNs so we can compute per-timestep stats
        max_T = max(len(traj) for traj in all_errors)

        err_mat = np.full((len(all_errors), max_T), np.nan, dtype=np.float32)
        for i, traj in enumerate(all_errors):
            err_mat[i, : len(traj)] = np.asarray(traj, dtype=np.float32)

        mean_err = np.nanmean(err_mat, axis=0)
        count = np.sum(~np.isnan(err_mat), axis=0).astype(np.float32)
        sem_err = np.nanstd(err_mat, axis=0, ddof=0) / np.sqrt(np.maximum(count, 1.0))

        timesteps = np.arange(max_T)

        plt.figure()
        plt.plot(timesteps, mean_err, linewidth=2, label="Mean error")
        plt.fill_between(
            timesteps,
            mean_err - sem_err,
            mean_err + sem_err,
            alpha=0.25,
            label="Std. error",
        )
        plt.xlabel("Model rollout step")
        plt.ylabel("Prediction error (L2)")
        plt.title("Model prediction error across trajectories")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig("l2_error_rollout.png")


if __name__ == "__main__":
    env_name = "Pushing2D-v1"
    seeds = list(range(3))
    start_timesteps = 5000  # How many steps to take random actions for
    eval_freq = 2500  # How often to run evaluation
    max_timesteps = 75000  # Max time steps to run environment
    expl_noise = 0.1  # Std of Gaussian exploration noise
    batch_size = 256
    num_nets = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_iterations = np.arange(1, (max_timesteps // eval_freq) + 1) * eval_freq

    # Q1.4.1
    synthetic_ratios = [0.0, 0.2, 0.5, 0.8, 1.0]
    all_success_rates = {}

    for synthetic_ratio in synthetic_ratios:
        all_success_rates[synthetic_ratio] = []

        for seed in seeds:
            trainer = TrainerTD3(
                env_name,
                seed,
                start_timesteps,
                eval_freq,
                max_timesteps,
                expl_noise,
                batch_size,
                num_nets,
                device,
            )
            _, eval_success_rates = trainer.train_td3_with_mix(
                synthetic_ratio,
            )
            all_success_rates[synthetic_ratio].append(eval_success_rates)

        all_success_rates[synthetic_ratio] = np.mean(
            np.array(all_success_rates[synthetic_ratio]), axis=0
        )

    plt.figure(figsize=(12, 8))
    for ratio, rates in all_success_rates.items():
        plt.plot(
            eval_iterations[: len(rates)],
            rates,
            marker=None,
            linestyle="-",
            label=f"Ratio = {ratio}",
        )
    plt.title("Success Rate vs. Total Timesteps", fontsize=16)
    plt.xlabel("Total Timesteps", fontsize=14)
    plt.ylabel("Success Rate", fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("synthetic_ratio_success_rate.png")

    # Q1.4.2
    rollout_lengths = [2, 4, 8]
    all_success_rates = {}
    for length in rollout_lengths:
        all_success_rates[length] = []
        for seed in seeds:
            print(f"\nTraining with Rollout Length = {length}")
            trainer = TrainerTD3(
                env_name,
                seed,
                start_timesteps,
                eval_freq,
                max_timesteps,
                expl_noise,
                batch_size,
                num_nets,
                device,
            )
            _, success_rates = trainer.train_with_model_rollouts(length)
            all_success_rates[length].append(success_rates)

        all_success_rates[length] = np.mean(np.array(all_success_rates[length]), axis=0)

    plt.figure(figsize=(12, 8))
    for rollout_length, rates in all_success_rates.items():
        plt.plot(
            eval_iterations[: len(rates)],
            rates,
            marker=None,
            linestyle="-",
            label=f"Rollout Length = {rollout_length}",
        )
    plt.title("Success Rate vs. Total Timesteps", fontsize=16)
    plt.xlabel("Total Timesteps", fontsize=14)
    plt.ylabel("Success Rate", fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("rollout_length_success_rate.png")

    # Q1.4.3
    trainer = TrainerTD3(
        env_name,
        0,
        start_timesteps,
        eval_freq,
        max_timesteps,
        expl_noise,
        batch_size,
        num_nets,
        device,
    )
    trainer.compute_and_plot_model_error(max_rollout_length=15, num_trajectories=50)
