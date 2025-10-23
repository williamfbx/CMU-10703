import gymnasium as gym
import numpy as np
import torch
from torch import device, nn
import argparse
import imageio
from modules import PolicyNet
from simple_network import SimpleNet
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

try:
    import wandb
except ImportError:
    wandb = None

class TrainDaggerBC:

    def __init__(self, env, model, optimizer, states, actions, expert_model=None, device="cpu", mode="DAgger"):
        """
        Initializes the TrainDAgger class. Creates necessary data structures.

        Args:
            env: an OpenAI Gym environment.
            model: the model to be trained.
            expert_model: the expert model that provides the expert actions.
            device: the device to be used for training.
            mode: the mode to be used for training. Either "DAgger" or "BC".

        """
        self.env = env
        self.model = model
        self.expert_model = expert_model
        self.optimizer = optimizer
        self.device = device
        model.set_device(self.device)

        self.mode = mode

        if self.mode == "BC":
            self.states = []
            self.actions = []
            self.timesteps = []
            for trajectory in range(states.shape[0]):
                trajectory_mask = states[trajectory].sum(axis=1) != 0
                self.states.append(states[trajectory][trajectory_mask])
                self.actions.append(actions[trajectory][trajectory_mask])
                self.timesteps.append(np.arange(0, trajectory_mask.sum()))
            self.states = np.concatenate(self.states, axis=0)
            self.actions = np.concatenate(self.actions, axis=0)
            self.timesteps = np.concatenate(self.timesteps, axis=0)

            self.clip_sample_range = 1
            self.actions = np.clip(self.actions, -self.clip_sample_range, self.clip_sample_range)

        else:
            self.expert_model = self.expert_model.to(self.device)
            self.states = None
            self.actions = None
            self.timesteps = None

    def generate_trajectory(self, env, policy, render=False):
        """Collects one rollout from the policy in an environment. The environment
        should implement the OpenAI Gym interface. A rollout ends when done=True. The
        number of states and actions should be the same, so you should not include
        the final state when done=True.

        Args:
            env: an OpenAI Gym environment.
            policy: The output of a deep neural network
            render: Whether to store frames from the environment
            Returns:
            states: a list of states visited by the agent.
            actions: a list of actions taken by the agent. Note that these actions should never actually be trained on...
            timesteps: a list of integers, where timesteps[i] is the timestep at which states[i] was visited.
            rewards: list of rewards given by the environment
            rgbs: list of rgb images from the environment for each timestep
        """

        states, old_actions, timesteps, rewards, rgbs = [], [], [], [], []

        done, trunc = False, False
        cur_state, _ = env.reset()  
        if render:
            rgbs.append(env.render())
        t = 0
        while (not done) and (not trunc):
            with torch.no_grad():
                p = policy(torch.from_numpy(cur_state).to(self.device).float().unsqueeze(0), torch.tensor(t).to(self.device).long().unsqueeze(0))
            a = p.cpu().numpy()[0]
            next_state, reward, done, trunc, _ = env.step(a)

            states.append(cur_state)
            old_actions.append(a)
            timesteps.append(t)
            rewards.append(reward)
            if render:
                rgbs.append(env.render())

            t += 1

            cur_state = next_state

        return states, old_actions, timesteps, rewards, rgbs

    def call_expert_policy(self, state):
        """
        Calls the expert policy to get an action.

        Args:
            state: the current state of the environment.
        """
        # takes in a np array state and returns an np array action
        with torch.no_grad():
            state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32, device=self.device)
            action = self.expert_model.choose_action(state_tensor, deterministic=True).cpu().numpy()
            action = np.clip(action, -1, 1)[0]
        return action

    def update_training_data(self, num_trajectories_per_batch_collection=20):
        """
        Updates the training data by collecting trajectories from the current policy and the expert policy.

        Args:
            num_trajectories_per_batch_collection: the number of trajectories to collect from the current policy.

        NOTE: you will need to call self.generate_trajectory and self.call_expert_policy in this function.
        NOTE: you should update self.states, self.actions, and self.timesteps in this function.
        """
        # BEGIN STUDENT SOLUTION
        new_states, new_actions, new_timesteps = [], [], []
        for _ in range(num_trajectories_per_batch_collection):
            states, _, timesteps, _, _ = self.generate_trajectory(self.env,self.model,render=False)
            expert_actions = [self.call_expert_policy(state) for state in states]
            new_states.append(states)
            new_actions.append(expert_actions)
            new_timesteps.append(timesteps)

        new_states = np.concatenate(new_states,axis =0)
        new_actions = np.concatenate(new_actions,axis =0)
        new_timesteps = np.concatenate(new_timesteps,axis =0)
        if self.states is None:
            self.states =new_states
            self.actions = new_actions
            self.timesteps = new_timesteps
        else:
            self.states = np.concatenate([self.states, new_states],axis=0)
            self.actions = np.concatenate([self.actions, new_actions],axis=0)
            self.timesteps = np.concatenate([self.timesteps, new_timesteps],axis=0)
        self.clip_sample_range = 1
        self.actions = np.clip(self.actions, -self.clip_sample_range, self.clip_sample_range)

        # END STUDENT SOLUTION
        returns = None
        # return rewards

    def generate_trajectories(self, num_trajectories_per_batch_collection=20):
        """
        Runs inference for a certain number of trajectories. Use for behavior cloning.

        Args:
            num_trajectories_per_batch_collection: the number of trajectories to collect from the current policy.
        
        NOTE: you will need to call self.generate_trajectory in this function.
        """
        # BEGIN STUDENT SOLUTION
        rewards = []
        for _ in range(num_trajectories_per_batch_collection):
            _, _, _, r, _ = self.generate_trajectory(self.env,self.model,render=False)
            rewards.append(sum(r))
        # END STUDENT SOLUTION

        return rewards

    def train(
        self, 
        num_batch_collection_steps=20, 
        num_training_steps_per_batch_collection=1000, 
        num_trajectories_per_batch_collection=20, 
        batch_size=128, 
        print_every=1000, 
        save_every=10000, 
        wandb_logging=False
    ):
        """
        Train the model using BC or DAgger

        Args:
            num_batch_collection_steps: the number of times to collecta batch of trajectories from the current policy.
            num_training_steps_per_batch_collection: the number of times to train the model per batch collection.
            num_trajectories_per_batch_collection: the number of trajectories to collect from the current policy per batch.
            batch_size: the batch size to use for training.
            print_every: how often to print the loss during training.
            save_every: how often to save the model during training.
            wandb_logging: whether to log the training to wandb.

        NOTE: for BC, you will need to call the self.training_step function and self.generate_trajectories function.
        NOTE: for DAgger, you will need to call the self.training_step and self.update_training_data function.
        """

        losses = np.zeros(num_batch_collection_steps * num_training_steps_per_batch_collection)
        self.model.train()
        mean_rewards, median_rewards, max_rewards = [], [], []
        # BEGIN STUDENT SOLUTION
        for batch_collection_step in tqdm(range(num_batch_collection_steps)):  
            if self.mode == "DAgger":
                self.update_training_data(num_trajectories_per_batch_collection=num_trajectories_per_batch_collection)

            rewards = self.generate_trajectories(num_trajectories_per_batch_collection=num_trajectories_per_batch_collection)
            mean_rewards.append(np.mean(rewards))
            median_rewards.append(np.median(rewards))
            max_rewards.append(np.max(rewards))

            for training_step in range(num_training_steps_per_batch_collection):
                global_step = batch_collection_step*num_training_steps_per_batch_collection + training_step

                loss = self.training_step(batch_size=batch_size)
                losses[global_step] = loss
                if (global_step + 1) % print_every == 0:
                    print(f"Step {global_step + 1}, Loss: {loss}, Mean Reward: {mean_rewards[-1]}, Median Reward: {median_rewards[-1]}, Max Reward: {max_rewards[-1]}")
                    
                    if wandb_logging and (wandb is not None):
                        wandb.log({
                            f"{self.mode}_loss":loss,
                            f"{self.mode}_mean":mean_rewards[-1],
                            f"{self.mode}_median":median_rewards[-1],
                            f"{self.mode}_max":max_rewards[-1],
                            "num_states": len(self.states) if self.states is not None else 0
                        }, step=global_step + 1)
                if (global_step +1) % save_every == 0:
                    torch.save(self.model.state_dict(), f"{self.mode}_model_step_{global_step + 1}.pt")
        # END STUDENT SOLUTION
        x_axis = np.arange(0, len(mean_rewards)) * num_training_steps_per_batch_collection
        plt.figure()
        plt.plot(x_axis, mean_rewards, label="mean rewards")
        plt.plot(x_axis, median_rewards, label="median rewards")
        plt.plot(x_axis, max_rewards, label="max rewards")
        plt.legend()
        plt.savefig(f"{self.mode}_rewards.png")

        plt.figure()
        plt.plot(np.arange(0, len(losses)), losses, label="training loss")
        plt.legend()
        plt.savefig(f"{self.mode}_losses.png")

        return losses

    def training_step(self, batch_size):
        """
        Simple training step implementation

        Args:
            batch_size: the batch size to use for training.
        """
        states, actions, timesteps = self.get_training_batch(batch_size=batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        timesteps = timesteps.to(self.device)

        loss_fn = nn.MSELoss()
        self.optimizer.zero_grad()
        predicted_actions = self.model(states, timesteps)
        loss = loss_fn(predicted_actions, actions)
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

    def get_training_batch(self, batch_size=64):
        """
        get a training batch

        Args:
            batch_size: the batch size to use for training.
        """
        # get random states, actions, and timesteps
        indices = np.random.choice(len(self.states), size=batch_size, replace=False)
        states = torch.tensor(self.states[indices], device=self.device).float()
        actions = torch.tensor(self.actions[indices], device=self.device).float()
        timesteps = torch.tensor(self.timesteps[indices], device=self.device)
            
        
        return states, actions, timesteps

def run_training(dagger: bool):
    """
    Simple Run Training Function
    """

    env = gym.make('BipedalWalker-v3', render_mode='rgb_array') # , render_mode="rgb_array"
    with open(f"data/states_BC.pkl", "rb") as f:
        states = pickle.load(f)
    with open(f"data/actions_BC.pkl", "rb") as f:
        actions = pickle.load(f)

    if dagger:
        # Load expert model
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        expert_model = PolicyNet(24, 4)
        model_weights = torch.load(f"data/models/super_expert_PPO_model.pt", map_location=dev)
        expert_model.load_state_dict(model_weights["PolicyNet"])
        # BEGIN STUDENT SOLUTION

       
        student = SimpleNet(24, 4, 128, max_episode_length=1600)
        student.set_device(dev)
        student.to(dev)
        
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=1e-4)
        
        trainer = TrainDaggerBC(
            env=env,
            model=student,
            optimizer=optimizer,
            states=None,
            actions=None,
            expert_model=expert_model,
            device= dev,
            mode="DAgger"
        )
        
        trainer.train(
            num_batch_collection_steps=20,
            num_training_steps_per_batch_collection= 1000,
            num_trajectories_per_batch_collection=20,
            batch_size=128,
            wandb_logging= False
        )
        
        # END STUDENT SOLUTION
        traj_reward = 0
        while traj_reward < 260:
            _, _, _, rewards, rgbs = trainer.generate_trajectory(trainer.env, trainer.model, render=True)
            traj_reward = sum(rewards)
            print(f"got trajectory with reward {traj_reward}")
            imageio.mimsave(f'gifs_{trainer.mode}.gif', rgbs, fps=33)
    else:
        # BEGIN STUDENT SOLUTION
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'

        student =SimpleNet(24, 4, 128, max_episode_length=1600) 
        student.set_device(dev)   
        student.to(dev)

        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=1e-4)

        trainer = TrainDaggerBC(
            env=env,
            model=student,
            optimizer=optimizer,
            states=states,
            actions=actions,
            expert_model=None,
            device= dev,
            mode="BC"
        )

        trainer.train(
            num_batch_collection_steps=20,
            num_training_steps_per_batch_collection =1000,
            num_trajectories_per_batch_collection=20,  
            batch_size =128,
            print_every =1000,
            save_every= 10000,
            wandb_logging=False
        )
        # END STUDENT SOLUTION
        traj_reward = 1
        while traj_reward > 0:
            _, _, _, rewards, rgbs = trainer.generate_trajectory(trainer.env, trainer.model, render=True)
            traj_reward = sum(rewards)
            print(f"got trajectory with reward {traj_reward}")
            imageio.mimsave(f'gifs_{trainer.mode}.gif', rgbs, fps=33)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dagger', action='store_true')
    args = parser.parse_args()
    run_training(args.dagger)
