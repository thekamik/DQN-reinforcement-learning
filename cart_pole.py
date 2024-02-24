import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count

import gymnasium as gym

# Neural network model
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)

    def forward(self, x): 
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# RL Agent class
class Agent:
    def __init__(self, n_observations, n_actions):

        # Select available device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # optimizer learning rate
        self.learning_rate = 1e-4

        # discount rate
        self.gamma = 0.99

        # size of learning buffer
        self.batch_size = 128

        # Update rate of the target network
        self.tau = 0.005

        # probability of choosing random move
        self.epsilon = 0.9
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.999

        # create memory deque 
        self.memory_size = 100000

        # Transition to store previous responses
        self.action_memory = np.zeros((self.memory_size, 1))
        self.state_memory = np.zeros((self.memory_size, n_observations))
        self.next_state_memory = np.zeros((self.memory_size, n_observations))
        self.reward_memory = np.zeros((self.memory_size, 1))
        self.terminated_memory = np.zeros((self.memory_size, 1))

        # current memory index
        self.memory_index = 0

        # q function is replaced by neural network
        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # define optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)

        # List with scores
        self.episode_reward = []
        self.episodes_means = []

    def add_to_memory(self, action, state, next_state, reward, terminated):

        # Get actual data index
        index = self.memory_index % self.memory_size

        # Store data into memory
        self.action_memory[index] = action
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.terminated_memory[index] = not terminated

        # increase memory index
        self.memory_index += 1

    def get_sample(self):

        if self.memory_index < self.memory_size:
            # Take sample only from used memory
            sample = np.random.randint(0, self.memory_index, size=self.batch_size)
        else:
            # All memory is used, new values are placed at beginning
            sample = np.random.randint(0, self.memory_size, size=self.batch_size)
        
        # Return samples from memory as arrays
        return self.action_memory[sample], self.state_memory[sample], self.next_state_memory[sample], self.reward_memory[sample], self.terminated_memory[sample]


    def select_action(self, state, actions):

        # epsilon decay strategy
        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)

        # choose action
        if np.random.random() <= self.epsilon:
            return np.random.choice(actions)
        else:
            # Convert state into tensor
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Select best possible move using neural network.
            output = torch.squeeze(self.policy_net(state), dim=0)

            # Select best action index
            action = torch.argmax(output).item()

            return action


    def train_model(self, action, state, next_state, reward, terminated):

        # Add state to memory        
        self.add_to_memory(action, state, next_state, reward, terminated)

        # Execute training
        if self.memory_index > self.batch_size:

            # Get sample from memory
            action_batch, state_batch, next_state_batch, reward_batch, terminated_batch = self.get_sample()
            
            # Find indexes of non final states
            non_final_indexes = np.where(terminated_batch == True)[0]

            # Convert arrays into tensors
            action_batch = torch.tensor(action_batch, device=self.device, dtype=torch.long)
            state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float32)
            next_state_batch = torch.tensor(next_state_batch[non_final_indexes, :], device=self.device, dtype=torch.float32)
            reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float32).squeeze()
            terminated_batch = torch.tensor(terminated_batch, device=self.device, dtype=torch.bool).squeeze()


            # Get output from the policy and pics out specific elements from output according to action batch
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)

            # Create tensor to store values from target network
            next_state_values = torch.zeros(self.batch_size, device=self.device)

            # Use only forward - do not optimize target network
            with torch.no_grad():
                # Use target network to calculate expected best action in next state.
                next_state_values[terminated_batch] = self.target_net(next_state_batch).max(1).values
            
            # Compute the expected Q values, using Bellman equation
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch

            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()

            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.optimizer.step()

        # Update target network weights
        # Q′ = tQ + (1 −t )Q′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

        # Return observation - new state
        return observation


    def plot_durations(self, reward, show_result=False):

        # Add game length to array
        self.episode_reward.append(reward)

        # Convert episode_reward into np.arrays
        durations = np.array(self.episode_reward)

        # Convert means into array
        if len(durations) < 100:
            self.episodes_means.append(0)
        else:
            # Add last 100 episodes mean to list
            self.episodes_means.append(durations[-100:].mean())

        # Convert episodes_means into np.array
        means = np.array(self.episodes_means)

        # plot means and durrations arrays
        plt.figure(1)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Episode Duration')
        plt.plot(durations)
        plt.plot(means)

        # Give some time to render chart
        plt.pause(0.001)


if __name__ == "__main__":
    # env = gym.make("Blackjack-v1", natural=False, sab=False, render_mode="human")
    env = gym.make("CartPole-v1", render_mode="human")

    # enable dinamic update
    plt.ion()

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)

    # Create Agent
    agent = Agent(n_observations=n_observations, n_actions=n_actions)

    # Select number of episodes
    num_episodes = 500

    for i_episode in range(num_episodes):
        # Initialize the environment
        state, info = env.reset()

        for t in count():

            # Get all possible actions
            actions = np.arange(env.action_space.n)

            # model select best action
            action = agent.select_action(state, actions)

            # execute action
            observation, reward, terminated, truncated, _ = env.step(action)

            # train model
            state = agent.train_model(action, state, observation, reward, terminated)

            # render enviroment
            env.render()
        
            # check final condition
            if terminated or truncated:
                agent.plot_durations(t+1)
                break

    print('Complete')
    agent.plot_durations(t+1, show_result=True)
    plt.ioff()
    plt.show()
