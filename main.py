# @author Dylan Goetting
from math import tau
import torch
from torch._C import device
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import namedtuple, deque
from environment import Environment
from utils import Transition, ReplayMemory, DQN, Distribution, ActorNet, CriticNet
import scipy.stats as d
import matplotlib.pyplot as plt


class NewsvendorDDPGAgent(object):
    """
    A class representing an AI agent that is able to interact with the environment, train its networks, 
    plot its progress and run for a number of episodes. 
    """

    def __init__(self, env, replay_memory, action_range, gamma=0.99, experiment_name='default', 
            buffer_priming_period=1500, tau=0.005, mini_batch_size=128, noise_std=5, eval_t=2000):

        # Hyperparameters
        self.env = env
        self.replay_memory = replay_memory
        self.action_range = action_range
        self.gamma = gamma
        self.experiment_name = experiment_name
        self.buffer_priming_period = buffer_priming_period
        self.tau = tau
        self.mini_batch_size = mini_batch_size
        self.results = []
        self.averages = [[],[],[],[],[],[],[]]
        self.t = 0
        self.noise_std = noise_std
        self.state_size = 7
        self.eval_t = eval_t
        self.evaluate = False

        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")

        # Initializing the neural networks 
        self.critic_net = self.create_critic_network(is_train=True)
        self.critic_net_target = self.create_critic_network(is_train=False)
        self.actor_net = self.create_actor_network(is_train=True)
        self.actor_net_target = self.create_actor_network(is_train=False)
        self.critic_net.to(self.device)
        self.critic_net_target.to(self.device)
        self.actor_net.to(self.device)
        self.actor_net_target.to(self.device)

    def select_action(self, day):
        """
        Select an action given an input from the environment. For the first period, the agent selects 
        an action completely randomly to generate experience. After that period, it begins training its 
        networks and starts selecting to maximize reward
        """
        state = self.day_to_state(day)

        if self.t < self.buffer_priming_period:
            # Random action
            action = random.uniform(self.action_range[0], self.action_range[1])
            action = torch.tensor([[action]], dtype=torch.float, device=self.device)

        else:
            with torch.no_grad():
                # Action chosen by the actor net with some added noise for exploration
                action = self.actor_net(state)
                noise = np.random.normal(0, self.noise_std)
                if not self.evaluate:
                    action += noise
            
        return action

    def create_critic_network(self, is_train=None):
        """
        Creates a neuralnetwork of fully connected layers, taking in the state as well as the action, and 
        outputing the Q value of the resulting state.
        """
        return CriticNet(self.state_size+1, 1)

    def create_actor_network(self, is_train=None):
        """
        Creates a neural network of fully connected layers, outbuting a single tensor representing 
        the chosen action. Takes in the the minimum and maximum possible actions to evenly spread its 
        output over that range.
        """
        return ActorNet(self.state_size, output_size=1, min_out=self.action_range[0], max_out=self.action_range[1])

    def update_target_networks(self, tau):
        """
        Performs a soft update of the target networks, by averaging the previous parameters with the 
        current parameters by a small factor TAU.
        """
        for target_param, param in zip(self.critic_net_target.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data*tau)

        for target_param, param in zip(self.actor_net_target.parameters(), self.actor_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data*tau)
    
    def generate_target_q_values(self, next_state_batch, reward_batch):
        """
        Generate the target Q values from a batch of next states, and the observed rewards
        """
        # Find the actions we would've taken from the next states. Add some noise in as part of the TD3 algorithm
        next_action_batch = self.actor_net_target(next_state_batch) + np.random.normal(0, self.noise_std)
        # Take the minimum of the 'twin' networks learned by the critic, as part of the TD3 algorithm
        q1, q2 = self.critic_net_target(next_state_batch, next_action_batch)
        next_state_action_values = torch.min(q1, q2)
        # Use the Bellman equation to calculate the true targets 
        target_values = reward_batch.unsqueeze(1) + self.gamma * next_state_action_values

        return target_values

    def train_minibatch(self):
        """
        Samples a batch and updates the parameters/networks of the agent according to the sampled batch.
        This means we ...
            1. Compute the targets
            2. Update the Q-function/critic by one step of gradient descent
            3. Update the policy/actor by one step of gradient ascent
            4. Update the target networks through a soft update
        """
        if len(self.replay_memory) < self.mini_batch_size:
            # Don't yet have enough memory experience to train
            return
        
        # Sample a batch from memory and convert to tensors and store on device 
        mini_batch = self.replay_memory.sample(self.mini_batch_size)
        batch = Transition(*zip(*mini_batch))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        state_batch.to(self.device)
        action_batch.to(self.device)
        reward_batch.to(self.device)
        next_state_batch.to(self.device)
        action_batch = action_batch.squeeze(dim=1)

        # Train critic network
        target_q_values = self.generate_target_q_values(next_state_batch, reward_batch)
        self.critic_net.optimizer.zero_grad()
        q1, q2 = self.critic_net(state_batch, action_batch)
        
        # Get proper dimensions of the tensors
        target_q_values = target_q_values.squeeze(dim=1)
        q1 = q1.squeeze(dim=1)
        q2 = q2.squeeze(dim=1)
        
        # Loss here is the sum of the losses of each 'twin' network 
        value_loss = self.critic_net.criterion(q1, target_q_values) + self.critic_net.criterion(q2, target_q_values)
        value_loss.backward()
        self.critic_net.optimizer.step()

        if self.t % 2 == 0:
            # Perform a 'delayed' training of the actor network, as part of TD3
            self.actor_net.optimizer.zero_grad()
            actions_taken = self.actor_net(state_batch)
            # Loss here is the opposite of the value of the actions the actor net would've taken,  
            policy_loss = -self.critic_net.q1(state_batch, actions_taken).mean()
            policy_loss.backward()
            self.actor_net.optimizer.step()
            self.update_target_networks(self.tau)
            del policy_loss

        del action_batch
        del reward_batch
        del state_batch
        del next_state_batch
        del value_loss

    def plot(self):
        """
        Plot the reward the agent has achieved in every episode up to the present 
        """
        plt.xlabel('Episode')
        plt.ylabel('Profit')
        plt.title(f'{self.experiment_name} results:')
        plt.plot(self.results)
        plt.pause(0.01)


    def day_to_state(self, day):
        """
        Convert an int value representing the day to a binary tensor vector for the networks to use
        """
        lst = [0]*7
        lst[day] = 1
        return torch.tensor([lst], dtype=torch.float, device=self.device)

    def run(self, num_episodes):
        """
        Main loop where the agent ineracts with the environemnts, chooses actions and trains its networks. Runs for 
        NUM_EPISODES until stopping.
        """
        for _ in range(num_episodes):
            # At the start of every episode, reset the environment and reward counter.
            self.env.reset()
            state = self.env.observe()
            total_reward = 0

            while True:
                self.t += 1
                # Select an action based on the state, interact with the environment, and observe the results
                action = self.select_action(state)
                reward, next_state, done = env.step(action)
                if self.t % 226 == 0:
                    # Log for debugging 
                    print("t:", self.t, "state: ", state, " action: ", action, " reward: ", reward)
                total_reward += reward
                
                # Some linear algebra to make the tensors in memory match what the networks expect 
                mem_state = self.day_to_state(state)
                mem_action = action.unsqueeze(dim=0)
                mem_next_state = self.day_to_state(next_state)
                mem_reward = torch.tensor([(reward/100)**3], dtype=torch.float, device = self.device)
                
                # Adding transition to memory 
                memory.push(mem_state, mem_action, mem_next_state, mem_reward)
                
                if self.t == self.buffer_priming_period:
                    print("BEGIN TRAINING")

                if self.t > self.buffer_priming_period and not self.evaluate:
                    self.train_minibatch()

                if self.t == self.eval_t:
                    print("BEGIN EVALUATING")
                    self.evaluate = True

                if self.evaluate:
                    self.averages[state].append(action.data)
                
                # Advance to next state
                state = next_state
                
                if done:
                    self.results.append(total_reward/(self.env.numWeeks*7))
                    self.plot()
                    break

# Create 7 distributions to use for the environment 
lst = [Distribution(d.norm, 50, 5)]
lst.append(Distribution(d.norm, 50, 10))
lst.append(Distribution(d.norm, 50, 20))
lst.append(Distribution(d.expon, 30, 5))
lst.append(Distribution(d.uniform, 70, 0))
lst.append(Distribution(d.uniform, 80, 0))
lst.append(Distribution(d.uniform, 90, 0))

env = Environment(lst, 10, 5, 30, 100)
memory = ReplayMemory(100000)
agent = NewsvendorDDPGAgent(env, memory, [0, 100], buffer_priming_period=25000, tau=0.001, mini_batch_size=128, 
eval_t=35000, noise_std=3, experiment_name="Exp0")

plt.figure(1)
# Run the agent for 300 episodes 
agent.run(300)

# Make a final bar plot showing the average actions the agent chose for each state 
plt.figure(2)
plt.title("Average actions chosen")
plt.bar(range(7), [sum(a)/len(a) for a in agent.averages])
plt.xlabel("State")
plt.ylabel("Action")
plt.show()
print("Complete")