# -*- coding: utf-8 -*-
"""
Advanced Machine Learning Lab
MLDM M2

Implimentation of multi-Arm Bandit Algorithm

Austin Schwinn
Jeremie Blanchard
Oussama Bouldjedri

"""

import numpy as np
import matplotlib.pyplot as plt



class bandit:
    '''
    Upper Confidence Bound k-bandit problem
    
    Inputs 
    ============================================
    k: number of arms (int)
    c:
    iters: number of steps (int)
    mu: set the average rewards for each of the k-arms.
        Set to "random" for the rewards to be selected from
        a normal distribution with mean = 0. 
        Set to "sequence" for the means to be ordered from 
        0 to k-1.
        Pass a list or array of length = k for user-defined
        values.
    '''
    def __init__(self, c, iters, mu='random'):
        # Exploration parameter
        self.c = c
        # Number of iterations
        self.iters = iters
        # Step count
        self.n = 1
        # Step count for each arm
        self.k_n = np.ones(10)
        # Total mean reward
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        # Mean reward for each arm
        self.k_reward = np.zeros(10)
        
        if type(mu) == list or type(mu).__module__ == np.__name__:
            # User-defined averages            
            self.mu = np.array(mu)
        elif mu == 'random':
            # Draw means from probability distribution
            self.mu = np.random.normal(0, 1, 10)
            print(self.mu)
        elif mu == 'sequence':
            # Increase the mean for each arm by one
            self.mu = np.linspace(0, 9, 10)
        
    def pull_ucb(self):
        # Select action according to UCB Criteria
        a = np.argmax(self.k_reward + self.c * np.sqrt(
                (np.log(self.n)) / self.k_n))
            
        reward = np.random.normal(self.mu[a], 1)
        # Update counts
        self.n += 1
        self.k_n[a] += 1
        
        # Update total
        self.mean_reward = self.mean_reward + (
            reward - self.mean_reward) / self.n
        
        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (
            reward - self.k_reward[a]) / self.k_n[a]
        
    def pull_incremental(self):
        # Select action according to incremental Uniform criteria
        place = np.argmax(self.k_reward)
        #print("place= ",place)
        reward = np.random.normal(self.mu[place], 1)
        # Update counts
        self.n += 1
        self.k_n[place] += 1
        
        # Update total
        self.mean_reward = self.mean_reward + (
            reward - self.mean_reward) / self.n
        
        # Update results for a_k
        self.k_reward[place] = self.k_reward[place] + (
            reward - self.k_reward[place]) / self.k_n[place]

    def pull_epsi(self):
        # Select action according to Epsi greedy criteria
        place = np.argmax(self.k_reward)
        
        if np.random.randint(0,100) > 95:
            place = np.random.randint(0,10)
            
        reward = np.random.normal(self.mu[place], 1) 
        # Update counts
        self.n += 1
        self.k_n[place] += 1
        
        # Update total
        self.mean_reward = self.mean_reward + (
            reward - self.mean_reward) / self.n
        
        # Update results for a_k
        self.k_reward[place] = self.k_reward[place] + (
            reward - self.k_reward[place]) / self.k_n[place]
        
    def run_incremental(self):
        for i in range(self.iters):
            global reward_tot
            reward_tot = 0
            self.pull_incremental()
            self.reward[i] = self.mean_reward
            #print("Reward total = ",reward_tot)
            
    def run_ucb(self):
        for i in range(self.iters):
            global reward_tot
            reward_tot = 0
            self.pull_ucb()
            self.reward[i] = self.mean_reward
            #print("Reward total = ",reward_tot)
            
    def run_epsi(self):
        for i in range(self.iters):
            global reward_tot
            reward_tot = 0
            self.pull_epsi()
            self.reward[i] = self.mean_reward
            #print("Reward total = ",reward_tot)
            
    def reset(self, mu=None):
        # Resets results while keeping settings
        self.n = 1
        self.k_n = np.ones(10)
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        self.k_reward = np.zeros(10)
        if mu == 'random':
            self.mu = np.random.normal(0, 1, 10)
                        
#%%
iters = 1000

ucb_rewards = np.zeros(iters)
incremental_rewards = np.zeros(iters)
epsi_rewards = np.zeros(iters)
# Initialize bandits
bandit_ucb = bandit(2, iters)
bandit_incre = bandit(2, iters)
bandit_epsi = bandit(2, iters)

episodes = 1000
# Run experiments
for i in range(episodes): 
    bandit_incre.reset('random')
    bandit_ucb.reset('random')
    bandit_epsi.reset('random')
    # Run experiments
    bandit_incre.run_incremental()
    bandit_ucb.run_ucb()
    bandit_epsi.run_epsi()
    
    # Update long-term averages
    incremental_rewards = incremental_rewards + (bandit_incre.reward - incremental_rewards) / (i + 1)
    ucb_rewards = ucb_rewards + (bandit_ucb.reward - ucb_rewards) / (i + 1)
    epsi_rewards = epsi_rewards + (bandit_epsi.reward - epsi_rewards) / (i + 1)
    
plt.figure(figsize=(10,8))
plt.plot(incremental_rewards, label="Incremental")
plt.plot(ucb_rewards, label="UCB")
plt.plot(epsi_rewards, label="epsi")
plt.legend(bbox_to_anchor=(1.2, 0.5))
plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.title("Average Rewards after " 
          + str(episodes) + " Episodes")
plt.show()