import numpy as np
from torch import Tensor
from torch.autograd import Variable
import torch
import math

class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(
            self, max_steps, num_agents,obs_dims, ac_dims, is_mamujoco=False, state_dims=None,latent_dim = 2
    ):
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        self.stra_buffs = []
        self.skill_buffs = []
        for odim, adim in zip(obs_dims, ac_dims):
            self.obs_buffs.append(np.zeros((max_steps, odim)))
            self.ac_buffs.append(np.zeros((max_steps, adim)))
            self.rew_buffs.append(np.zeros(max_steps))
            self.next_obs_buffs.append(np.zeros((max_steps, odim)))
            self.done_buffs.append(np.zeros(max_steps))
            self.skill_buffs.append(np.zeros((max_steps,latent_dim)))

        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)
        self.traj_idx = [] # indexs of all traj 
    
    def __len__(self):
        return self.filled_i

    def load_batch_data(self,dirs):
        
        # dataset_1 = np.load(dir+'/data_a.npy',allow_pickle=True)
        # dataset_2 = np.load(dir+'/data_b.npy',allow_pickle=True)
        # # print(len(dataset_1))
        # # all_data = [dataset_2]
        # all_data = [dataset_1 ,dataset_2]
        all_data = [np.load(dir,allow_pickle=True) for dir in dirs]
        for dataset in all_data:
            for traj in dataset:
                # print(traj)
                self.traj_idx.append(self.filled_i)
                num_experiences = len(traj["observations"])
                for i in range(self.num_agents):
                    self.obs_buffs[i][self.filled_i:self.filled_i+num_experiences] = np.array(traj["observations"])[:,i,:]
                    self.ac_buffs[i][self.filled_i:self.filled_i+num_experiences] = np.array(traj["actions"])[:,i,:]
                    self.rew_buffs[i][self.filled_i:self.filled_i+num_experiences] = np.array(traj["rewards"])[:,i]
                    self.next_obs_buffs[i][self.filled_i:self.filled_i+num_experiences] = np.array(traj["next_observations"])[:,i,:]
                    self.done_buffs[i][self.filled_i:self.filled_i+num_experiences] = np.array(traj["terminals"])[:,i]
                    self.skill_buffs[i][self.filled_i:self.filled_i+num_experiences] = np.array(traj["skill"])[:,i,:]
                self.filled_i += num_experiences
                self.curr_i = 0 if self.curr_i == self.max_steps else num_experiences
            self.traj_idx.append(self.filled_i)
        # print(self.ac_buffs)

        
    def sample(self, N, to_gpu=False):
        
        inds = np.random.choice(np.arange(self.filled_i), size=N, replace=False)
        # print(self.ac_buffs)
        # print(inds)
        # print(self.rew_buffs[i][inds])
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
        return (
            [cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
            [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
            ret_rews,
            [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
            [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)],
            [cast(self.skill_buffs[i][inds]) for i in range(self.num_agents)]
        )
        
    def sample_traj(self,N):
        # N is num of batches
        indices = []
        while len(indices)<N:
            start = np.random.choice(self.traj_idx[:-1])
            pos_idx = self.traj_idx.index(start)
            indices += list(range(start,self.traj_idx[pos_idx+1]))
        indices = indices[:N]
        return self.sample(indices)
            
        
           
    # def __init__(self, max_steps, num_agents, obs_dims, ac_dims, is_mamujoco=False, state_dims=None):
    #     """
    #     Inputs:
    #         max_steps (int): Maximum number of timepoints to store in buffer
    #         num_agents (int): Number of agents in environment
    #         obs_dims (list of ints): number of obervation dimensions for each
    #                                  agent
    #         ac_dims (list of ints): number of action dimensions for each agent
    #     """
    #     self.max_steps = max_steps
    #     self.num_agents = num_agents
    #     self.obs_buffs = []
    #     self.ac_buffs = []
    #     self.rew_buffs = []
    #     self.next_obs_buffs = []
    #     self.done_buffs = []
    #     for odim, adim in zip(obs_dims, ac_dims):
    #         self.obs_buffs.append(np.zeros((max_steps, odim)))
    #         self.ac_buffs.append(np.zeros((max_steps, adim)))
    #         self.rew_buffs.append(np.zeros(max_steps))
    #         self.next_obs_buffs.append(np.zeros((max_steps, odim)))
    #         self.done_buffs.append(np.zeros(max_steps))

    #     self.is_mamujoco = is_mamujoco
    #     if self.is_mamujoco:
    #         self.state_buffs = []
    #         self.next_state_buffs = []
    #         for sdim in state_dims:
    #             self.state_buffs.append(np.zeros((max_steps, sdim)))
    #             self.next_state_buffs.append(np.zeros((max_steps, sdim)))

    #     self.filled_i = 0  # index of first empty location in buffer (last index when full)
    #     self.curr_i = 0  # current index to write to (ovewrite oldest data)

    # def __len__(self):
    #     return self.filled_i

    # def sample(self, N, to_gpu=False):
    #     inds = np.random.choice(np.arange(self.filled_i), size=N, replace=False)
    #     if to_gpu:
    #         cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
    #     else:
    #         cast = lambda x: Variable(Tensor(x), requires_grad=False)
    #     ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
    #     if self.is_mamujoco:
    #         return (
    #             [cast(self.state_buffs[i][inds]) for i in range(self.num_agents)],
    #             [cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
    #             [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
    #             ret_rews,
    #             [cast(self.next_state_buffs[i][inds]) for i in range(self.num_agents)],
    #             [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
    #             [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)]
    #         )
    #     else:
    #         return (
    #             [cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
    #             [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
    #             ret_rews,
    #             [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
    #             [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)]
    #         )

    # def load_batch_data(self, dir):
    #     print ('\033[1;33mloading batch data from {}...\033[1;0m'.format(dir))
    #     all_min_rews = []
    #     for i in range(self.num_agents):
    #         curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))
    #         curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))
    #         curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))
    #         curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))
    #         curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))
        
    #         num_experiences = curr_obs.shape[0]

    #         self.obs_buffs[i][:num_experiences] = curr_obs
    #         self.ac_buffs[i][:num_experiences] = curr_acs
    #         self.rew_buffs[i][:num_experiences] = curr_rews
    #         self.next_obs_buffs[i][:num_experiences] = curr_next_obs
    #         self.done_buffs[i][:num_experiences] = curr_dones

    #         if self.is_mamujoco:
    #             curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
    #             curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
    #             self.state_buffs[i][:num_experiences] = curr_states
    #             self.next_state_buffs[i][:num_experiences] = curr_next_states

    #     self.filled_i = num_experiences
    #     self.curr_i = 0 if self.curr_i == self.max_steps else num_experiences
