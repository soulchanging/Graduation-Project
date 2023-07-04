import os
import sys
import argparse
import numpy as np
from env.TwoDots import TwoDots
import time


def create_env():
   
    env_args = type('Config', (object,), {
        'scenario_name': 'simple_mapping',
        'episode_length': 25,
        'num_agents': 2,
        'num_landmarks': 2,
        'use_discrete_action': False,
    })

    env = TwoDots(env_args)
    return env

def generate_data_a(num):
    # all_args = parser.parse_known_args(sys.argv[1:])[0]

    rootpath = os.path.dirname(os.path.abspath(__file__))

    dataset = {
        'observations': [],
        'actions': [],
        'next_observations': [],
        'rewards': [],
        'terminals': [],
        'strategy': [],
    }
    traj = []

    env = create_env()
    episode = 0

    obs = env.reset()
    # env.render()
    while episode < num:
        dataset['observations'].append(np.array(obs))
        action = np.array([[1.0 - np.random.rand() / 10],[ -1.0 + np.random.rand() / 10]])
        next_obs, reward, done, info = env.step(action)
        dataset['actions'].append(np.array(action))
        dataset['next_observations'].append(np.array(next_obs))
        dataset['rewards'].append(np.array(reward))
        dataset['terminals'].append(np.array(done))
        dataset['strategy'].append(np.array([0.0,0.0]))
        obs = next_obs
        # print(next_obs,done)
        if np.all(done):
            episode += 1
            # print(episode)
            traj.append(dataset)
            dataset = {
                'observations': [],
                'actions': [],
                'next_observations': [],
                'rewards': [],
                'terminals': [],
                'strategy': [],
            }
            obs = env.reset()
            # dataset['next_observations'][-1] = np.array(obs)
        # env.render()
        # time.sleep(0.2)
        
    data_path = "data"
    if os.path.exists(data_path):
        pass
    else:
        os.makedirs(data_path)
    np.save(os.path.join(data_path,"data_a.npy"), traj, allow_pickle=True)

def generate_data_b(num):
    # all_args = parser.parse_known_args(sys.argv[1:])[0]

    rootpath = os.path.dirname(os.path.abspath(__file__))
    traj = []
    dataset = {
        'observations': [],
        'actions': [],
        'next_observations': [],
        'rewards': [],
        'terminals': [],
        'strategy': []
    }

    env = create_env()
    episode = 0

    obs = env.reset()
    # env.render()
    while episode < num:
        dataset['observations'].append(np.array(obs))
        action = np.array([[-1.0 + np.random.rand() / 10],[1.0 - np.random.rand() / 10]])
        next_obs, reward, done, info = env.step(action)
        dataset['actions'].append(np.array(action))
        dataset['next_observations'].append(np.array(next_obs))
        dataset['rewards'].append(np.array(reward))
        dataset['terminals'].append(np.array(done))
        dataset['strategy'].append(np.array([10.0,10.0]))
        obs = next_obs
        if np.all(done):
            episode += 1
            traj.append(dataset)
            dataset = {
                'observations': [],
                'actions': [],
                'next_observations': [],
                'rewards': [],
                'terminals': [],
                'strategy': [],
            }
            # print(episode)
            obs = env.reset()
            # dataset['next_observations'][-1] = np.array(obs)
        # env.render()
        # time.sleep(0.2)
        
    data_path = "data"
    if os.path.exists(data_path):
        pass
    else:
        os.makedirs(data_path)
    np.save(os.path.join(data_path,"data_b.npy"), traj, allow_pickle=True)

def generate_data_r(num):
    # all_args = parser.parse_known_args(sys.argv[1:])[0]

    rootpath = os.path.dirname(os.path.abspath(__file__))

    dataset = {
        'observations': [],
        'actions': [],
        'next_observations': [],
        'rewards': [],
        'terminals': [],
    }

    env = create_env()
    episode = 0

    obs = env.reset()
    action_num = 5
    # env.render()
    while episode < num:
        action_num = 0
        while action_num < 5:
            dataset['observations'].append(np.array(obs))
            action = np.array([[2 * (0.5 - np.random.rand())], [2 * (0.5 - np.random.rand())]])
            next_obs, reward, done, info = env.step(action)
            dataset['actions'].append(np.array(action))
            dataset['next_observations'].append(np.array(next_obs))
            dataset['rewards'].append(np.array(reward))
            dataset['terminals'].append(np.array(done))
            obs = next_obs
            if np.all(done):
                obs = env.reset()
            action_num += 1
            # env.render()
            # time.sleep(0.2)
        episode += 1
        obs = env.reset()

        
    data_path = "data"
    if os.path.exists(data_path):
        pass
    else:
        
        os.makedirs(data_path)
    np.save(os.path.join(data_path,"data_r.npy"), dataset, allow_pickle=True)
    # 
# def generate_data_b():
generate_data_a(200)
generate_data_b(200)


data = np.load(os.path.join("data","data_b.npy"),allow_pickle=True)

print(data)
# for traj in data:
#     print(traj["observations"],traj["terminals"])


  