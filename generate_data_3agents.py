import os
import sys
import argparse
import numpy as np
from env.TwoDots import TwoDots
import time


def create_env():
   
    env_args = type('Config', (object,), {
        'scenario_name': 'three_mapping',
        'episode_length': 25,
        'num_agents': 3,
        'num_landmarks': 3,
        'use_discrete_action': False,
    })

    env = TwoDots(env_args)
    return env

def generate_data(num,f,dir):
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
    length = 0
    while episode < num:
        dataset['observations'].append(np.array(obs))
        action = f()
        next_obs, reward, done, info = env.step(action)
        dataset['actions'].append(np.array(action))
        dataset['next_observations'].append(np.array(next_obs))
        dataset['rewards'].append(np.array(reward))
        dataset['terminals'].append(np.array(done))
        dataset['strategy'].append(np.array([0.0,0.0]))
        obs = next_obs
        length += 1
        # print(next_obs,done)
        if np.all(done) or length>4:
            length = 0
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
        #     # dataset['next_observations'][-1] = np.array(obs)
        # env.render()
        # time.sleep(0.05)
        
    data_path = "data"
    if os.path.exists(data_path):
        pass
    else:
        os.makedirs(data_path)
    np.save(os.path.join(data_path,dir), traj, allow_pickle=True)

# generate random data
def f0():
    return np.array([[2 * (0.5 - np.random.rand()), 2 * (0.5 - np.random.rand())],
                    [2 * (0.5 - np.random.rand()), 2 * (0.5 - np.random.rand())],
                    [2 * (0.5 - np.random.rand()), 2 * (0.5 - np.random.rand())]])

def f1():
    return np.array([[-1.0 + np.random.rand() / 10,-1.0 + np.random.rand() / 10],
                     [1.0 - np.random.rand() / 10,1.0 - np.random.rand() / 10],
                     [-1.0 + np.random.rand() / 10,1.0 - np.random.rand() / 10]])

def f2():
    return np.array([[1.0 - np.random.rand() / 10,1.0 - np.random.rand() / 10],
                     [-1.0 + np.random.rand() / 10,1.0 - np.random.rand() / 10],
                     [-1.0 + np.random.rand() / 10,-1.0 + np.random.rand() / 10]])

def f3():
    return np.array([[-1.0 + np.random.rand() / 10,1.0 - np.random.rand() / 10],
                     [-1.0 + np.random.rand() / 10,-1.0 + np.random.rand() / 10],
                     [1.0 - np.random.rand() / 10,1.0 - np.random.rand() / 10]])


def f4():
    return np.array([[-1.0 + np.random.rand() / 10,1.0 - np.random.rand() / 10],
                     [1.0 - np.random.rand() / 10,1.0 - np.random.rand() / 10],
                     [-1.0 + np.random.rand() / 10,-1.0 + np.random.rand() / 10]])
    
def f5():
    return np.array([[-1.0 + np.random.rand() / 10,-1.0 + np.random.rand() / 10],
                     [-1.0 + np.random.rand() / 10,1.0 - np.random.rand() / 10],
                     [1.0 - np.random.rand() / 10,1.0 - np.random.rand() / 10]])
    
def f6():
    return np.array([[1.0 - np.random.rand() / 10,1.0 - np.random.rand() / 10],
                     [-1.0 + np.random.rand() / 10,-1.0 + np.random.rand() / 10],
                     [-1.0 + np.random.rand() / 10,1.0 - np.random.rand() / 10]])

generate_data(2000,f0,'data_0.npy')
generate_data(400,f1,'data_1.npy')
generate_data(400,f2,'data_2.npy')
generate_data(400,f3,'data_3.npy')
generate_data(400,f4,'data_4.npy')
generate_data(400,f5,'data_5.npy')
generate_data(400,f6,'data_6.npy')

data = np.load(os.path.join("data","data_1.npy"),allow_pickle=True)
print(data)
data = np.load(os.path.join("data","data_2.npy"),allow_pickle=True)
print(len(data))
data = np.load(os.path.join("data","data_3.npy"),allow_pickle=True)
print(len(data))
data = np.load(os.path.join("data","data_4.npy"),allow_pickle=True)
print(len(data))
data = np.load(os.path.join("data","data_5.npy"),allow_pickle=True)
print(len(data))
data = np.load(os.path.join("data","data_6.npy"),allow_pickle=True)
print(len(data))

for traj in data:
    print(traj["rewards"],traj["terminals"])


  