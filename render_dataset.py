import os
import sys
import argparse
import numpy as np
from env.MPE_env import MPEEnv


def create_env(datapath):
   
    env_args = type('Config', (object,), {
        'scenario_name': 'simple_mapping',
        'episode_length': 25,
        'num_agents': 2,
        'num_landmarks': 2,
        'use_discrete_action': False,
    })

    env = MPEEnv(env_args)
    return env


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_path", type=str, default='/data/simple_mapping/dataset-0.npy')
    parser.add_argument("--save_path", type=str, default='')
    parser.add_argument("--render", default=True, action='store_true')
    all_args = parser.parse_known_args(sys.argv[1:])[0]

    DATAPATH = all_args.data_path

    rootpath = os.path.dirname(os.path.abspath(__file__))
    dataset = np.load(rootpath + DATAPATH, allow_pickle=True).item()    # type: dict
    if 'states' in dataset:
        dataset.pop('states')
    if 'next_states' in dataset:
        dataset.pop('next_states')

    env = create_env(DATAPATH)
    env.reset()
    for i in range(len(env.agents)):
        env.agents[i].state.p_pos = np.array(dataset['observations'][0][i, 2:4]).copy()
        env.agents[i].init_p_pos = np.array(dataset['observations'][0][i, 2:4]).copy()
        env.agents[i].state.p_vel = np.array(dataset['observations'][0][i, 0:2]).copy()
        env.agents[i].state.c = np.zeros(env.world.dim_c)
    _obs = np.array([env._get_obs(agent) for agent in env.agents])
    if all_args.render:
        env.render()
    episode, ep_step, ep_reward = 0, 0, 0
    ep_reward_list = {}

    for idx, obs, action, next_obs, reward, done in zip(range(len(dataset['observations'])), dataset['observations'], dataset['actions'], dataset['next_observations'], dataset['rewards'], dataset['terminals']):

        _obs, _reward, _done, _info = env.step(action)   # reproduce in environment
        ep_step += 1
        if all_args.save_path:
            ep_reward += np.mean(_reward)
            dataset['rewards'][idx] = np.mean(_reward)
        else:
            ep_reward += np.mean(reward)
        if all_args.render:
            env.render()

        if np.all(done):

            if 'random' not in DATAPATH:
                datatype = np.sum(np.array(_obs)[:, 2] > 0)
                if datatype == 1:
                    cooperate_mode = np.where(np.array(_obs)[:, 2] > 0)[0].item()
                    datatype = f"{datatype}({cooperate_mode})"
                elif datatype == 2:
                    cooperate_mode = np.where(np.array(_obs)[:, 2] <= 0)[0].item()
                    datatype = f"{datatype}({cooperate_mode})"
                else:
                    datatype = f"{datatype}"
            else:
                datatype = 'random'
            if datatype not in ep_reward_list:
                ep_reward_list[datatype] = [ep_reward]
            else:
                ep_reward_list[datatype].append(ep_reward)
            print(f"type[{datatype}] {episode}|{ep_step}: {ep_reward}")
            env.reset()
            for i in range(len(env.action_space)):
                env.agents[i].state.p_pos = np.array(next_obs[i, 2:4]).copy()
                env.agents[i].init_p_pos = np.array(next_obs[i, 2:4]).copy()
                env.agents[i].state.p_vel = np.array(next_obs[i, 0:2]).copy()
                env.agents[i].state.c = np.zeros(env.world.dim_c)
            _obs = np.array([env._get_obs(agent) for agent in env.agents])
            if all_args.render:
                env.render()
            episode += 1
            ep_step, ep_reward = 0, 0

        # reproduction consistency
        assert np.all(_obs == next_obs)
        assert np.all(np.mean(_reward) - dataset['rewards'][idx] < 1e-10), f"{np.mean(_reward)}, {reward}"
        assert np.all(_done == done)

    print(f">>>>>>>>>>> Summary for {DATAPATH} | episode_num = {episode}")
    for datatype, ep_r_lst in ep_reward_list.items():
        print(f"type[{datatype}] count={len(ep_r_lst)} reward: mean={np.mean(ep_r_lst):.2f} |"
              f"max={np.max(ep_r_lst):.2f} | min={np.min(ep_r_lst):.2f} | std={np.std(ep_r_lst):.2f}")

    for key in dataset.keys():
        dataset[key] = np.array(dataset[key])
        if dataset[key].ndim == 1:
            dataset[key] = np.expand_dims(dataset[key], axis=-1)
        print(key, dataset[key].shape)
    if all_args.save_path:
        np.save(all_args.save_path, dataset, allow_pickle=True)
