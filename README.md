# omar with skill extraction

该部分代码仅对toy example进行了测试：三个智能体合作覆盖三个固定地标。

面向多种合作策略混杂的数据集，使用CVAE对数据集将数据编码到隐藏空间，然后使用dbscan对隐藏变量聚类，得到离散的合作策略。为原始数据的添加其对应的合作策略种类，作为其隐藏状态。
得到processed_data,使用omar进行训练。在执行动作时，隐藏状态skill从聚类得到的离散集合中选取，最终智能体将会按照所选取的合作策略执行动作。

实验证明，添加策略提取后，智能体能够有效利用混杂数据集训练正确的合作策略。

注意：训练时需要加上data_0(随机样本)作为负样本。

## Citing

If you used this code in your research or found it helpful, please consider citing our paper:
```
@inproceedings{pan2021regularized,
  title={Plan Better Amid Conservatism: Offline Multi-Agent Reinforcement Learning with Actor Rectification},
  author={Pan, Ling and Huang, Longbo and Ma, Tengyu and Xu, Huazhe},
  booktitle={International Conference on Machine
Learning},
  year={2022}
}
```

## Requirements

- Multi-agent Particle Environments: in envs/multiagent-particle-envs and install it by `pip install -e .`
- python: 3.6
- torch
- baselines (https://github.com/openai/baselines)
- seaborn
- gym==0.9.4
- Multi-Agent MuJoCo: Please check the [multiagent_mujoco](https://github.com/schroederdewitt/multiagent_mujoco) repo for more details about the environment. Note that this depends on gym with version 0.10.5.



Note: The datasets are too large, and the Baidu (Chinese) online disk requires a password for accessing it. Please just enter the password in the input box and click the blue button. The dataset can then be downloaded by cliking the "download" button (the second white button).

## Usage

Please follow the instructions below to replicate the results in the paper. 

```
pythonmain.py --env_id <ENVIRONMENT_NAME> --data_type <DATA_TYPE> --seed <SEED> --omar 1
```

- env_id: simple_spread/tag/world/HalfCheetah-v2
- data_type: random/medium-replay/medium/expert

