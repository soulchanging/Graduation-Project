import torch
import torchvision
from torch import nn
from torch import optim
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
from env.TwoDots import TwoDots
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import random

def create_env():
   
    env_args = type('self.config', (object,), {
        'scenario_name': 'simple_mapping',
        'episode_length': 25,
        'num_agents': 2,
        'num_landmarks': 2,
        'use_discrete_action': False,
    })

    env = TwoDots(env_args)
    return env

# 图像转为二维可视化
def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x
 
class VAE(nn.Module):
    def __init__(self,input_num,output_num):
        super(VAE, self).__init__()
        INOUT_num = 784                                          # 输入（输出）大小
        hidden_num1 = 400                                        # 隐藏层大小
        hidden_num2 = output_num                                         # 隐变量大小
        self.fc1 = nn.Linear(input_num, hidden_num1)             # （编码） 全连接层
        self.fc21 = nn.Linear(hidden_num1, hidden_num2)          # （编码） 计算 mean
        self.fc22 = nn.Linear(hidden_num1, hidden_num2)          # （编码） 计算 logvar
        self.fc3 = nn.Linear(hidden_num2, hidden_num1)           # （解码） 隐藏层
        self.fc4 = nn.Linear(hidden_num1, input_num)    
     
    def encode(self, x):
        # 全连接层
        hidden1 = self.fc1(x)
        # relu层
        h1 = F.relu(hidden1)
        # 计算mean
        mu = self.fc21(h1)
        # 计算var
        logvar = self.fc22(h1)
        return mu, logvar
 
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()                            # mul是乘法的意思，然后exp_是求e的次方并修改原数值  所有带"—"都是inplace的 意思就是操作后 原数也会改动
 
        # if torch.cuda.is_available():
        #     eps = torch.cuda.FloatTensor(std.size()).normal_()  # 在cuda中生成一个std.size()的张量，标准正态分布采样，类型为FloatTensor
        # else:
        eps = torch.FloatTensor(std.size()).normal_()       # 生成一个std.size()的张量，正态分布，类型为FloatTensor
        eps = Variable(eps)                                     # Variable是torch.autograd中很重要的类。它用来包装Tensor，将Tensor转换为Variable之后，可以装载梯度信息。
        repar = eps.mul(std).add_(mu)
        return repar
 
    def decode(self, z):
        # 隐藏层
        hidden2 = self.fc3(z)
        # relu层
        h3 = F.relu(hidden2)
        # 隐藏层
        hidden3 = self.fc4(h3)
        # sigmoid层
        output = F.sigmoid(hidden3)
        return output
 
    def forward(self, x):
        mu, logvar = self.encode(x)           # 编码
        z = self.reparametrize(mu, logvar)    # 重新参数化成正态分布
        decodez = self.decode(z)              # 解码
        return z,mu,logvar


class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu, constrain_out=False, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        self.in_fn = nn.BatchNorm1d(input_dim)
        self.in_fn.weight.data.fill_(1)
        self.in_fn.bias.data.fill_(0)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = torch.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))

        return out

class CLUSTER():
    def __init__(self):
        # dim of latent space (z)
        
        self.latent_dim = 2
        # self.traj_dim = 44
        # self.agent_num = 2
        # self.state_dim = 4
        
        self.traj_dim = 66
        self.agent_num = 3
        self.state_dim = 6
        
        self.action_dim = 2
        self.beta = 0.0001
 
    
    
    
    
    
    def train(self):
        # q(z|tau)
        self.state_encoder = VAE(input_num = self.traj_dim, output_num = self.latent_dim)
        # p1(a1|s,z)
        self.action_encoders = [MLPNetwork(input_dim = self.state_dim + self.latent_dim , out_dim = self.action_dim) for i in range(self.agent_num)]        

        state_optimizer = optim.Adam(self.state_encoder.parameters(),lr = 0.001)
        action_optimzers = [optim.Adam(action_encoder.parameters(),lr = 0.001) for action_encoder in self.action_encoders]
        
        dataset1 = np.load('data/data_1.npy',allow_pickle = True)
        dataset2 = np.load('data/data_2.npy',allow_pickle = True)
        dataset3 = np.load('data/data_3.npy',allow_pickle = True)
        dataset4 = np.load('data/data_4.npy',allow_pickle = True)
        dataset5 = np.load('data/data_5.npy',allow_pickle = True)
        dataset6 = np.load('data/data_6.npy',allow_pickle = True)   
             
        # display_data = [dataset1,dataset2]
        display_data = [dataset1,dataset2,dataset3,dataset4,dataset5,dataset6]
        all_data = np.append(dataset1, dataset2)
        all_data = np.append(all_data, dataset3)
        all_data = np.append(all_data, dataset4)
        all_data = np.append(all_data, dataset5)
        all_data = np.append(all_data, dataset6)
        
        random.shuffle(all_data)
        
        self.traj_num = 0
        self.traj_num = len(all_data)
        print(self.traj_num)
    
        for t in range(2000):
            loss_mean = 0
            state_optimizer.zero_grad()
            for action_optimzer in action_optimzers:
                action_optimzer.zero_grad()
            # # sample n trajs
            # trajs = self.train_buffer.sample_traj(N)
            print(t,"th round........................")
            z_list = []
        
            for traj in all_data:
                
                state = Tensor(np.array(traj["observations"])[:,0,:])
                action0 = Tensor(np.array(traj["actions"])[:,0,:])
                action1 =Tensor(np.array(traj["actions"])[:,1,:])
                action2 =Tensor(np.array(traj["actions"])[:,2,:])
                a_tensor_list = [action0,action1,action2]
                # flatten the traj
                s_t = Tensor(np.array(traj["next_observations"])[-1,0,:])
                batch = torch.cat((state,action0,action1,action2),dim = 1)
                traj = torch.cat((batch.view(-1),s_t),-1)
                # print(traj)
                
                z,mu,logvar = self.state_encoder(traj)
                # print(z)
                z_num = z.detach().numpy().tolist()
                z_list.append(z_num)
                # print(z,mu,logvar)
                # todo:搞清系数正负
                kl_loss = -self.beta * self.compute_kl_div(mu, logvar)
                # # todo:搞成tensor
                # s_tensor,a_tensor_list = extract_data(traj)
                # todo2:z搞成与s相同大小的tensor              
                b,_ = state.size()
                z_tensor = z.repeat(b,1)
                a_in = torch.cat((state,z_tensor),dim = 1)
                a_hat_list = [action_encoder(a_in)for action_encoder in self.action_encoders]
                
                mse_loss = nn.MSELoss()
                
            
                # MSELOSS or crossentropy
                recon_loss = 100*sum([mse_loss(a_hat,a_tensor) for a_hat,a_tensor in zip(a_hat_list,a_tensor_list)])
                loss = recon_loss 
                loss_mean += loss/self.traj_num
                # print(a_tensor_list)
                # print(a_hat_list)
                # print(z)       
                    # print(loss_mean)
            print("loss",loss_mean)
            loss_mean.backward()
            state_optimizer.step()
            for action_optimzer in action_optimzers:
                action_optimzer.step()

            if t%100 == 0:
                # 保存模型
                # torch.save({'model': model.state_dict()}, 'model_name.pth')

                # ## 读取模型
                # model = net()
                # state_dict = torch.load('model_name.pth')
                # model.load_state_dict(state_dict['model'])

                # save model
                path = 'state_encoder_model'
                torch.save({'model': self.state_encoder.state_dict()}, path +'/2th_round'+str(t)+'.pth')
                print(path +'/round'+str(t)+'.pth')
                plt.figure()
                plt.title('round'+str(t))
                
                
                colors = ['r','b','g','gold','gray','black']
                for dataset,color in zip(display_data,colors):
                    z_list = []
                    for traj in dataset:
                        state = Tensor(np.array(traj["observations"])[:,0,:])
                        action0 = Tensor(np.array(traj["actions"])[:,0,:])
                        action1 =Tensor(np.array(traj["actions"])[:,1,:])
                        action2 =Tensor(np.array(traj["actions"])[:,2,:])
                        a_tensor_list = [action0,action1,action2]
                        # flatten the traj
                        s_t = Tensor(np.array(traj["next_observations"])[-1,0,:])
                        batch = torch.cat((state,action0,action1,action2),dim = 1)
                        traj = torch.cat((batch.view(-1),s_t),-1)
                        # print(traj)
                        
                        z,mu,logvar = self.state_encoder(traj)
                        z_num = z.detach().numpy().tolist()
                        z_list.append(z_num)
                        
                    z_list = np.array(z_list)
                
                    plt.scatter(z_list[:,0],z_list[:,1],marker ='o',color = color)
                # plt.scatter(z_list[1,:,0],z_list[1,:,1],marker = 'X',color = 'g')
                plt.show()
                
    # add latent variable to data
    def get_skill(self,model_path,dataset):
        state_encoder = VAE(input_num = self.traj_dim, output_num = self.latent_dim)    
        state_dict = torch.load(model_path)
        state_encoder.load_state_dict(state_dict['model'])   
        z_list = []
        for i,data in enumerate(dataset): 
            new_data = []
            print(i,"............................................",i)
            for  traj in data:
                # print(self.process_traj(traj))
                z,_,_ = state_encoder(self.process_traj(traj))
                print(z)
                z_list.append(z.detach().numpy())
        return z_list
            #     l = len(traj["observations"])
            #     n = len(traj["observations"][0])
            #     # z = torch.Tensor([0.0,0.0])
            #     print(z)
            #     z = z.repeat(n,1)
            #     z = z.repeat(l,1,1)
            #     traj.update({"skill":z.detach().numpy()})
               
                
            #     new_data.append(traj)
                
            # data_path = "processed_data"
            # if os.path.exists(data_path):
            #     pass
            # else:
            #     os.makedirs(data_path)
            # # print(new_data)
            # np.save(os.path.join(data_path,"new_data_"+str(i)+'.npy'), new_data, allow_pickle=True)
            
    def process_traj(self,traj):
        state = Tensor(np.array(traj["observations"])[:,0,:])
        action0 = Tensor(np.array(traj["actions"])[:,0,:])
        action1 =Tensor(np.array(traj["actions"])[:,1,:])
        action2 =Tensor(np.array(traj["actions"])[:,2,:])
        a_tensor_list = [action0,action1,action2]
        # flatten the traj
        s_t = Tensor(np.array(traj["next_observations"])[-1,0,:])
        batch = torch.cat((state,action0,action1,action2),dim = 1)
        traj = torch.cat((batch.view(-1),s_t),-1)
        return traj
            
    def compute_kl_div(self,mu,logvar):
        # torch.distributions.Uniform(low, high)
        ''' compute KL( q(z|c) || r(z) ) '''
        std = logvar.mul(0.5).exp_()
        # print(mu,var)
        # print(torch.zeros(self.latent_dim))
        prior = torch.distributions.Normal(torch.zeros(self.latent_dim), 0.05*torch.ones(self.latent_dim))
        posterior = torch.distributions.Normal(mu, std)
        kl_divs = torch.distributions.kl.kl_divergence(posterior, prior) 
        # print(kl_divs)
        kl_div_sum = torch.sum(kl_divs)
        # print(kl_div_sum)
        return kl_div_sum
            

class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, k=2, tolerance=0.0001, max_iter=300):
        self.k_ = k
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def fit(self, data):
        self.centers_ = {}
        for i in range(self.k_):
            self.centers_[i] = data[i]

        for i in range(self.max_iter_):
            self.clf_ = {}
            for i in range(self.k_):
                self.clf_[i] = []
            # print("质点:",self.centers_)
            for feature in data:
                # distances = [np.linalg.norm(feature-self.centers[center]) for center in self.centers]
                distances = []
                for center in self.centers_:
                    # 欧拉距离
                    # np.sqrt(np.sum((features-self.centers_[center])**2))
                    distances.append(np.linalg.norm(feature - self.centers_[center]))
                classification = distances.index(min(distances))
                self.clf_[classification].append(feature)

            # print("分组情况:",self.clf_)
            prev_centers = dict(self.centers_)
            for c in self.clf_:
                self.centers_[c] = np.average(self.clf_[c], axis=0)

            # '中心点'是否在误差范围
            optimized = True
            for center in self.centers_:
                org_centers = prev_centers[center]
                cur_centers = self.centers_[center]
                if np.sum((cur_centers - org_centers) / org_centers * 100.0) > self.tolerance_:
                    optimized = False
            if optimized:
                break

    def predict(self, p_data):
        distances = [np.linalg.norm(p_data - self.centers_[center]) for center in self.centers_]
        index = distances.index(min(distances))
        return index


# if __name__ == '__main__':
#     x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
#     k_means = K_Means(k=2)
#     k_means.fit(x)
#     print(k_means.centers_)
#     for center in k_means.centers_:
#         pyplot.scatter(k_means.centers_[center][0], k_means.centers_[center][1], marker='*', s=150)

#     for cat in k_means.clf_:
#         for point in k_means.clf_[cat]:
#             pyplot.scatter(point[0], point[1], c=('r' if cat == 0 else 'b'))

#     predict = [[2, 1], [6, 9]]
#     for feature in predict:
#         cat = k_means.predict(predict)
#         pyplot.scatter(feature[0], feature[1], c=('r' if cat == 0 else 'b'), marker='x')

#     pyplot.show()

        



if __name__ ==  '__main__':
    k_means = K_Means(k=6)
    
    mycluster = CLUSTER()
    mycluster.train()
    # dataset0为随机动作（负样本数据）
    # dataset0 = np.load('data/data_0.npy',allow_pickle = True)
    # dataset1 = np.load('data/data_1.npy',allow_pickle = True)
    # dataset2 = np.load('data/data_2.npy',allow_pickle = True)
    # dataset3 = np.load('data/data_3.npy',allow_pickle = True)
    # dataset4 = np.load('data/data_4.npy',allow_pickle = True)
    # dataset5 = np.load('data/data_5.npy',allow_pickle = True)
    # dataset6 = np.load('data/data_6.npy',allow_pickle = True)   
             
    # dataset = [dataset0,dataset1,dataset2,dataset3,dataset4,dataset5,dataset6]
    
    # z_list =  mycluster.get_skill(model_path = os.path.join('state_encoder_model','round700.pth') , dataset = dataset)
    # k_means.fit(z_list)
    # print(k_means.centers_)