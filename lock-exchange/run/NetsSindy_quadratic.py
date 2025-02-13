import os, sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import interpolate
from torch.optim.lr_scheduler import StepLR, LambdaLR
from util_nets import *
import pickle
import time
from torch.utils.data import Dataset, DataLoader
import random
# import tensorflow as tf
# # We configure TensorFlow to work in double precision 
# tf.keras.backend.set_floatx('float64')

# import utils
# import optimization

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 12
seed_everything(seed)

num_latent_states = 4 

loss_sindy_weight = 0.1

print('seed', seed)
print('num_latent_states', num_latent_states)

date = time.localtime()
date_str = "{month:02d}_{day:02d}_{year:04d}_{hour:02d}_{minute:02d}"
date_str = date_str.format(month = date.tm_mon, day = date.tm_mday, year = date.tm_year, hour = date.tm_hour, minute = date.tm_min) + '_nz_' + str(num_latent_states) + '_seed_' + str(seed) + '_sindy_weight_' + str(loss_sindy_weight) + '_' + str(441)


device = torch.device('cuda:0')  
torch.cuda.set_device(device)

cuda = torch.cuda.is_available()
if cuda:
    device = 'cuda'
else:
    device = 'cpu'




path_data = './lock_exchange/data/'
path_checkpoint = './lock_exchange/checkpoint/'
path_results = './lock-exchange/results/'



data = np.load(path_data + 'data_test_le_441_150_0-2.npy', allow_pickle = True).item()
coor = np.load(path_data + 'coor.npy')



idx_train = random.sample(range(441), 55)
idx_test =  list(range(0, 441, 1)) 

def out_fiedl_normal_all(dataset, v_range):
    for i in range(dataset['num_samples']):
        inp_para = dataset['inp_parameters'][i, :]
        v_max = inp_para[0]
        v_min = - v_max
        v = dataset['out_fields'][i, :, :, :] 
        dataset['out_fields'][i, :, :, :]  = v_range * (2.0*v - v_min - v_max) / (v_max - v_min)


def normalize_forw(v, v_min, v_max, axis = None, mode = 1, v_range = 1):
    v_min, v_max = reshape_min_max(len(v.shape), v_min, v_max, axis)
    if mode == 1:  
        return v_range * (2.0*v - v_min - v_max) / (v_max - v_min)
    elif mode == 2:  
        return (v - v_min) / (v_max - v_min)
    

def reshape_min_max(n, v_min, v_max, axis = None):
    if axis is not None: 
        shape_min = [1] * n
        shape_max = [1] * n
        shape_min[axis] = len(v_min)  
        shape_max[axis] = len(v_max)
        v_min = np.reshape(v_min, shape_min)
        v_max = np.reshape(v_max, shape_max)
    return v_min, v_max



def create_dataset(data, coor, idx):  
    new_dataset = {
        'points': coor[:, :],  # (1491,2)
        'times': data['t'],   # (150,) dt=0.05
        'out_fields': data['X_test'][idx, :, :, None],
        'inp_parameters': data['param_grid'][idx, :],
        'inp_signals': None,
    }

    return new_dataset


def process_dataset(dataset, dt = None):

    if dt is not None:
        times = np.arange(dataset['times'][0], dataset['times'][-1] + dt * 1e-10, step = dt)
        if dataset['inp_signals'] is not None:
            dataset['inp_signals'] = interpolate.interp1d(dataset['times'], dataset['inp_signals'], axis = 1)(times)
        dataset['out_fields'] = interpolate.interp1d(dataset['times'], dataset['out_fields'], axis = 1)(times)
        dataset['times'] = times
    
    
    num_samples = dataset['out_fields'].shape[0]
    num_times = dataset['times'].shape[0]
    num_points = dataset['points'].shape[0]
    num_x = dataset['points'].shape[1]

    points_full = np.broadcast_to(dataset['points'][None,None,:,:], [num_samples, num_times, num_points, num_x])  # (n_sample, n_time, 100, 1) 将space坐标扩展到所有时间和样本点上

    # dataset['points_full'] = points_full
    dataset['points_full'] = torch.tensor(points_full, dtype=torch.float32)
    dataset['num_points'] = dataset['points_full'].shape[2]  # s_dim
    dataset['num_times'] = num_times
    dataset['num_samples'] = num_samples

    # convert to tensor
    dataset['inp_parameters'] = torch.tensor(dataset['inp_parameters'], dtype=torch.float32)
    dataset['out_fields'] = torch.tensor(dataset['out_fields'], dtype=torch.float32)

    
    # return dataset


dataset_train = create_dataset(data, coor, idx_train)
dataset_test = create_dataset(data, coor, idx_test)
# process_dataset(dataset_train, dt = 0.5)
process_dataset(dataset_train)
process_dataset(dataset_test)

out_fiedl_normal_all(dataset_train, v_range=10)
out_fiedl_normal_all(dataset_test, v_range=10)



cuda = torch.cuda.is_available()
print('cuda', cuda)

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


num_latent_states = 4


problem = {
    'space': {
        'dimension' : 2 # lock-exchange problem
    },
    'input_parameters': [
        { 'name': 'velocity' },
        { 'name': 'viscosity' }
    ],
    'input_signals': [],
    'output_fields': [
        { 'name': 'Temperature' }
    ]
}




# idx_train
coefficient_initialization = 'uniform'
poly_order = 1
library_dim = library_size(num_latent_states, poly_order)

class NN(nn.Module):
    def __init__(self, problem=problem, num_latent_states=num_latent_states):
        super(NN, self).__init__()
        input_shape = (num_latent_states + len(problem['input_parameters']),)  # 低维维度s(t)2 + 参数维数3  # 3+2
        self.dyn = nn.Sequential(
            nn.Linear(input_shape[0], 6),
            nn.Tanh(),
            nn.Linear(6, 6),
            nn.Tanh(),
            nn.Linear(6, num_latent_states * 2)  # (ds, d2s)
        )
        input_shape = (num_latent_states + problem['space']['dimension'],)  # s(t)维度2 + 空间总维数1
        self.rec = nn.Sequential(
            nn.Linear(input_shape[0], 9),
            nn.Tanh(),
            nn.Linear(9, 9),
            nn.Tanh(),
            nn.Linear(9, len(problem['output_fields']))
        )
        for i in range(len(idx_train)):
            s_c = torch.nn.Parameter(torch.empty(library_dim, num_latent_states))
            if coefficient_initialization == 'uniform':
                torch.nn.init.uniform_(s_c)
            elif coefficient_initialization == 'constant':
                torch.nn.init.constant_(s_c, 0.0)
            setattr(self, f'sindy_coef_{i}', s_c)

    # def NNdyn(self, x):
    #     x = self.dyn(x)

    # def NNrec(self, x):
    #     x = self.rec(x)

    def forward(self, x):  
        x_ds = self.dyn(x)
        x = self.rec(x_ds)
        return x



model = NN()
# NNdyn = model.NNdyn
# NNrec = model.NNrec



dt = 0.05



def evolve_dynamics(dataset, model, dt): 
    # 
    state = torch.zeros((dataset['num_samples'], num_latent_states), dtype=torch.float32).to(device)  # (100,2)  /  (147, 3)
    state_history = []
    state_history.append(state)  # s(t_0).shape=(n_sample,n_z)=0
    d_state_history = []
    # dt_ref = normalization['time']['time_constant']

    for _ in range(dataset['num_times'] - 1):
        d_state = model.dyn(torch.cat([state, dataset['inp_parameters']], dim=-1))
        d_state_history.append(d_state[: , :len(dataset['inp_parameters'])])  
        state = state + dt * d_state[: , :num_latent_states] + 0.5 * dt * dt * d_state[: , num_latent_states:]   
        state_history.append(state)  

    d_state = model.dyn(torch.cat([state, dataset['inp_parameters']], dim=-1))
    d_state_history.append(d_state)
        
    
    ds_t = torch.stack(d_state_history).permute(1, 0, 2)
    s_t = torch.stack(state_history).permute(1, 0, 2)
    
    # return torch.stack(state_history).permute(1, 0, 2)  # (n_sample,n_t,n_latent_state) s(t)
    return ds_t, s_t


def reconstruct_output(dataset, states, model):
    states_expanded = torch.unsqueeze(states, dim=2).to(device)
    states_expanded = states_expanded.expand(dataset['num_samples'], dataset['num_times'], dataset['num_points'], num_latent_states)
    # states_expanded = tf.broadcast_to(tf.expand_dims(states, axis = 2), 
    #     [dataset['num_samples'], dataset['num_times'], dataset['num_points'], num_latent_states])  # (100,101,1,2)→(n_sample100,n_t101,s_dim100,2) 

    return model.rec(torch.cat([states_expanded, dataset['points_full']], dim=3)) # NNrec(s(t),x)=y/z  s(t)(n_sample,n_t,2)+x(s_dim,) → (n_sample,n_t,s_dim,3) → (n_sample,n_t,s_dim,1)

def LDNet(dataset, model, dt):
    d_states, states = evolve_dynamics(dataset, model, dt)  # dt,dt_ref,n_ts,n_sample,n_latent_state,NNdyn,u → s(t) (n_sample,n_t,n_latent_state=2)
    return d_states, states, reconstruct_output(dataset, states, model)  # y(x,t)=(n_sample,n_t,s_dim,1)

def MSE(dataset, model, dt):
    d_states, state, out_fields = LDNet(dataset, model, dt)
    error = out_fields - dataset['out_fields']  # (n_sample, n_t, s_dim, 1)

    return d_states, state, error # torch.mean(torch.square(error))




lr = 0.05  
n_iter = 20000 # 10000
n_sindy_solve = 500

class TotalModel:
    def __init__(self, model, problem, dataset_train, dataset_test):
        # path_results = model_parameters['path_results']

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        # MSE = torch.nn.MSELoss()
        training_losses = []
        
        self.MSE = MSE
        self.optimizer = optimizer
        self.model = model
        self.training_losses = training_losses

        # scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)  # 每经过step_size个epoch，将学习率乘以gamma
        scheduler = StepLR(optimizer, step_size=1000, gamma=0.9)  # 每经过step_size个epoch，将学习率乘以gamma
        self.scheduler = scheduler

        
        self.problem = problem

        self.n_iter = n_iter


        self.dt = dt
        self.points = dataset_train['points']
        self.times = dataset_train['times']
        self.out_fields = dataset_train['out_fields'].to(device)
        self.inp_parameters = dataset_train['inp_parameters'].to(device)
        self.points_full = dataset_train['points_full'].to(device)  

        dataset = dataset_train
        dataset['times'] = self.times
        dataset['points'] = self.points
        dataset['out_fields'] = self.out_fields
        dataset['inp_parameters'] = self.inp_parameters
        dataset['points_full'] = self.points_full

        self.dataset_train = dataset


        dataset = dataset_test
        dataset['times'] = self.times
        dataset['points'] = self.points
        dataset['out_fields'] = dataset_test['out_fields'].to(device)
        dataset['inp_parameters'] = dataset_test['inp_parameters'].to(device)
        dataset['points_full'] = dataset_test['points_full'].to(device)

        self.dataset_test = dataset

        self.poly_order = poly_order
        self.num_latent_states = num_latent_states
        sindy_coef = []
        for i in range(len(dataset_train['inp_parameters'])):
            sindy_coef_i = getattr(model, f'sindy_coef_{i}')
            setattr(self, f'sindy_coef_{i}', sindy_coef_i)
            sindy_coef.append(sindy_coef_i)
        self.sindy_coef = sindy_coef

    def train(self):

        n_iter = self.n_iter
        MSE = self.MSE

        dt = self.dt
        points = self.points
        times = self.times
        out_fields = self.out_fields
        inp_parameters = self.inp_parameters
        points_full = self.points_full

        dataset_train = self.dataset_train
        dataset_test = self.dataset_test

        num_samples = dataset_train['num_samples']

        # device = self.device

        model = self.model
        optimizer = self. optimizer
        scheduler = self.scheduler

        model = model.to(device)
        print("Model device:", next(model.parameters()).device)

        training_losses = self.training_losses


        sindy_coef = self.sindy_coef

        MSEloss = torch.nn.MSELoss()

        for iter in range(0, n_iter+1):  # 28000
        
            optimizer.zero_grad()
            # s_t = model.dyn(dataset_train)
            # X_pred = autoencoder.decoder(Z)
            # z = Z.cpu()

            d_states, state, error = MSE(dataset_train, model, dt)  # d_states.shape(cuda)=(n_sample,n_t,n_latent_states), error.shape=# (n_sample, n_t, s_dim, 1)

            if d_states.device.type == 'cuda':
                # d_states = d_states.cpu()
                state = state.cpu()


            Theta = []
            dz_predict = []
            loss_coef = 0
            for i in range(num_samples):
                Theta.append(sindy_library_torch(state[i,:,:], self.num_latent_states, self.poly_order).to(device))
            for i in range(num_samples):
                dz_predict.append(torch.matmul(Theta[i], sindy_coef[i]))
                loss_coef += torch.norm(sindy_coef[i])
            
            dz_predict = torch.stack(dz_predict, axis=0)  # (n_sample,n_t,3)

            
            loss_sindy = MSEloss(d_states[:, :, :num_latent_states], dz_predict)

            loss_nn = torch.mean(torch.square(error))

            loss_model = loss_nn + 0.5 * loss_sindy

            loss_model.backward()
            optimizer.step()
            scheduler.step()
            
            model.eval()
            with torch.no_grad():
                _, _, valid_error = MSE(dataset_test, model, dt) # loss(dataset_test, model, dt)
                valid_loss_model = torch.mean(torch.square(valid_error))

            if iter % 100 == 0:
                if iter % 500 == 0:
                    print('restruction:', 1 - (torch.norm(error, p=2)/torch.norm(dataset_train['out_fields'], p=2)).item())
                print("Iter: %05d/%d, Loss: %3.6f, SINDy Loss: %3.6f, NN Loss: %3.6f, Valid NN Loss: %3.6f, COEF Loss: %3.6f, "
                    % (iter + 1,
                        n_iter,
                        loss_model.item(),
                        loss_sindy.item(),
                        loss_nn.item(),
                        valid_loss_model.item(),
                        loss_coef.item() / len(state)),
                    end=' ')
                print('LR: %3.6f' % (scheduler.get_last_lr()[0]))

            training_losses.append(loss_model.item())

            if iter > 0 and iter % n_sindy_solve == 0:
                # X_train = X_train.cpu()
                # autoencoder = autoencoder.cpu()
                # autoencoder.load_state_dict(torch.load(path_results + date_str + '/checkpoint.pth')['model_state_dict'])
                best_sindy_coef = []
                for i in range(num_samples):  
                    sindy_coef_i = (sindy_coef[i]).clone()
                    if sindy_coef_i.device.type == 'cuda':
                        sindy_coef_i = sindy_coef_i.cpu()
                    best_sindy_coef.append(sindy_coef_i.detach().numpy())  # list num_samples list[0].shape=(library_size,num_latent_states)
                
                

                s_simu, of_simu = model_sindy(best_sindy_coef, dataset_train, model, poly_order, num_latent_states)  # S0=0 

                print('restru sindy s: ', 1 - torch.norm(s_simu - state, p=2)/torch.norm(state, p=2))
                print('restru sindy of: ', 1 - torch.norm(of_simu - out_fields, p=2)/torch.norm(out_fields, p=2))


                
        result = {}
        re_s = 1 - torch.norm(s_simu - state, p=2)/torch.norm(state, p=2)
        re_of = 1 - torch.norm(of_simu - out_fields, p=2)/torch.norm(out_fields, p=2)
        result['restru_sindy_s'] = re_s
        result['restru_sindy_of'] = re_of
        result['training_losses'] = training_losses


        np.save(path_results + date_str + '/result.npy', result, allow_pickle=True)
        np.save(path_results + date_str + '/of_simu.npy', of_simu)
        np.save(path_results + date_str + '/s_simu.npy', s_simu.cpu().detach().numpy())
        np.save(path_results + date_str + '/state.npy', state.cpu().detach().numpy())

        np.save(path_results + date_str + '/sindy_coef.npy', best_sindy_coef)

        model.cpu()
        
        torch.save({
                        'epoch': iter,
                        'model_state_dict': model.state_dict(),
                        # 'optimizer_state_dict': optimizer.state_dict(),
                    }, path_results + date_str + '/checkpoint_last.pth')
        # np.save(path_results + date_str + '/param_train_update.npy', param_train)
        
        # if loss.item() < best_loss and iter % 100 == 0:
        #         best_loss = loss
        #         with open(path_results + date_str + '/training_losses.pkl', 'wb') as f:  
        #             pickle.dump(training_losses, f)
        #         torch.save({
        #                         'epoch': iter,
        #                         'model_state_dict': autoencoder.state_dict(),
        #                         # 'optimizer_state_dict': optimizer.state_dict(),
        #                     }, path_results + date_str + '/checkpoint.pth')
        #         np.save(path_results + date_str + '/param_train_update.npy', param_train)


retrain = False



if retrain == False:
    # model_parameters['date_str'] = date_str
    if not os.path.isdir(path_results + date_str):
        os.mkdir(path_results + date_str)
else:
    date_str = '03_01_2024_26_07_save_copy'


totalmodel = TotalModel(model, problem, dataset_train, dataset_test)
totalmodel.train()


