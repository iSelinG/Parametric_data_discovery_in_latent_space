type = 'new'

print('type', type)

# import modules
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
import random
from torch.utils.data import Dataset, DataLoader


num_latent_states = 7 
print('num_latent_states', num_latent_states)


date = time.localtime()
date_str = "{month:02d}_{day:02d}_{year:04d}_{hour:02d}_{minute:02d}"
if type == 'new':
    date_str = 'train_more_' + date_str.format(month = date.tm_mon, day = date.tm_mday, year = date.tm_year, hour = date.tm_hour , minute = date.tm_min) + '_225_new_quadratic_' + str(num_latent_states)
if type == 'small':
    date_str = 'train_more_' +  date_str.format(month = date.tm_mon, day = date.tm_mday, year = date.tm_year, hour = date.tm_hour , minute = date.tm_min) + '_225_small_quadratic_' + str(num_latent_states)

device = torch.device('cuda:0')  
torch.cuda.set_device(device)

cuda = torch.cuda.is_available()
if cuda:
    device = 'cuda'
else:
    device = 'cpu'

1

path_data = './data/'
path_results = './results/new/'



if type == 'new':
    data = np.load(path_data + 'data_2D_burgers_441_new.npy', allow_pickle = True).item()
elif type == 'small':
    data = np.load(path_data + 'data_2D_burgers_441_small.npy', allow_pickle = True).item()
coor = data['coor']





idx_train = random.sample(range(225), 25)
idx_test =  list(range(0, 225, 1)) 


def create_dataset(data, coor, idx):  
    if 'data_v' not in data:
        data['data_v'] = data['data_u']
    new_dataset = {
        'points': coor[:, :], 
        'times': data['t'],   
        'out_fields' : np.concatenate([data['data_u'][idx,:,:,None], data['data_v'][idx,:,:,None]], axis = 3),
        'inp_parameters': data['param'][idx, :],
        'inp_signals': None,
    }

    return new_dataset


def process_dataset(dataset, normalization_definition = None, dt = None):

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

    
        
    if normalization_definition is not None:
        dataset_normalize(dataset, problem, normalization_definition)


    #  convert to tensor
    dataset['inp_parameters'] = torch.tensor(dataset['inp_parameters'], dtype=torch.float32)
    dataset['out_fields'] = torch.tensor(dataset['out_fields'], dtype=torch.float32)

    


problem = {
    'space': {
        'dimension' : 2 
    },
    'input_parameters': [
        { 'name': 'coeff_a' },
        { 'name': 'coeff_w' }
    ],
    'input_signals': [],
    'output_fields': [
        { 'name': 'u' },
        { 'name': 'v' }
    ]
}

normalization = {
    'space': { 'min' : [-1], 'max' : [+1]},

    'output_fields': {
        'u': { 'min': -1, 'max': +1 },
        'v': { 'min': -1, 'max': +1 }
    }
}


if type == 'small':
    normalization['input_parameters'] = {
        'coeff_a': { 'min': 0.7, 'max': 0.9 },
        'coeff_w': { 'min': 0.9, 'max': 1.1 },
        # 'coeff_a': { 'min': 0.5, 'max': 1.0 },  
        # 'coeff_w': { 'min': 0.5, 'max': 1.5 },
    }
elif type == 'new':
    normalization['input_parameters'] = {
        'coeff_a': { 'min': 0.5, 'max': 1.0 },
        'coeff_w': { 'min': 0.5, 'max': 1.5 },
    }

dt = 0.005   

dataset_train = create_dataset(data, coor, idx_train)
dataset_test = create_dataset(data, coor, idx_test)
out_fields_normalizetion(dataset_train)  
out_fields_normalizetion(dataset_test)  
process_dataset(dataset_train, normalization)
process_dataset(dataset_test, normalization)


for i in dataset_train.keys():
    print(i)






cuda = torch.cuda.is_available()
print('cuda', cuda)






import torch


def print_gpu_memory():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:")
        print(f"    Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 3:.2f} GB")
        print(f"    Cached:    {torch.cuda.memory_reserved(i) / 1024 ** 3:.2f} GB")



def print_tensor_memory():
    for obj in dir(torch.cuda):
        if 'Tensor' in obj:
            tensor_class = getattr(torch.cuda, obj)
            try:
                tensor = tensor_class(1)
                print(f"{tensor_class.__name__}: {tensor.element_size() * tensor.nelement() / 1024 ** 3:.2f} GB")
            except Exception as e:
                print(f"Failed to create {tensor_class.__name__}: {e}")



coefficient_initialization = 'uniform'
poly_order = 1
library_dim = library_size(num_latent_states, poly_order)

class ResNetBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ResNetBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out += residual  
        out = self.relu(out)
        return out

class NN(nn.Module):
    def __init__(self, problem=problem, num_latent_states=num_latent_states):
        super(NN, self).__init__()
        input_shape = (num_latent_states + len(problem['input_parameters']),)  
        self.dyn = nn.Sequential(
            nn.Linear(input_shape[0], 6),
            nn.Tanh(),
            ResNetBlock(6, 6),
            nn.Linear(6, 6),
            nn.Tanh(),
            ResNetBlock(6, 6),
            nn.Linear(6, num_latent_states * 2) 
        )
        input_shape = (num_latent_states + problem['space']['dimension'] + len(problem['input_parameters']),)  
        self.rec = nn.Sequential(
            nn.Linear(input_shape[0], 11),
            nn.Tanh(),
            nn.Linear(11, 15),
            nn.Tanh(),
            ResNetBlock(15, 15),
            nn.Linear(15, 15),
            nn.Tanh(),
            ResNetBlock(15, 15),
            nn.Linear(15, 11),
            nn.Tanh(),
            nn.Linear(11, len(problem['output_fields']))
        )
        input_shape = (len(problem['input_parameters']),)
        self.state0_predict = nn.Sequential(
            nn.Linear(input_shape[0], 6),
            nn.Tanh(),
            nn.Linear(6, 6),
            nn.Tanh(),
            nn.Linear(6, 6),
            nn.Tanh(),
            # ResNetBlock(9, 9),
            # nn.Linear(9, 9),
           # nn.Tanh(),
            nn.Linear(6, num_latent_states)
        ) 
        for i in range(len(idx_train)):
            s_c = torch.nn.Parameter(torch.empty(library_dim, num_latent_states))
            if coefficient_initialization == 'uniform':
                torch.nn.init.uniform_(s_c)
            elif coefficient_initialization == 'constant':
                torch.nn.init.constant_(s_c, 0.0)
            setattr(self, f'sindy_coef_{i}', s_c)
        
        


    def forward(self, x):  
        x_ds = self.dyn(x)
        x = self.rec(x_ds)
        return x



model = NN()
# NNdyn = model.NNdyn
# NNrec = model.NNrec







def evolve_dynamics(dataset, model, dt): 
    if hasattr(model, 'state0'):
        state0 = model.state0
        state0_expand = torch.unsqueeze(state0, 0)
        
        state = state0_expand.repeat(dataset['num_samples'], 1)
    elif hasattr(model, 'state0_0'):
        state0 = [param for name, param in model.named_parameters() if 'state0_' in name]
        state = torch.stack(state0)
    elif hasattr(model, 'state0_predict'):
        state = model.state0_predict(dataset['inp_parameters'])
    else:
        state = torch.zeros((dataset['num_samples'], num_latent_states), dtype=torch.float32).to(device)  
    state_history = []
    state_history.append(state)  
    d_state_history = []

    for _ in range(dataset['num_times'] - 1):
        d_state = model.dyn(torch.cat([state, dataset['inp_parameters']], dim=-1))
        d_state_history.append(d_state[: , :len(dataset['inp_parameters'])])  
        state = state + dt * d_state[: , :num_latent_states] + 0.5 * dt * dt * d_state[: , num_latent_states:]  
        state_history.append(state)  
    
    d_state = model.dyn(torch.cat([state, dataset['inp_parameters']], dim=-1))
    d_state_history.append(d_state)
        
    
    ds_t = torch.stack(d_state_history).permute(1, 0, 2)
    s_t = torch.stack(state_history).permute(1, 0, 2)
    
   
    return ds_t, s_t


def reconstruct_output(dataset, states, model):
    states_expanded = torch.unsqueeze(states, dim=2).to(device)
    states_expanded = states_expanded.expand(dataset['num_samples'], dataset['num_times'], dataset['num_points'], num_latent_states)
    inp_para_expanded = dataset['inp_parameters'].unsqueeze(1).unsqueeze(2)
    inp_para_expanded = inp_para_expanded.expand(dataset['num_samples'], dataset['num_times'], dataset['num_points'], 2)

    return model.rec(torch.cat([states_expanded, dataset['points_full'], inp_para_expanded], dim=3)) 

def LDNet(dataset, model, dt):
    d_states, states = evolve_dynamics(dataset, model, dt) 
    return d_states, states, reconstruct_output(dataset, states, model) 

def MSE(dataset, model, dt):
    d_states, state, out_fields = LDNet(dataset, model, dt)
    error = out_fields - dataset['out_fields'] 
    state0_error = error[:, 0, :, :]

    return d_states, state, error, state0_error 




lr = 0.05  
n_iter = 6000
n_sindy_solve = 500

def lr_lambda(epoch):
    if epoch < 3000:
        if epoch % 500 == 0:
            return 0.9
    else:
        if epoch % 1000 == 0:
            return 0.6
    return 1.0

class TotalModel:
    def __init__(self, model, problem, dataset_train, dataset_test):
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        training_losses = []
        self.MSE = MSE
        self.optimizer = optimizer
        self.model = model
        self.training_losses = training_losses
        scheduler = StepLR(optimizer, step_size=1000, gamma=0.9)
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
        best_loss = 10000
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
        model = self.model
        optimizer = self. optimizer
        scheduler = self.scheduler
        model = model.to(device)
        print("Model device:", next(model.parameters()).device)
        training_losses = self.training_losses
        sindy_coef = self.sindy_coef
        MSEloss = torch.nn.MSELoss()


        for iter in range(0, n_iter+1): 
            optimizer.zero_grad()

            d_states, state, error, state0_error = MSE(dataset_train, model, dt)  

            if d_states.device.type == 'cuda':
                state = state.cpu()


            Theta = []
            dz_predict = []
            loss_coef = 0
            for i in range(num_samples):
                Theta.append(sindy_library_torch(state[i,:,:], self.num_latent_state, self.poly_order).to(device))
            for i in range(num_samples):
                dz_predict.append(torch.matmul(Theta[i], sindy_coef[i]))
                loss_coef += torch.norm(sindy_coef[i])
            
            dz_predict = torch.stack(dz_predict, axis=0)  

            
            loss_sindy = MSEloss(d_states[:, :, :num_latent_states], dz_predict)

            loss_state0 = torch.mean(torch.square(state0_error))

            loss_nn = torch.mean(torch.square(error))

            loss_model = loss_nn + 0.05 * loss_sindy + 0.5 * loss_state0

            loss_model.backward()
            optimizer.step()
            scheduler.step()

            if iter % 100 == 0:
                best_loss = loss_model
                torch.save({
                                'epoch': iter,
                                'model_state_dict': model.state_dict(),
      
                            }, path_results + date_str + '/checkpoint_500.pth')


            if iter % 100 == 0:
                if iter % 500 == 0:
                    print('restruction:', 1 - (torch.norm(error, p=2)/torch.norm(dataset_train['out_fields'], p=2)).item())
                print("Iter: %05d/%d, Loss: %3.6f, SINDy Loss: %3.9f, NN Loss: %3.6f, COEF Loss: %3.6f, "
                    % (iter + 1,
                        n_iter,
                        loss_model.item(),
                        loss_sindy.item(),
                        loss_nn.item(),
                        loss_coef.item() / len(state)),
                    end=' ')
                print('LR: %3.6f' % (scheduler.get_last_lr()[0]))

            training_losses.append(loss_model.item())



            if iter > 0 and iter % n_sindy_solve == 0:
                best_sindy_coef = []
                for i in range(num_samples): 
                    sindy_coef_i = (sindy_coef[i]).clone()
                    if sindy_coef_i.device.type == 'cuda':
                        sindy_coef_i = sindy_coef_i.cpu()
                    best_sindy_coef.append(sindy_coef_i.detach().numpy())  
                


                s_simu, of_simu = model_sindy(best_sindy_coef, dataset_train, model, poly_order, num_latent_states) 

                state0 = model.state0_predict(dataset_train['inp_parameters'])
                state0 = state0.cpu().detach().numpy()
                print('restru sindy s: ', 1 - torch.norm(s_simu - state, p=2)/torch.norm(state, p=2))
                print('restru sindy of: ', 1 - np.linalg.norm((of_simu - out_fields.cpu().detach().numpy()).flatten(), ord=2)/np.linalg.norm(out_fields.cpu().detach().numpy().flatten(), ord=2))


            

        result = {}
        re_s = 1 - torch.norm(s_simu - state, p=2)/torch.norm(state, p=2)
        re_of = 1 - np.linalg.norm((of_simu - out_fields.cpu().detach().numpy()).flatten(), ord=2)/np.linalg.norm(out_fields.cpu().detach().numpy().flatten(), ord=2)
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
                    }, path_results + date_str + '/checkpoint_last.pth')

                
retrain = False



if retrain == False:
    if not os.path.isdir(path_results + date_str):
        os.mkdir(path_results + date_str)
else:
    date_str = '03_01_2024_26_07_save_copy'

totalmodel = TotalModel(model, problem, dataset_train, dataset_test)
totalmodel.train()

