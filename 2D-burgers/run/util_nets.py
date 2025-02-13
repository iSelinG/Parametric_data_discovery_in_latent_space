import numpy as np
from scipy.special import binom
from scipy.integrate import odeint
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
import torch
from sklearn.preprocessing import MinMaxScaler

cuda = torch.cuda.is_available()
if cuda:
    device = 'cuda'
else:
    device = 'cpu'

def library_size(n, poly_order, use_sine=False, use_cosine=False, include_constant=True):
    l = 0
    for k in range(poly_order+1):
        l += int(binom(n+k-1,k))
    if use_sine:
        l += n
    if use_cosine:
        l += n
    if not include_constant:
        l -= 1
    return l


def sindy_library_torch(z, latent_dim, poly_order, include_sine=False, include_cosine=False):
    """
    Build the SINDy library.

    Arguments:
        z - 2D tensorflow array of the snapshots on which to build the library. Shape is number of
        time points by the number of state variables.
        latent_dim - Integer, number of state variable in z.
        poly_order - Integer, polynomial order to which to build the library. Max value is 5.
        include_sine - Boolean, whether or not to include sine terms in the library. Default False.

    Returns:
        2D tensorflow array containing the constructed library. Shape is number of time points by
        number of library functions. The number of library functions is determined by the number
        of state variables of the input, the polynomial order, and whether or not sines are included.
    """
    library = [torch.ones(z.shape[0])]
    
    if poly_order > 0:
        for i in range(latent_dim):
            library.append(z[:,i])
    
    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                # library.append(torch.matmul(z[:,i], z[:,j]))
                library.append(torch.mul(z[:,i], z[:,j]))
    
    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    library.append(z[:,i]*z[:,j]*z[:,k])

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p])

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        for q in range(p,latent_dim):
                            library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p]*z[:,q])

    if include_sine:
        for i in range(latent_dim):
            library.append(torch.sin(z[:,i]))
            
    if include_cosine:
        for i in range(latent_dim):
            library.append(torch.cos(z[:,i]))
            
    return torch.stack(library, axis=1)

def sindy_library(X, poly_order, include_sine=False, include_cosine=False):
    m,n = X.shape
    l = library_size(n, poly_order, include_sine, include_cosine, True)
    library = np.ones((m,l))
    index = 1

    if poly_order > 0: 
        for i in range(n):
            library[:,index] = X[:,i]
            index += 1
        
    if poly_order > 1:
        for i in range(n):
            for j in range(i,n):
                library[:,index] = X[:,i]*X[:,j]
                index += 1

    if poly_order > 2:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    library[:,index] = X[:,i]*X[:,j]*X[:,k]
                    index += 1

    if poly_order > 3:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        library[:,index] = X[:,i]*X[:,j]*X[:,k]*X[:,q]
                        index += 1
                    
    if poly_order > 4:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        for r in range(q,n):
                            library[:,index] = X[:,i]*X[:,j]*X[:,k]*X[:,q]*X[:,r]
                            index += 1

    if include_sine:
        for i in range(n):
            library[:,index] = np.sin(X[:,i])
            index += 1
            
    if include_cosine:
        for i in range(n):
            library[:,index] = np.cos(X[:,i])
            index += 1
            
    return library


def sindy_simulate(x0, t, Xi, poly_order, include_sine=False, include_cosine=False):
    '''
    只解一个Xi
    '''
    m = t.size
    n = x0.size  # x0.shape=(nz,)
    f = lambda x,t : np.dot(sindy_library(np.array(x).reshape((1,n)), poly_order, include_sine, include_cosine), Xi).reshape((n,))

    x = odeint(f, x0, t)
    return x   # (t_dim, nz)


def model_sindy(sindy_coef, dataset, model, poly_order,  num_latent_states):
    # state0 = np.zeros((dataset['num_samples'], num_latent_states), dtype=np.float32)
    if hasattr(model, 'state0'):
        state0 = model.state0
        state0 = state0.cpu().detach().numpy()  # (num_latent_sates,)
        state0 = np.tile(state0, (dataset['num_samples'], 1))
    elif hasattr(model, 'state0_0'):
        state0 = [param for name, param in model.named_parameters() if 'state0_' in name]
        state0 = torch.stack(state0)
        state0 = state0.cpu().detach().numpy()
    elif hasattr(model, 'state0_predict'):
        state0 = model.state0_predict(dataset['inp_parameters'])
        state0 = state0.cpu().detach().numpy()
    else:
        state0 = np.zeros((dataset['num_samples'], num_latent_states), dtype=np.float32)

    times = dataset['times']

    s_simu = []
    # of_simu = []

    for k in range(dataset['num_samples']):
        s_simu_k = sindy_simulate(state0[k], times, sindy_coef[k], poly_order)
        s_simu.append(s_simu_k)
    s_simu = np.stack(s_simu, axis=0)

    s_simu = torch.tensor(s_simu, dtype=torch.float32)

    s_simu_expanded = torch.unsqueeze(s_simu, dim=2).to(device)
    s_simu_expanded = s_simu_expanded.expand(dataset['num_samples'], dataset['num_times'], dataset['num_points'], num_latent_states)

    of_simu = []
    inp_para_expanded = dataset['inp_parameters'].unsqueeze(1).unsqueeze(2)
    inp_para_expanded = inp_para_expanded.expand(dataset['num_samples'], dataset['num_times'], dataset['num_points'], 2)
    input_rec = torch.cat([s_simu_expanded, dataset['points_full'], inp_para_expanded], dim=3)  # (num_sample, n_t, s_dim, 2+4)
    # input_rec = torch.cat([s_simu_expanded, dataset['points_full'], inp_para_expanded], dim=3)  # (num_sample, n_t, s_dim, 2+4)
    for i in range(len(input_rec)):
        of_simu.append(model.rec(input_rec[i, :, :, :]).cpu().detach().numpy())
    
    of_simu = np.stack(of_simu)
    # print('of_simu', of_simu.shape)

    return s_simu, of_simu





def normalize_forw(v, v_min, v_max, axis = None, mode = 1, v_range = 1):
    v_min, v_max = reshape_min_max(len(v.shape), v_min, v_max, axis)
    if mode == 1:  
        return v_range * (2.0*v - v_min - v_max) / (v_max - v_min)
    elif mode == 2: 
        return (v - v_min) / (v_max - v_min)
        

def normalize_back(v, v_min, v_max, axis = None, mode = 1):
    v_min, v_max = reshape_min_max(len(v.shape), v_min, v_max, axis)
    if mode == 1:   
        return 0.5*(v_min + v_max + (v_max - v_min) * v)
    elif mode == 2:   
        return v * (v_max - v_min) + v_min



def reshape_min_max(n, v_min, v_max, axis = None):
    if axis is not None:  
        shape_min = [1] * n
        shape_max = [1] * n
        shape_min[axis] = len(v_min)  
        shape_max[axis] = len(v_max)
        v_min = np.reshape(v_min, shape_min)
        v_max = np.reshape(v_max, shape_max)
    return v_min, v_max


def dataset_normalize(dataset, problem, normalization_definition):
    normalization = dict()
    # normalization['dt_base'] = normalization_definition['time']['time_constant']  
    normalization['x_min'] = np.array(normalization_definition['space']['min'])  # -1
    normalization['x_max'] = np.array(normalization_definition['space']['max'])  # +1
    if len(problem.get('input_parameters', [])) > 0:  # 3>0
        normalization['inp_parameters_min'] = np.array([normalization_definition['input_parameters'][v['name']]['min'] for v in problem['input_parameters']])  # (2,)
        normalization['inp_parameters_max'] = np.array([normalization_definition['input_parameters'][v['name']]['max'] for v in problem['input_parameters']])  # (2,)
    # if len(problem.get('input_signals', [])) > 0:
    #     normalization['inp_signals_min'] = np.array([normalization_definition['input_signals'][v['name']]['min'] for v in problem['input_signals']])
    #     normalization['inp_signals_max'] = np.array([normalization_definition['input_signals'][v['name']]['max'] for v in problem['input_signals']])
    normalization['out_fields_min'] = np.array([normalization_definition['output_fields'][v['name']]['min'] for v in problem['output_fields']])
    normalization['out_fields_max'] = np.array([normalization_definition['output_fields'][v['name']]['max'] for v in problem['output_fields']])

    # dataset['times']              = dataset['times'] / normalization['dt_base'] # [0, 0.05, 0.1, ..., 5]/0.5 → [0, 0.1, ..., 10]
    dataset['points']             = normalize_forw(dataset['points']        , normalization['x_min']             , normalization['x_max']             , axis = 1) 
    dataset['points_full']        = normalize_forw(dataset['points_full']   , normalization['x_min']             , normalization['x_max']             , axis = 3).float() 
    if dataset['inp_parameters'] is not None:
        dataset['inp_parameters'] = normalize_forw(dataset['inp_parameters'], normalization['inp_parameters_min'], normalization['inp_parameters_max'], axis = 1, mode=2)
    # if dataset['inp_signals'] is not None:
    #     dataset['inp_signals']    = normalize_forw(dataset['inp_signals']   , normalization['inp_signals_min']   , normalization['inp_signals_max']   , axis = 2)
    # dataset['out_fields']         = normalize_forw(dataset['out_fields']    , normalization['out_fields_min']    , normalization['out_fields_max']    , axis = 3, v_range=20)


def out_fields_normalizetion(dataset):
    out_fields = dataset['out_fields']  # (num_sample/para, n_t, s_dim, 1)
    out_fields_max = np.max(out_fields, axis=(1, 2, 3))
    out_fields_min = np.min(out_fields, axis=(1, 2, 3))  # (num_sample,)
    # np.save(path_results + date_str + './of_max.npy', out_fields_max)
    # np.save(path_results + date_str + './of_max.npy', out_fields_min)
    # dataset['out_fields'] = 10 * (out_fields - out_fields_min[:, np.newaxis, np.newaxis, np.newaxis]) / (out_fields_max - out_fields_min)[:, np.newaxis, np.newaxis, np.newaxis] - 5
    dataset['out_fields'] = 2 * (out_fields - out_fields_min[:, np.newaxis, np.newaxis, np.newaxis]) / (out_fields_max - out_fields_min)[:, np.newaxis, np.newaxis, np.newaxis] - 1
    # return normalized_data

def out_fields_reverse(dataset, of_simu):
    param_grid = dataset['inp_parameters'].cpu().detach().numpy()
    out_fields_max = param_grid[:, 0]
    out_fields_min = -param_grid[:, 0]
    of_simu = (5 + of_simu) / 10 * (out_fields_max - out_fields_min)[:, np.newaxis, np.newaxis, np.newaxis]  + out_fields_min[:, np.newaxis, np.newaxis, np.newaxis]
    return of_simu

