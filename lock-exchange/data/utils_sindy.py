from scipy.special import binom
from scipy.integrate import odeint
import os
import pandas as pd
import numpy as np

def library_size(nz, poly_order, use_sine=False, include_constant=True):
    l = 0
    for k in range(poly_order+1):
        l += int(binom(nz+k-1,k))
    if use_sine:
        l += nz
    if not include_constant:
        l -= 1
    return l

def read_parameter(path, para_csv='parameters.csv'):
    parameters = pd.read_csv(path + os.sep + para_csv)
    return parameters

def derivate_single(data, dt):
    # data.shape=(n_ts,nz)
    n_ts = data.shape[0]
    nz = data.shape[1]
    ddata = np.zeros((n_ts-2, nz))
    for i in range(0, n_ts-2):  
        # ddata[i, :] = (data[i-1, :] + data[i+1, :] - 2*data[i, :])/dt/dt
        ddata[i, :] = (data[i+2, :] - data[i, :])/(dt*2)
    return ddata

def derivate(data, dt):
    # U.shape=(para,n_ts,nz)
    p = data.shape[0]
    n_ts = data.shape[1]
    nz = data.shape[2]
    ddata = np.zeros((p, n_ts-2, nz))
    for i in range(p):
        ddata[i, :, :] = derivate_single(data[i, :, :], dt)
    return ddata




import operator
import itertools
def Omega(k, n):
    # k: latent_dim
    # n: poly_order
    if k == 1:
        return [(i,) for i in range(n + 1)]
    else:
        powers = []
        for i in range(n + 1):
            for j in Omega(k - 1, n - i):
                powers.append((i,) + j)
        # print(sorted(powers, reverse=False))
        return sorted(powers, reverse=False)
    


def sindy_library_np(z, poly_order, include_sine=False):
    latent_dim = z.shape[-1]
    # library_term = library_size(latent_dim, poly_order, include_sine, True)
    # print(poly_order,latent_dim)
    # print('library_term',library_term)
    n_ts = z.shape[0]

    usfl = 0 

    if include_sine:
        usfl = 2*latent_dim

    rhs_functions = {}  # dict
    powers = Omega(latent_dim, poly_order)
    f = lambda x,y: np.prod(np.power(list(x),list(y)))  # (x1^y1)*(x2^y2)*...
    for power in powers:
        rhs_functions[power] = [lambda x,y=power: f(x,y), power]

    RHS = np.ones((n_ts, len(powers)+usfl),dtype=np.float64)  # sindy_library
    
    l = 0  

    for pw in rhs_functions.keys():
        func, power = rhs_functions[pw][0], rhs_functions[pw][1]
        # print(power)
        new_column = np.zeros((n_ts,))
        for i in range(n_ts):
            new_column[i] = func(z[i,:],power)  
        RHS[:,l]=new_column
        l=l+1

    if include_sine:
        for i in range(latent_dim):
            RHS[:,l] = np.sin(z[:,i])
            RHS[:,l+1] = np.cos(z[:,i])
            # RHS[:,l] = torch.sin(z[:,i])
            # RHS[:,l+1] = torch.cos(z[:,i])
            l = l+2

    return RHS


from sklearn.linear_model import Lasso
from copy import deepcopy as cp
from scipy.integrate import solve_ivp
l_max_iter=1000 # Default
l_tol=0.0001 # Default

def alasso(RHS, LHS, alr, delta):
    # params
    n_lasso_iterations = 100
    tol = 1e-10
    absl = lambda w: (np.abs(w) ** delta + 1e-30) 

    n_samples, n_features = RHS.shape 
    weights = np.ones(n_features)
    for k in range(n_lasso_iterations):
        prevw = cp(weights)
        RHS_w = RHS / weights[np.newaxis, :]  # step 1: Q**
        clf = Lasso(alpha=alr, fit_intercept=False, max_iter=l_max_iter, tol=l_tol)  # step 2: 2-1
        clf.fit(RHS_w, LHS)  # step 2: 2-2  
        coef_ = clf.coef_ / weights  # step 2: 2-3  
        weights = 1 / absl(coef_)  # step 3  weight=abs(beta)^{-delta}  delta=kaijo=3>0
        if np.mean((weights - prevw) ** 2) < tol:  # weight=abs(beta)^{-delta} 
            break
    return coef_

def tlsa_norm(RHS, LHS, alpha, lamda=10 ** -2, iter=100): # , normalize=2):
  n, d = RHS.shape

  if lamda != 0:
    w = np.linalg.lstsq(RHS.T.dot(RHS) + lamda * np.eye(d), RHS.T.dot(LHS), rcond=None)[0]
  else:
    w = np.linalg.lstsq(RHS, LHS, rcond=None)[0]

  bigcoeffs = np.where(abs(w) > alpha)[0]

  relevant_coeff_num = d

  for it in range(iter):
    smallcoeffs = np.where(abs(w) <= alpha)[0]
    new_bigcoeffs = [i for i in range(d) if i not in smallcoeffs]
    if relevant_coeff_num == len(new_bigcoeffs):
      break  
    else:
      relevant_coeff_num = len(new_bigcoeffs)
    if len(new_bigcoeffs) == 0:
      if it == 0:
        print('Tolenrance too high, all coefficients set below tolerance')
        return w
      else:
        break
    bigcoeffs = new_bigcoeffs  
    w[smallcoeffs] = 0
    if lamda != 0:
      w[bigcoeffs] = np.linalg.lstsq(RHS[:, bigcoeffs].T.dot(RHS[:, bigcoeffs]) + lamda * np.eye(relevant_coeff_num),
                                     RHS[:, bigcoeffs].T.dot(LHS), rcond=None)[0]
    else:
      w[bigcoeffs] = np.linalg.lstsq(RHS[:, bigcoeffs], LHS)[0]
  if len(bigcoeffs) != 0:
    w[bigcoeffs] = np.linalg.lstsq(RHS[:, bigcoeffs], LHS, rcond=None)[0]
    return w
  else:
    return w


import timeout_decorator
@timeout_decorator.timeout(100)
def solve_sindy(n, data, ddata, alr_list, t, poly_order, delta, include_sine=False, regression='alasso'):
    # data.shape=(n_ts-2,nz)
    alpha = alr_list[n]
    latent_dim = data.shape[1]
    usesin = False

    LHS = cp(ddata)

    RHS = sindy_library_np(data, poly_order,usesin)


    xi1 = np.zeros((len(RHS[0,:]),len(LHS[0,:])))
  
    if regression=='alasso':
        for i in range(xi1.shape[1]):
            xi1[:, i] = alasso(RHS, LHS[:,i], alpha, delta)  
    elif regression=='tlsa_norm':
        for i in range(xi1.shape[1]):
            xi1[:, i] = tlsa_norm(RHS, LHS[:,i], alpha)
    else:
        print('regression no exist!')
        return
    xi = cp(xi1)  
    xi = xi.astype(np.float16)  

    def sindy_fun(t,y): 
        re_r = np.zeros((len(LHS[0,:],))) 
        for j in range(len(LHS[0,:])):  
            re_r[j] = y[j]
        ii = 0
        usfl = 0
        if include_sine:
            usfl = 2*latent_dim

        rhs_functions = {}  # dict
        powers = Omega(latent_dim, poly_order)

        f = lambda x, y: np.prod(np.power(list(x), list(y)))
        for power in powers:
            rhs_functions[power] = [lambda x, y=power: f(x, y), power]
       
        re_rhs = np.ones([len(powers) + usfl]) 
        ii = 0
        for pw in rhs_functions.keys():
            func, power = rhs_functions[pw][0], rhs_functions[pw][1]
            re_rhs[ii] = func(re_r,power)
            ii = ii+1
        if usesin:
            for i in range(latent_dim):
                re_rhs[ii] = np.sin(re_r[i])
                re_rhs[ii+1] = np.cos(re_r[i])
                ii = ii+2
        reconstx = np.dot(re_rhs,xi)  

        return (list(reconstx))
    data0 = data[0,:]
    v0 = list(data0)
    taxint = t
    stat = taxint[0]
    endt = taxint[-1]
    vodeivp = solve_ivp(sindy_fun,[stat, endt+1e-05], v0, method='RK45', t_eval=taxint)
    vodet = vodeivp.t   # (n_ts,)
    vodey = vodeivp.y   # (n_ts,k)

    solve_or_not = 1
    if len(vodet)<len(data):
        solve_or_not = 0

    return alpha, xi, vodet, vodey, solve_or_not


def sindy_loop_i(n, data, ddata, alr_list, t, poly_order, delta=3, include_sine=False, method='alasso'): 
    try: 
        alpha, xi, vodet, vodey, solve_or_not = solve_sindy(n, data, ddata, alr_list, t, poly_order, delta, include_sine, method)
        num_nozero = np.count_nonzero(xi)
    except:
        alpha = alr_list[n]
        xi, vodet, vodey, solve_or_not=np.nan,np.nan,np.nan,np.nan
        num_nozero = np.nan
        print('Time out')
    return alpha, xi, vodet, vodey, num_nozero, solve_or_not


def compute_frequency_and_period_for_each_column(data, dt):
  frequencies = []
  periods = []

  for i in range(data.shape[1]):
    values = data[:, i]

    time_step = dt 

    fft_result = np.fft.fft(values)
    fft_freq = np.fft.fftfreq(len(fft_result), d=time_step)

    positive_freq_mask = fft_freq > 0
    positive_fft_freq = fft_freq[positive_freq_mask]
    positive_fft_result = fft_result[positive_freq_mask]
    max_amplitude_index = np.argmax(np.abs(positive_fft_result))

    dominant_frequency = positive_fft_freq[max_amplitude_index]
    dominant_period = 1 / dominant_frequency if dominant_frequency != 0 else np.inf

    frequencies.append(dominant_frequency)
    periods.append(dominant_period)

  return frequencies, periods

import math
@timeout_decorator.timeout(100)
def solve_sindy_T(n, data, ddata, period, num_periods, alr_list, t, poly_order, delta, include_sine=False, regression='alasso'):
  alpha = alr_list[n]
  latent_dim = data.shape[1]
  usesin = False

  T0 = t[(t >= 0) & (t < period)]
  LHS = cp(ddata[(t >= 0) & (t < period)])
  RHS = sindy_library_np(data[(t >= 0) & (t < period),:], poly_order,usesin)
  xi1 = np.zeros((len(RHS[0,:]),len(LHS[0,:])))
  if regression=='alasso':
    for i in range(xi1.shape[1]):
      xi1[:, i] = alasso(RHS, LHS[:,i], alpha, delta) 
  elif regression=='tlsa_norm':
      for i in range(xi1.shape[1]):
          xi1[:, i] = tlsa_norm(RHS, LHS[:,i], alpha)
  else:
    print('regression no exist!')
    return
  xi = cp(xi1)  
  xi = xi.astype(np.float16)

  def sindy_fun(t,y): 
    re_r = np.zeros((len(LHS[0,:],)))  # (latent_dim,)
    for j in range(len(LHS[0,:])):  # r1,r2,r3,r4|t=0
      re_r[j] = y[j]
    ii = 0
    usfl = 0
    if include_sine:
      usfl = 2*latent_dim

    rhs_functions = {}  # dict
    powers = Omega(latent_dim, poly_order)
    f = lambda x, y: np.prod(np.power(list(x), list(y)))  # (x1^y1)*(x2^y2)*...
    for power in powers:
      rhs_functions[power] = [lambda x, y=power: f(x, y), power]

    re_rhs = np.ones([len(powers) + usfl])

    ii = 0
    for pw in rhs_functions.keys():
      func, power = rhs_functions[pw][0], rhs_functions[pw][1]
      re_rhs[ii] = func(re_r,power)
      ii = ii+1
 
    if usesin:
      for i in range(latent_dim):
        re_rhs[ii] = np.sin(re_r[i])
        re_rhs[ii+1] = np.cos(re_r[i])
        ii = ii+2
    reconstx = np.dot(re_rhs,xi)  
   
    return (list(reconstx))

  vodet = np.zeros_like(t)
  vodey = np.zeros_like(data)

  for n_peri in range(num_periods):
    Tn = t[(t >= n_peri*period) & (t < (n_peri+1)*period)]  
    # print('Tn:', len(Tn), Tn)
    Tn_normalize = Tn - n_peri*period  
    taxint = np.concatenate(([0], Tn_normalize))
    data0 = data[0,:]
    v0 = list(data0)
    stat = taxint[0]
    endt = taxint[-1]
    vodeivp1 = solve_ivp(sindy_fun,[stat, endt+1e-05], v0, method='RK45', t_eval=taxint)
    vodet1 = vodeivp1.t   
    vodey1 = vodeivp1.y   

    index_first = np.argmax(t>=n_peri*period)
 
    vodet[index_first: index_first+len(vodet1)-1] = vodet1[1:]
    vodey[index_first: index_first+len(vodet1)-1,:] = vodey1[:,1:].T

  solve_or_not = 1
  if (vodet.size - np.count_nonzero(vodet))>1:  
    solve_or_not = 0


  return alpha, xi, vodet, vodey, solve_or_not







def sindy_loop_i_T(n, data, ddata, num_periods, period, alr_list, t, poly_order, delta=3, include_sine=False):
  try:
    alpha, xi, vodet, vodey, solve_or_not = solve_sindy_T(n, data, ddata, period, num_periods, alr_list, t, poly_order, delta, include_sine, 'alasso')
    num_nozero = np.count_nonzero(xi)
  except:
    alpha = alr_list[n]
    xi, vodet, vodey, solve_or_not=np.nan,np.nan,np.nan,np.nan
    num_nozero = np.nan
    print('Time out')
  return alpha, xi, vodet, vodey, num_nozero, solve_or_not, period, num_periods


  


def find_sindy_coef(z, t, dt):
    dz = np.zeros()




def rms_and_mean(vodey, vodet): 
    data = vodey.T
    tempval=len(data)
    k = data.shape[1]

    meanx=np.mean(data,axis=0)
    mean1=np.sum(meanx)

    msx=np.zeros(k)
    rmsx=np.zeros(k)
    for i in range(k):
        msx[i]=np.sum((data[-tempval:,i]-meanx[i])**2)
        rmsx[i]=np.sqrt(msx[i]/tempval)
    rms1=np.sum(rmsx)



    return mean1,rms1




@timeout_decorator.timeout(100)
def solve_xi(data, data0, xi, t, poly_order, delta, include_sine=False, regression='alasso'):
    latent_dim = data.shape[1]
    usesin = False
    
    xi = xi.astype(np.float16) 

    def sindy_fun(t,y):
        re_r = np.zeros((latent_dim,)) 
        for j in range(latent_dim):  
            re_r[j] = y[j]
        ii = 0
        usfl = 0
        if include_sine:
            usfl = 2*latent_dim

        rhs_functions = {}  # dict
        powers = Omega(latent_dim, poly_order)

        f = lambda x, y: np.prod(np.power(list(x), list(y)))  
        for power in powers:
            rhs_functions[power] = [lambda x, y=power: f(x, y), power]
       
        re_rhs = np.ones([len(powers) + usfl])


        ii = 0
        for pw in rhs_functions.keys():
            func, power = rhs_functions[pw][0], rhs_functions[pw][1]
            re_rhs[ii] = func(re_r,power)
            ii = ii+1
  
        if usesin:
            for i in range(latent_dim):
                re_rhs[ii] = np.sin(re_r[i])
                re_rhs[ii+1] = np.cos(re_r[i])
                ii = ii+2
        reconstx = np.dot(re_rhs,xi)  

        return (list(reconstx))
    data0 = data[0,:]
    v0 = list(data0)
    taxint = t
    stat = taxint[0]
    endt = taxint[-1]
    vodeivp = solve_ivp(sindy_fun,[stat, endt+1e-05], v0, method='RK45', t_eval=taxint)
    vodet = vodeivp.t   
    vodey = vodeivp.y  

    solve_or_not = 1
    if len(vodet)<len(data):
        solve_or_not = 0


    return xi, vodet, vodey, solve_or_not


def xi_loop_i(data, data0, xi, t, poly_order, delta=3, include_sine=False):
    try:
        xi, vodet, vodey, solve_or_not = solve_xi(data, data0, xi, t, poly_order, delta, include_sine, 'alasso')
        num_nozero = np.count_nonzero(xi)
    except:
        xi, vodet, vodey, solve_or_not=np.nan,np.nan,np.nan,np.nan
        num_nozero = np.nan
        print('Time out')
    return xi, vodet, vodey, num_nozero, solve_or_not



@timeout_decorator.timeout(100)
def solve_xi_T(data, data0, period, num_periods, xi, t, poly_order, delta, include_sine=False, regression='alasso'):

  latent_dim = data.shape[1]
  usesin = False
  

  T0 = t[(t >= 0) & (t < period)]

  xi = xi.astype(np.float16)


  def sindy_fun(t,y): 
    re_r = np.zeros((len(LHS[0,:],)))  # (latent_dim,)
    for j in range(len(LHS[0,:])):  # r1,r2,r3,r4|t=0
      re_r[j] = y[j]
    ii = 0
    usfl = 0
    if include_sine:
      usfl = 2*latent_dim


    rhs_functions = {}  # dict
    powers = Omega(latent_dim, poly_order)

    f = lambda x, y: np.prod(np.power(list(x), list(y)))  # (x1^y1)*(x2^y2)*...
    for power in powers:
      rhs_functions[power] = [lambda x, y=power: f(x, y), power]


    re_rhs = np.ones([len(powers) + usfl]) 

    ii = 0
    for pw in rhs_functions.keys():
      func, power = rhs_functions[pw][0], rhs_functions[pw][1]
      re_rhs[ii] = func(re_r,power)
      ii = ii+1

    if usesin:
      for i in range(latent_dim):
        re_rhs[ii] = np.sin(re_r[i])
        re_rhs[ii+1] = np.cos(re_r[i])
        ii = ii+2
    reconstx = np.dot(re_rhs,xi) 
    return (list(reconstx))

  vodet = np.zeros_like(t)
  vodey = np.zeros_like(data)

  for n_peri in range(num_periods):
    Tn = t[(t >= n_peri*period) & (t < (n_peri+1)*period)] 
    Tn_normalize = Tn - n_peri*period
    taxint = np.concatenate(([0], Tn_normalize))
    data0 = data[0,:]
    v0 = list(data0)
    stat = taxint[0]
    endt = taxint[-1]
    vodeivp1 = solve_ivp(sindy_fun,[stat, endt+1e-05], v0, method='RK45', t_eval=taxint)
    vodet1 = vodeivp1.t   
    vodey1 = vodeivp1.y   
    index_first = np.argmax(t>=n_peri*period)
    vodet[index_first: index_first+len(vodet1)-1] = vodet1[1:] +  n_peri*period
    vodey[index_first: index_first+len(vodet1)-1,:] = vodey1[:,1:].T

  solve_or_not = 1
  if (vodet.size - np.count_nonzero(vodet))>1: 
    solve_or_not = 0


  return xi, vodet, vodey, solve_or_not


def xi_loop_i_T(data, data0, num_periods, period, xi, t, poly_order, delta=3, include_sine=False):

    try:
        xi, vodet, vodey, solve_or_not = solve_xi_T(data, data0, period, num_periods, xi, t, poly_order, delta, include_sine, 'alasso')
        num_nozero = np.count_nonzero(xi)
    except:
        xi, vodet, vodey, solve_or_not=np.nan,np.nan,np.nan,np.nan
        num_nozero = np.nan
        print('Time out')
    return xi, vodet, vodey, num_nozero, solve_or_not


  