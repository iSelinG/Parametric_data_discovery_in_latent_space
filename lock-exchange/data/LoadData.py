import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import numpy as np
# import pandas as pd
import meshio
# import joblib
# from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
# from tqdm import tqdm

from torch_params import torch_params

from GenerateData import *

import pyvista as pv

class LoadData(object):
    def __init__(self):
        super(LoadData, self).__init__()

    def get_vtu_num(self, vtu_path):
        f_list = os.listdir(vtu_path)
        vtu_num = 0
        for i in f_list:
            if os.path.splitext(i)[1] == '.vtu':
                vtu_num = vtu_num + 1
        return vtu_num

    
    def coordinate_from_vtu(self, vtu_path, save=0):  
        mesh = meshio.read(os.path.join(vtu_path, 'lock_exchange_0.vtu'))
        coor = mesh.points[:, [0,1]]
        if save == 1:
            np.save(os.path.join(torch_params['coor_folder'], "coor.npy"), coor)
        return coor.shape[0]  

    def coordinate_from_npy(self):
        if not os.path.exists(torch_params['coor_folder'] + os.sep + 'coor.npy'):
            file_list = sorted(os.listdir(torch_params['datasets_folder']))
            self.coordinate_from_vtu(os.path.join(torch_params['datasets_folder'], file_list[0]), 1)
        coor = np.load(torch_params['coor_folder'] + os.sep + 'coor.npy')
        num_points = coor.shape[0]
        return num_points
            
    
    def read_from_vtu(self, vtu_data_path = torch_params['datasets_folder'], full_model_path = torch_params['FullModel_folder']):
        '''
        '''
        num_points = self.coordinate_from_npy()
        vtu_start = torch_params['vtu_start']
        vtu_end = vtu_start + torch_params['vtu_time_num'] - 1
        temperature_whole = np.zeros((len(os.listdir(vtu_data_path)), torch_params['vtu_time_num'], num_points))
        
        i = 0
        for folder in sorted(os.listdir(vtu_data_path)):  
            temperature = np.zeros((vtu_end-vtu_start+1, num_points)) 
            le_name = "/lock_exchange_"
            for n in range(0, vtu_end-vtu_start+1):
                filename = vtu_data_path + folder + le_name + str(vtu_start+n) + ".vtu"
                mesh = pv.read(filename)
                temperature[n, :] = mesh.point_data["Temperature"]
                # print(str(n + vtu_start) + ".vtu has been interpolate.")
            # np.save(os.path.join(torch_params['FullModel_folder'], 'npy_file_9', folder + "_velocity.npy"), velocity)
            np.save(os.path.join(full_model_path,  folder + "_temperature.npy"), temperature)
            # print(velocity.shape)
            temperature_whole[i, :, :] = temperature
            print(folder + " vtu has been read into npy.")
            i = i + 1
        print('temperature_whole.shape: ', temperature_whole.shape)

        return temperature_whole
    
    def read_data(self, npy_path = torch_params['FullModel_folder']):
        '''
        '''
        # npy_path = os.path.join(torch_params['FullModel_folder'], 'npy_file_9')
        # npy_path = torch_params['FullModel_folder']
        if len(os.listdir(npy_path))<2:
            self.read_from_vtu()
        num_points = self.coordinate_from_npy()
        temperature_whole = np.zeros((len(os.listdir(npy_path)), torch_params['vtu_time_num'], num_points))  # (paras, n_ts, node, 2)
        i = 0
        for file in sorted(os.listdir(npy_path)):  
            if file != 'coor.npy':
                temperature = np.load(os.path.join(npy_path, file))
                temperature_whole[i, :, :] = temperature
                i = i + 1
        # print('velocity_whole.shape: ', velocity_whole.shape)
        T = temperature_whole
        return T  # (para, n_ts, nodes)
    


        



if __name__ == '__main__':
    loador = LoadData()

    
    num_points = loador.coordinate_from_npy()
    print(num_points)
    

    
    temperature_whole = loador.read_from_vtu(torch_params['datasets_folder'], torch_params['FullModel_folder'])

    T = loador.read_data(torch_params['FullModel_folder'])
    print(T.shape)
    
            

    
    
