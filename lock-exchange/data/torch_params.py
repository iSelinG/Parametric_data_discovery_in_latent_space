
import numpy as np
# from LoadData import LoadData
# import LoadData
import torch
import os
from utils_sindy import *





torch_params = {


    'source_path': '/root/data1/lock_exchange/data/',  # 原始flml和msh存在的地方

    'datasets_folder': '/root/data1/lock_exchange/data/vtu_data_test/',  # 将存放vtu的文件夹
    'datasets_folder_test': '/root/data1/lock_exchange/data/vtu_data_test/',

    'FullModel_folder': '/root/data1/lock_exchange/data/npy_data_train/',  # 存放读取vtu后的npy文件（相当于LoadData结果），存放coor.npy，未备份flml和msh
    'FullModel_folder_test': '/root/data1/lock_exchange/data/npy_data_test/',

    'coor_folder': '/root/data1/lock_exchange/data/',
    
    
    'Docs': '/root/data1/lock_exchange/Docs/',  # 存放参数.csv

    # 时间相关
    'start_time': 0,
    'end_time': 7.5, 
    'timestep': 0.05, 
    'dump_period_in_timesteps': 1, 
    'dt': 0.05,  # 0.05
    # dump_period1=0.05, dump_period2=1.25, start_time=0, end_time=40, timestep=0.005
    

    
    # 'vtu_time_num': 102,   # vtu时间文件总数（加上头0尾52）
    'vtu_start': 0,
    'vtu_end': 149, #199, # 299,
    'vtu_time_num': 150, # 200, # 300,

    # # train
    # 'temp_min': 10,
	# 'temp_max': 15, # 0.5->5
	# 'temp_num': 11,

	# 'viso_min': 1e-6,
	# 'viso_1st_ss': 1e-5, # 1e-6->1e-5
	# 'viso_2nd_ss': 1e-4, # 1e-5->1e-4
	# 'viso_max': 1e-3, # 1e-4->1e-3
	# # 'viso_num_sum': 5,
	# 'viso_num_ss': 6,  # -1

    # # train √
    # 'temp_min': 10,
	# 'temp_max': 15, # 0.5->5
	# 'temp_num': 6,

	# # 'viso_min': 1e-6,
	# 'viso_1st_ss': 1e-5, # 1e-6->1e-5
	# 'viso_2nd_ss': 1e-4, # 1e-5->1e-4
	# 'viso_max': 1e-3, # 1e-4->1e-3
	# # 'viso_num_sum': 5,
	# 'viso_num_ss': 5,  # -1

    # test √
    'temp_min': 10.5,
	'temp_max': 14.5, # 0.5->5
	'temp_num': 5,

	# 'viso_min': 1e-6,
	'viso_1st_ss': 1e-5, # 1e-6->1e-5
	'viso_2nd_ss': 1e-4, # 1e-5->1e-4
	'viso_max': 1e-3, # 1e-4->1e-3
	# 'viso_num_sum': 5,
	'viso_num_ss': 3,  # -1

    
    
    



    # POD
    'Model': '/root/data1/lock_exchange/Model/',
    'POD_name': '/root/data1/1cylinder/Model/pod.pkl',

    
}


# 存放生成的参数
torch_params['temperature_uniform_data_name'] = torch_params['Docs'] + os.sep + 'temperature_uniform_data.npy'
torch_params['viscosity_uniform_data_name'] = torch_params['Docs'] + os.sep + 'viscosity_uniform_data.npy'
torch_params['t_vis_table_name'] = torch_params['Docs'] + os.sep + 'temp_visco_table_data.csv'

torch_params['msh_file'] = 'channel_2d.msh'
torch_params['flml_file'] ='lock_exchange.flml'
