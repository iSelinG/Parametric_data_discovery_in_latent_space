import os
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

from torch_params import torch_params
# from Process import *


class GenerateData(object):
	"""docstring for generate_data"""
	def __init__(self):
		super(GenerateData, self).__init__


	def mkdir(self, path_folder, name): 
		folder = path_folder + os.sep + name
		if not os.path.exists(folder):
			os.makedirs(folder)  
			print('The folder ', name, ' has been created.')
		else:
			print('The folder ', name, ' has already existed.')
			
	

	def copy_file(self, path, name):

		# copy files into the new folder
		src_msh_path = torch_params['source_path'] + os.sep + torch_params['msh_file'] # source file path
		dst_msh_path = path + os.sep + name + os.sep + torch_params['msh_file']# destination file path

		src_flml_path = torch_params['source_path'] + os.sep + torch_params['flml_file']
		dst_flml_path = path + os.sep + name + os.sep + torch_params['flml_file']

		if os.path.exists(src_msh_path) and os.path.exists(src_flml_path): # if two of them both exist
			shutil.copyfile(src_msh_path,dst_msh_path)
			shutil.copyfile(src_flml_path,dst_flml_path)
		print('The mesh file and flml file have been copied')
		return dst_flml_path


	def parse_xml(self, fn, t, v):

		# print(t,t)
		tree = ET.parse(fn)
		root = tree.getroot()


		temperature_root = root[-1][-1][0] # material_phase(fluid)->scalar_field(Temperature)->prognostic)
		viscosity_root = root[-1][2][0] # find the root of (material_phase(0)->vector_field(Velocity)->prognostic)


		for temperature in temperature_root[5].iter('string_value'):# (tensor_field(Viscosity)->prescribed->value->isotropic->constant)
			temperature.text = 'def val(X,t):\n if(X[0]<0.4):\n  return -' + str(t) + '\n else:\n  return ' + str(t)


		for viscosity in viscosity_root[9].iter('real_value'):# (tensor_field(Viscosity)->prescribed->value->isotropic->constant)
			viscosity.text = v # 0.5e-4
		
		tree.write(fn) # save the change

	def parse_xml_time(self, fn, dump_period_in_timesteps, start_time=0, end_time=40, timestep=0.005):
		'''
		
		'''
		tree = ET.parse(fn)
		root = tree.getroot()    

		
		dump_time_field = root[3][1]  # (io->dump_period)
		for dump_value in dump_time_field.iter('integer_value'):
			dump_value.text = str(dump_period_in_timesteps)


		timestepping_field = root[4]  # (timestepping->?)
		for current_time in timestepping_field[0].iter('real_value'):  # (timestepping->current_time)
			current_time.text = str(start_time)
		for timestep_value in timestepping_field[1].iter('real_value'):  # (timestepping->timestep)
			timestep_value.text = str(timestep)
		for finish_time in timestepping_field[2].iter('real_value'):
			finish_time.text = str(end_time)
		print('start_time: ', start_time, '; end_time: ', end_time, '; dt: ', torch_params['dt']) # , dump_period2)

		
		tree.write(fn) # save the change



	def get_random_number(self, t_name, v_name, t_v_name):

		'''
		This function is used to get the ramdom values of picked parametrics within the defined boundaries and save them as npy and csv files. 
		
		Flow
		----------
			Tell if random data files exist, if so, load them directly, if not:

				create two sorted random list for two parametrics;
				create new lists to save parameters sets; (conbined by both: a1->b1b2b3b4b5b6..,a2->b1b2b3b4b5b6...)
				Iteration two parametrics with variable i and j;
				convert lists to arrays with shape(2500,1);
				concat these three list->(2500,3);
				save values as npy and csv forms.

		Parameters
		----------
			t_name: name of temperature ramdom data list(npy)
			v_name: name of viscosity random data list(npy)
			t_v_name: name of parametrics groups (csv)

		Returns:
		----------
			out: temperature ramdom data array and viscosity random data array

		'''

		if os.path.exists(t_name) and os.path.exists(v_name):
			temperature= np.load(t_name) 
			viscosity = np.load(v_name)

		else:
			temperature = sorted(np.random.uniform(torch_params['temp_min'], torch_params['temp_max'], torch_params['temp_num']))
			viscosity_1 = sorted(np.random.uniform(torch_params['viso_min'], torch_params['viso_1st_ss'], torch_params['viso_num_ss']))
			viscosity_2 = sorted(np.random.uniform(torch_params['viso_1st_ss'], torch_params['viso_2nd_ss'], torch_params['viso_num_ss']))
			viscosity_3 = sorted(np.random.uniform(torch_params['viso_2nd_ss'], torch_params['viso_max'], torch_params['viso_num_ss']))
			viscosity = np.concatenate((viscosity_1, viscosity_2, viscosity_3), axis = 0)

			whole_list = np.zeros((len(temperature)*len(viscosity),2))
			n = 0
			for i in temperature:
				for j in viscosity:
					whole_list[n] = (i,j)
					n += 1

			np.save(t_name , temperature)
			np.save(v_name ,viscosity)
			np.save(t_v_name, whole_list)

			# whole_list = np.load(t_v_name)
			data_pd = pd.DataFrame(whole_list)
			data_pd.to_csv(t_v_name[:-4] + '.csv', header = ['Temperature','Viscosity'], index = False)
		
		return temperature, viscosity
	
	def get_uniform_number(self, t_name, v_name, t_v_name):
		'''
        This function is used to get the ramdom values of picked parametrics within the defined boundaries and save them as npy and csv files. 
        
        Flow
        ----------
            Tell if random data files exist, if so, load them directly, if not:

                create two sorted ramdom list for two parametrics;
                create new lists to save parameters sets; (conbined by both: a1->b1b2b3b4b5b6..,a2->b1b2b3b4b5b6...)
                Iteration two parametrics with variable i and j;
                convert lists to arrays with shape(2500,1);
                concat these three list->(2500,3);
                save values as npy and csv forms.

        Parameters
        ----------
            t_name: name of temperature ramdom data list(npy)
            v_name: name of viscosity random data list(npy)
            t_v_name: name of parametrics groups (csv)

        Returns:
        ----------
            out: temperature ramdom data array and viscosity random data array

		'''
		if os.path.exists(t_name) and os.path.exists(v_name):
			temperature= np.load(t_name) 
			viscosity = np.load(v_name)
		else:
			temperature = sorted(np.linspace(torch_params['temp_min'], torch_params['temp_max'], torch_params['temp_num']))
			# viscosity_1 = sorted(np.linspace(torch_params['viso_min'], torch_params['viso_1st_ss'], torch_params['viso_num_ss']))[:-1]
			# viscosity_2 = sorted(np.linspace(torch_params['viso_1st_ss'], torch_params['viso_2nd_ss'], torch_params['viso_num_ss']))[:-1]
			viscosity_3 = sorted(np.linspace(torch_params['viso_2nd_ss'], torch_params['viso_max'], torch_params['viso_num_ss']))[:-1]
			# viscosity = np.concatenate((viscosity_1, viscosity_2, viscosity_3), axis = 0)
			# viscosity = np.concatenate((viscosity_2, viscosity_3), axis = 0)
			viscosity = viscosity_3

			whole_list = np.zeros((len(temperature)*len(viscosity),2))
			n = 0
			for i in temperature:
				for j in viscosity:
					whole_list[n] = (i,j)
					n += 1

			# np.save(t_name , temperature)
			np.save(v_name ,viscosity)
			np.save(t_v_name, whole_list)

			# whole_list = np.load(t_v_name)
			data_pd = pd.DataFrame(whole_list)
			data_pd.to_csv(t_v_name[:-4] + '.csv', header = ['Temperature','Viscosity'], index = False)

		return temperature, viscosity

	def generate_data_command(self, m, n, path):
		# By default path = params['datasets_folder']

		m =format(float(m), '.6f') # Temperature
		n =format(float(n), '.10f') # viscosity
		print('the temperature is equal to ' + str(m), 'The viscosity is equal to ' + str(n))

		name = 'temperature_' + str(m) +'_viscosity_' + str(n)

		# mkdir(path + os.sep + name)
		self.mkdir(path, name)
		dst_flml_path = self.copy_file(path, name)
		self.parse_xml(dst_flml_path, m, n)#t,v
		self.parse_xml_time(dst_flml_path, torch_params['dump_period_in_timesteps'], torch_params['start_time'], torch_params['end_time'], torch_params['timestep'])

		print('enter the folder', path)
		commend = 'cd "' + path + os.sep + name + '" && fluidity ' + torch_params['flml_file']
		os.system(commend)
		print('vtu files have been generated.')




	def copy_folder(self, path_1, path_2):
		self.mkdir(torch_params['model_save_folder'], '')	
		try:	
			shutil.copytree(path_1, path_2)
		except FileExistsError:
			shutil.rmtree(path_2)
			shutil.copytree(path_1, path_2)


def generate_data(vtu_path=torch_params['datasets_folder']):

	'''
	This function is used to generate vtu data groups based on generated ranged random values of two parameters. 
	
	Flow
	----------
		load parameters data from function 'get_random_number'
		nested for loop to create the folder with specific name; 
		copy required files in the folder and update its 'flml' file with selected parameters values;
		run commends to generate specific data groups

	Notes:
	----------
		.10f: expressed to ten decimal places

	'''
	generator = GenerateData()
	# temperature, viscosity = generator.get_random_number(params['temperature_random_data_name'], 
	# 												params['viscosity_random_data_name'], 
	# 												params['t_vis_table_name'])
	temperature, viscosity = generator.get_uniform_number(torch_params['temperature_uniform_data_name'], 
													torch_params['viscosity_uniform_data_name'], 
													torch_params['t_vis_table_name'])

	# temperature[:5], temperature[5:10], temperature[10:15], temperature[15:]
	for m in temperature:
		for n in viscosity:
			print(m,n)
			generator.generate_data_command(m, n, vtu_path)
	# for m in temperature:
	# 	for n in viscosity:
	# 		print(m,n)
	
	# print('creat a generated data groups folder copy as reconstruct data saving folder. ')
	# generator.copy_folder(torch_params['datasets_folder'], torch_params['reconstructed_data_folder'])
	

if __name__ == '__main__':

	generate_data(torch_params['datasets_folder_test'])
