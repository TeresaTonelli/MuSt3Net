"""
In this script the pypeline to generate the tensors for the training is presented.
By selecting a type (model or float), a variable (chlorophyll, temperature, etc...) and a year ((2019, 2020), (2020, 2021), etc...), 
you can generate the final weekly tensor relative to that variabe in that year
"""
import numpy as np
import os

from hyperparameter import *
from data_preprocessing.plot_save_tensor import Save_Tensor, save_routine, save_tensors_directory
from data_preprocessing.interpolation import directory_tensor_interpolation, directory_tensor_interpolation_float
from data_preprocessing.make_dataset_single_var import routine_insert_model_biog, routine_insert_model_physics
from data_preprocessing.make_dataset_float import routine_insert_float_biog, create_list_date_time


type_tensors = ["model", "float"]
variables = list_physics_vars + list_biogeoch_vars
years = year_interval
data_directory = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/"
float_path = "/leonardo_scratch/large/userexternal/ttonelli/OGS/SUPERFLOAT/SUPERFLOAT"


def routine_generation_input_dataset(type_data, variable, year, path_phys, path_biogeoch, data_directory, float_path):
    """function that geenrates the input dataset starting from the data in .nc files"""
    if type_data in type_tensors and variable in variables and year in years:
        match type_data:
            case "model":
                if variable in list_physics_vars:
                    routine_insert_model_physics(variable, path_phys, year, resolution)
                    save_tensors_directory(type_data, data_directory + str(type_data) + "/" + str(year) + "/saved_parallelepiped_physics_single/" + str(variable) + "/", year, "t", variable, data_directory)
                    directory_tensor_interpolation(data_directory + "MODEL/" + str(year) + "/tensor", data_directory + "MODEL/" + str(year) + "/interp_tensor", variable)            
                elif variable in list_biogeoch_vars:
                    routine_insert_model_biog(variable, path_biogeoch, year, resolution)
                    save_tensors_directory(type_data, data_directory + str(type_data) + "/" + str(year) + "/saved_parallelepiped_biogeoch_single/" + str(variable) + "/", year, "t", variable, data_directory)
                    directory_tensor_interpolation(data_directory + "MODEL/" + str(year) + "/tensor", data_directory + "MODEL/" + str(year) + "/interp_tensor", variable)
            case "float":
                if variable in list_biogeoch_vars:
                    list_data_times = [create_list_date_time((year_interval[i], year_interval[i+1])) for i in range(len(year_interval) - 1)]
                    routine_insert_float_biog(float_path, resolution)
                    list_vars_list_parallelepiped = np.load(os.getcwd() + "/dataset/float/" + str(year) + "/saved_parallelepiped_biogeoch/" + str(variable) + "_parallelepiped.npy", allow_pickle=True)
                    new_list_vars_list_parallelepiped = [list_vars_list_parallelepiped[j, :, :, :, :, :] for j in range(len(list_vars_list_parallelepiped))]
                    save_routine(kindof, new_list_vars_list_parallelepiped, list_data_times[-1], (year, int(year)+1), "t", variable, data_directory)
                    directory_tensor_interpolation_float(data_directory + str(type_data) + "/" + str(year) + "/tensor", data_directory + str(type_data) + "/" + str(year) + "/interp_tensor", variable)


#Example routine for a single variable and for 1 year
type_data = "model"
variable = "vosaline"
year = 2024
if variable in list_physics_vars:
    path_phys = "/leonardo_scratch/large/userexternal/ttonelli/OGS/WEEKLY/" + str(year) + '/' + variable + '/' 
elif variable in list_biogeoch_vars:
    path_biogeoch = path_biogeoch = "/leonardo_scratch/large/userexternal/ttonelli/OGS/AVE_FREQ_2/" + str(year) + "/" + variable + "/"
routine_generation_input_dataset(type_data, variable, year, path_phys, path_biogeoch, data_directory, float_path)