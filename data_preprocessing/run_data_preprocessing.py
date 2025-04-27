"""
In this script the pypeline to generate the tensors for the training is presented.
By selecting a type (model or float), a variable (chlorophyll, temperature, etc...) and a year ((2019, 2020), (2020, 2021), etc...), 
you can generate the final weekly tensor relative to that variabe in that year
"""
from hyperparameter import *
from data_preprocessing.plot_save_tensor import Save_Tensor, save_routine, save_tensors_directory
from data_preprocessing.interpolation import directory_tensor_interpolation, directory_tensor_interpolation_float
from data_preprocessing.make_dataset_single_var import routine_insert_model_biog, routine_insert_model_physics
from data_preprocessing.make_dataset_float import routine_insert_float_biog


type_tensors = ["model", "float"]
variables = list_physics_vars + list_biogeoch_vars
years = year_interval
npy_directory = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/MODEL/2023/saved_parallelepiped_physics_single/votemper/"
pt_directory = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/"


def routine_generation_input_dataset(type_data, variable, year, path_phys, path_biogeoch, npy_directory, pt_directory):
    """it ..."""
    if type_data in type_tensors and variable in variables and year in years:
        match type_data:
            case "model":
                if variable in list_physics_vars:
                    routine_insert_model_physics(variable, path_phys, year, resolution)
                    save_tensors_directory(type_data, npy_directory, year, "t", variable, pt_directory)
                    directory_tensor_interpolation(pt_directory + "MODEL/" + str(year) + "/tensor", pt_directory + "MODEL/" + str(year) + "/interp_tensor", variable)
                elif variable in list_biogeoch_vars:
                    routine_insert_model_biog(variable, path_biogeoch, year, resolution)
                    save_tensors_directory(type_data, npy_directory, year, "t", variable, pt_directory)
                    directory_tensor_interpolation(pt_directory + "MODEL/" + str(year) + "/tensor", pt_directory + "MODEL/" + str(year) + "/interp_tensor", variable)
            case "float":
                if variable in list_physics_vars:

                elif variable in list_biogeoch_vars:

            
    
    
    
routine_insert_model_physics()
save_tensors_directory(type_tensor, npy_directory, year, "t", variable, pt_directory)
directory_tensor_interpolation(pt_directory + "MODEL/" + str(year[0]) + "/tensor", pt_directory + "MODEL/" + str(year[0]) + "/interp_tensor", variable)
