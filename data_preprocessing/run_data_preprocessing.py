"""
In this script the pypeline to generate the tensors for the training is presented
"""
from hyperparameter import *
from plot_save_tensor import Save_Tensor, save_routine, save_tensors_directory
from interpolation import directory_tensor_interpolation, directory_tensor_interpolation_float


type_tensor = "model"
variable = "votemper"
year = (2023, 2024)
npy_directory = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/MODEL/2023/saved_parallelepiped_physics_single/votemper/"
pt_directory = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/"

save_tensors_directory(type_tensor, npy_directory, year, "t", variable, pt_directory)
directory_tensor_interpolation(pt_directory + "MODEL/" + str(year[0]) + "/tensor", pt_directory + "MODEL/" + str(year[0]) + "/interp_tensor", variable)
