#In this script we write the code to perform the interpolation of the input data tensor 
#The interpolation is made wrt to depths and the variable of interest (physical vars, oxygen, chlorophyll, etc)


import numpy as np
import torch
import os

#creo il device per le GPU e porto i dati su GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device", device)


def tensor_interpolation(tensor):
    """this function interpolates the variables in the missing depths"""
    """it works with a resolution of at most 5 meters (not higher, as 2 meters)"""
    interp_tensor = torch.clone(tensor)
    index_depths = np.arange(0, tensor.shape[2])
    for i_d in index_depths[1:-2]:    #perchè gli ultimi layer prima dei 300 metri sono entrambi nulli
        if torch.sum(interp_tensor[:, :, i_d, :, :]) == 0.0 and torch.sum(interp_tensor[:, :, i_d + 1, :, :]) != 0:
            for i_h in range(tensor.shape[3]):
                for i_w in range(tensor.shape[4]):
                    interp_tensor[:,:, i_d, i_h, i_w] = (interp_tensor[:,:, i_d - 1, i_h, i_w] + interp_tensor[:,:, i_d + 1, i_h, i_w]) / 2
        elif torch.sum(interp_tensor[:, :, i_d, :, :]) == 0.0 and torch.sum(interp_tensor[:, :, i_d + 1, :, :]) == 0:
            for i_h in range(tensor.shape[3]):
                for i_w in range(tensor.shape[4]):
                    interp_tensor[:,:, i_d, i_h, i_w] = (2 * interp_tensor[:,:, i_d - 1, i_h, i_w] + interp_tensor[:,:, i_d + 2, i_h, i_w]) / 3
                    interp_tensor[:,:, i_d + 1, i_h, i_w] = (interp_tensor[:,:, i_d - 1, i_h, i_w] + 2 * interp_tensor[:,:, i_d + 2, i_h, i_w]) / 3
    return interp_tensor



def tensor_interpolation_float(tensor):
    """this function interpolates the variables in the missing depths"""
    """it works with a resolution of at most 5 meters (not higher, as 2 meters)"""
    interp_tensor = torch.clone(tensor)
    index_depths = np.arange(0, tensor.shape[2])
    #select the coordinate in which there is the float measure: 
    for i_h in range(tensor.shape[3]):
        for i_w in range(tensor.shape[4]):
            if torch.count_nonzero(tensor[:, :, :, i_h, i_w]) > 20:
                print("interpolation float start")
                for i_d in index_depths[1:-2]: 
                    if torch.sum(interp_tensor[:, :, i_d, i_h, i_w]) == 0.0 and torch.sum(interp_tensor[:, :, i_d + 1, i_h, i_w]) != 0:
                        print("inside if")
                        interp_tensor[:,:, i_d, i_h, i_w] = (interp_tensor[:,:, i_d - 1, i_h, i_w] + interp_tensor[:,:, i_d + 1, i_h, i_w]) / 2
                    elif torch.sum(interp_tensor[:, :, i_d, :, :]) == 0.0 and torch.sum(interp_tensor[:, :, i_d + 1, :, :]) == 0:
                        print("insid elif")
                        interp_tensor[:,:, i_d, i_h, i_w] = (2 * interp_tensor[:,:, i_d - 1, i_h, i_w] + interp_tensor[:,:, i_d + 2, i_h, i_w]) / 3
                        interp_tensor[:,:, i_d + 1, i_h, i_w] = (interp_tensor[:,:, i_d - 1, i_h, i_w] + 2 * interp_tensor[:,:, i_d + 2, i_h, i_w]) / 3
    return interp_tensor



def directory_tensor_interpolation(directory_of_tensors, directory_of_interp_tensors, var):
    """this function recallls the tensor_interpolation and interpolates an entire directory of tensors anmd saves them in another directory"""
    list_tensor_ptFiles = os.listdir(directory_of_tensors + "/" + var)
    for tensor_ptFile in list_tensor_ptFiles:
        data_time = tensor_ptFile[9:16]
        data_time = "".join(c for c in data_time if c.isdecimal()) 
        print("data_time", data_time)
        my_tensor = torch.load(directory_of_tensors + "/" + var + "/" + tensor_ptFile, map_location=torch.device('cpu'))
        interp_tensor = tensor_interpolation(my_tensor)
        torch.save(interp_tensor, directory_of_interp_tensors + "/" + var + "/datetime_" + str(data_time) + ".pt")
    return None


def directory_tensor_interpolation_float(directory_of_tensors, directory_of_interp_tensors, var):
    """this function recallls the tensor_interpolation and interpolates an entire directory of tensors anmd saves them in another directory"""
    list_tensor_ptFiles = os.listdir(directory_of_tensors + "/" + var)
    for tensor_ptFile in list_tensor_ptFiles:
        data_time = tensor_ptFile[9:16]
        data_time = "".join(c for c in data_time if c.isdecimal()) 
        data_time = data_time[:4] + "_" + data_time[4:]
        print("data_time", data_time)
        my_tensor = torch.load(directory_of_tensors + "/" + var + "/" + tensor_ptFile, map_location=torch.device('cpu'))
        interp_tensor = tensor_interpolation_float(torch.from_numpy(my_tensor))
        torch.save(interp_tensor, directory_of_interp_tensors + "/" + var + "/datetime_" + str(data_time) + ".pt")
    return None


#directory_tensor_interpolation("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/MODEL/2022/tensor", "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/MODEL/2022/interp_tensor", "P_l")
#directory_tensor_interpolation("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/MODEL/2022/tensor", "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/MODEL/2022/interp_tensor", "O2o")
#directory_tensor_interpolation("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/MODEL/2022/tensor", "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/MODEL/2022/interp_tensor", "soshfldo")
#directory_tensor_interpolation("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/MODEL/2022/tensor", "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/MODEL/2022/interp_tensor", "sossheig")
#directory_tensor_interpolation("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/MODEL/2022/tensor", "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/MODEL/2022/interp_tensor", "vomecrty")
#directory_tensor_interpolation("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/MODEL/2022/tensor", "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/MODEL/2022/interp_tensor", "vosaline")
#directory_tensor_interpolation("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/MODEL/2022/tensor", "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/MODEL/2022/interp_tensor", "votemper")
#directory_tensor_interpolation("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/MODEL/2022/tensor", "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/MODEL/2022/interp_tensor", "vozocrtx")

directory_tensor_interpolation_float("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/float/2022/tensor", "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/float/2022/interp_tensor", "P_l")
directory_tensor_interpolation_float("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/float/2021/tensor", "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/float/2021/interp_tensor", "P_l")
directory_tensor_interpolation_float("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/float/2020/tensor", "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/float/2020/interp_tensor", "P_l")
directory_tensor_interpolation_float("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/float/2019/tensor", "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/CNN_reconstruction_final_resolution/dataset/float/2019/interp_tensor", "P_l")