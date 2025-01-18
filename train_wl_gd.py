"""
Implementation of the training routine for the 3D CNN with GAN
- train_dataset : list/array of 5D (or 5D ?) tensor in form (bs, input_channels, D_in, H_in, W_in)
"""
import numpy as np
import torch.nn as nn
from torch.optim import Adadelta
import matplotlib.pyplot as plt
import random
import datetime

from convolutional_network import CompletionN
from losses import convolutional_network_weighted_loss, convolutional_network_float_weighted_loss, convolutional_network_exp_weighted_loss, convolutional_network_float_exp_weighted_loss
from mean_pixel_value import MV_pixel
from utils_mask import generate_input_mask, generate_sea_land_mask
from normalization import Normalization
from denormalization import Denormalization
from get_dataset import *
from plot_error import Plot_Error
from plot_results import *
from utils_function import *
from utils_mask import generate_float_mask, compute_exponential_weights
from generation_training_dataset import generate_training_dataset_phase_1, generate_training_dataset_phase_2, generate_multiple_dataset_phase_1, generate_multiple_dataset_phase_2, generate_dataset_phase_1_saving

num_channel = number_channel  
name_datetime_folder = str(datetime.datetime.utcnow())

path_job = "results_job_" + name_datetime_folder
if not os.path.exists(path_job):
    print("new path created")
    os.mkdir(path_job)

path_results = path_job + "/results_training_1"           
if not os.path.exists(path_results):                   
    os.mkdir(path_results)

path_mean_std = path_results + "/mean_and_std_tensors"
if not os.path.exists(path_mean_std):
    os.makedirs(path_mean_std)

path_land_sea_masks = path_results + "/land_sea_masks"
if not os.path.exists(path_land_sea_masks):
    os.makedirs(path_land_sea_masks)

#creo il device per le GPU e porto i dati su GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device", device)


biogeoch_var_to_predict = list_biogeoch_vars[0]



#generate_dataset_phase_1_saving(biogeoch_var_to_predict, path_results, 2019, "dataset_training")
#generate_dataset_phase_1_saving(biogeoch_var_to_predict, path_results, 2020, "dataset_training")
#generate_dataset_phase_1_saving(biogeoch_var_to_predict, path_results, 2021, "dataset_training")
#generate_dataset_phase_1_saving(biogeoch_var_to_predict, path_results, 2022, "dataset_training")

#PHASE 1  --> MODEL data
#generate the new train and test data

#biogeoch_train_dataset, land_sea_masks, old_total_dataset, index_training, index_internal_testing, index_external_testing, train_dataset, internal_test_dataset, external_test_dataset, list_duplicates_biogeoch_weeks, transposed_duplicates_biogeoch_coordinates = generate_training_dataset_phase_1(list_biogeoch_vars[0], path_results)
biogeoch_train_dataset, land_sea_masks, old_total_dataset, index_training, index_internal_testing, index_external_testing, train_dataset, internal_test_dataset, external_test_dataset, list_duplicates_biogeoch_weeks, transposed_duplicates_biogeoch_coordinates = generate_multiple_dataset_phase_1(list_biogeoch_vars[0], path_results, [2019, 2020])
print("end generation data 1")

# HYPERPARAMETERS
pretrain = 1 
pretrain = 0
epoch_pretrain = 0


model_completion = CompletionN()    


alpha = torch.tensor(4e-4)         
lr_c = 0.001           
epoch_c = 400  #200
snaperiod = 50   #25

exp_weights = compute_exponential_weights(d, depth_interval[1], superficial_bound_depth)



#SCRIVIAMO I NUOVI PATH IN CUI SALVARE TUTTI I RISULTATI CHE OTTENIAMO
path_configuration = path_results + "/" + str(biogeoch_var_to_predict) + "/" + str(epoch_c + epoch_pretrain) 
if not os.path.exists(path_configuration):
    os.makedirs(path_configuration)
path_lr = path_configuration + "/lrc_" + str(lr_c) 
if not os.path.exists(path_lr):
    os.makedirs(path_lr)
path_losses = path_lr + "/losses"
if not os.path.exists(path_losses):
    os.makedirs(path_losses)
path_model = path_lr + "/partial_models/"
if not os.path.exists(path_model):
    os.mkdir(path_model)
path_plots = path_lr + "/plots"
if not os.path.exists(path_plots):
    os.makedirs(path_plots)

losses_1_c = []
losses_1_c_test = []
test_losses = []

# PHASE 1   #la phase 1 del training consiste nel trainare solo la Completion

optimizer_completion =torch.optim.Adam(model_completion.parameters(), lr=lr_c)   #questo è l'optimizer che uso per questa rete --> che implementa praticamente una stochastic gradient descend

f, f_test = open(path_losses + "/train_loss.txt", "w+"), open(path_losses + "/test_loss.txt", "w+")

my_mean_tensor = torch.unsqueeze(torch.load(path_mean_std + "/mean_tensor.pt")[:, 6, :, :, :], 1).to(device)
my_std_tensor = torch.unsqueeze(torch.load(path_mean_std + "/std_tensor.pt")[:, 6, :, :, :], 1).to(device)

print("len old dataset", len(old_total_dataset))
print("start train")
for ep in range(epoch_c):
    if ep > 400:   
        lr_c = 0.0001

    #PHASE OF TRAINING
    for i in range(len(train_dataset)):
        training_x = train_dataset[i]
        training_x = training_x.to(device)
        input = training_x
        model_completion = model_completion.to(device)
        output = model_completion(input.float())

        #find denormalized tensors to compute the loss
        
        #my_mean_tensor = torch.unsqueeze(torch.load(path_mean_std + "/mean_tensor.pt")[:, 6, :, :, :], 1).to(device)
        #my_std_tensor = torch.unsqueeze(torch.load(path_mean_std + "/std_tensor.pt")[:, 6, :, :, :], 1).to(device)
        denormalized_NN_output = Denormalization(output, my_mean_tensor, my_std_tensor).to(device)

        print("len input list", len(old_total_dataset))

        biog_input = old_total_dataset[int(index_training[i] / n_duplicates_biogeoch)].to(device)
        loss_completion = convolutional_network_exp_weighted_loss(biog_input[:, :, :-1, :, 1:-1].float(), denormalized_NN_output.float(), land_sea_masks, exp_weights.to(device))
        print("loss completion", loss_completion)
        losses_1_c.append(loss_completion.item())

        print(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_completion.item():.12f}")
        f.write(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_completion.item():.12f} \n")

        optimizer_completion.zero_grad()
        loss_completion.backward()
        optimizer_completion.step()

        #remove the data from the gpu
        del training_x
        del output
        del biog_input
        torch.cuda.empty_cache()
    

    #PHASE OF TESTING
    if (ep+1) % snaperiod == 0:
        model_completion.eval()

        losses_1_c_test = []
        with torch.no_grad():

            for i_test in range(len(internal_test_dataset)): 
                test_data = internal_test_dataset[i_test] 
                test_data = test_data.to(device)   
                testing_input = test_data
                testing_output = model_completion(testing_input.float())

                denormalized_testing_output = Denormalization(testing_output, my_mean_tensor, my_std_tensor).to(device)
                biog_input = old_total_dataset[int(index_internal_testing[i_test] / n_duplicates_biogeoch)].to(device)
                loss_1c_test = convolutional_network_exp_weighted_loss(biog_input[:, :, :-1, :, 1:-1].float(), denormalized_testing_output.float(), land_sea_masks, exp_weights)
                losses_1_c_test.append(loss_1c_test.cpu())

                print(f"[EPOCH]: {ep + 1}, [TEST LOSS]: {loss_1c_test.item():.12f}")
                f_test.write(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_1c_test.item():.12f} \n")

                del test_data
                del denormalized_testing_output
                del biog_input
                torch.cuda.empty_cache()

            print("len test_1_c_", len(losses_1_c_test))
            test_loss = np.mean(np.array(losses_1_c_test))      
            test_losses.append(test_loss)


    if (ep+1) % snaperiod == 0: 
        torch.save(model_completion.state_dict(), path_model + "/ep_" + str(ep + epoch_pretrain) + ".pt")

f.close()
f_test.close()

print("len training dataset", len(train_dataset))
print("len internal test dataset", len(internal_test_dataset))
print("len old dataset", len(old_total_dataset))


Plot_Error(test_losses, "1p", path_losses + "/")

torch.save(model_completion.state_dict(), path_lr + "/final_model" + ".pt")



#Plot the model evaluation on the test data --> with 3 different plots
var = biogeoch_var_to_predict
path_profiles = path_plots + "/profiles_1p"
path_NN_reconstruction = path_plots + "/NN_maps"
path_BFM_reconstruction = path_plots + "/BFM_maps"
model_completion.eval()
with torch.no_grad():
    for i in range(len(external_test_dataset)):
        test_data = external_test_dataset[i]
        test_data = test_data.to(device)
        testing_input = test_data
        testing_output = model_completion(testing_input.float())

        denorm_testing_input = Denormalization(testing_input, my_mean_tensor, my_std_tensor)
        denorm_testing_output = Denormalization(testing_output, my_mean_tensor, my_std_tensor)

        path_profiles_test_data = path_profiles + "/test_data_week_" + str(list_duplicates_biogeoch_weeks[index_external_testing[i]])
        if not os.path.exists(path_profiles_test_data):
            os.makedirs(path_profiles_test_data)
        path_NN_reconstruction_test_data = path_NN_reconstruction + "/test_data_week_" + str(list_duplicates_biogeoch_weeks[index_external_testing[i]]) 
        if not os.path.exists(path_NN_reconstruction_test_data):
            os.makedirs(path_NN_reconstruction_test_data)
        path_BFM_reconstruction_test_data = path_BFM_reconstruction + "/test_data_week_" + str(list_duplicates_biogeoch_weeks[index_external_testing[i]])
        if not os.path.exists(path_BFM_reconstruction_test_data):
            os.makedirs(path_BFM_reconstruction_test_data)
        
        plot_models_profiles_1p(torch.unsqueeze(denorm_testing_input[:, 6, :, :, :], 1), denorm_testing_output, torch.unsqueeze(old_total_dataset[int(index_external_testing[i] / n_duplicates_biogeoch)][:, :, :-1, :, 1:-1][:, 6, :, :, :], 1),  
                             var, path_profiles_test_data, transposed_duplicates_biogeoch_coordinates[index_external_testing[i]]) 
        plot_NN_maps(denorm_testing_output, land_sea_masks, var, path_NN_reconstruction_test_data)
        plot_NN_maps(torch.unsqueeze(old_total_dataset[int(index_external_testing[i] / n_duplicates_biogeoch)][:, :, :-1, :, 1:-1][:, 6, :, :, :], 1), land_sea_masks, var, path_BFM_reconstruction_test_data)

        #remove all the tensors from the gpu 
        del test_data
        torch.cuda.empty_cache()


#remove the mean and standard deviation from the gpu
del my_mean_tensor
del my_std_tensor
torch.cuda.empty_cache()








#PHASE 2

#SCRIVIAMO I NUOVI PATH IN CUI SALVARE TUTTI I RISULTATI CHE OTTENIAMO
# HYPERPARAMETERS
print("start 2 phase")
pretrain_2 = 1  # 0 means that we don"t use pretrained model to fine tuning
pretrain_2 = 0
epoch_pretrain_2 = 0

alpha_2 = torch.tensor(4e-4)          
lr_c_2 = 0.001           
epoch_c_2 = 40  #20 
snaperiod_2 = 10
path_results_2 = path_job + "/results_training_2" 
path_configuration_2 = path_results_2 + "/" + str(biogeoch_var_to_predict) + "/" + str(epoch_c_2 + epoch_pretrain_2) 
if not os.path.exists(path_configuration_2):
    os.makedirs(path_configuration_2)
path_mean_std_2 = path_results_2 + "/mean_and_std_tensors"
if not os.path.exists(path_mean_std_2):
    os.makedirs(path_mean_std_2)
path_lr_2 = path_configuration_2 + "/lrc_" + str(lr_c_2) 
if not os.path.exists(path_lr_2):
    os.makedirs(path_lr_2)
path_losses_2 = path_lr_2 + "/losses"
if not os.path.exists(path_losses_2):
    os.makedirs(path_losses_2)
path_model_2 = path_lr_2 + "/partial_models/"
if not os.path.exists(path_model_2):
    os.mkdir(path_model_2)
path_plots_2 = path_lr_2 + "/plots"
if not os.path.exists(path_plots_2):
    os.makedirs(path_plots_2)

#generation dataset for the 2 phaase
#float_list_weeks, denorm_old_float_train_dataset, list_float_profiles_coordinates, sampled_list_float_profile_coordinates, index_training_2, index_internal_testing_2, index_external_testing_2, train_dataset_2, internal_test_dataset_2, external_test_dataset_2  = generate_training_dataset_phase_2(list_biogeoch_vars[0], land_sea_masks, path_results_2)
float_list_weeks, denorm_old_float_train_dataset, list_float_profiles_coordinates, sampled_list_float_profile_coordinates, index_training_2, index_internal_testing_2, index_external_testing_2, train_dataset_2, internal_test_dataset_2, external_test_dataset_2  = generate_multiple_dataset_phase_2(list_biogeoch_vars[0], land_sea_masks, path_results_2, [2019, 2020])

model_completion_1_load = CompletionN()    
model_completion_1_load.load_state_dict(torch.load(path_lr + "/final_model" + ".pt", map_location=torch.device('cpu')))
model_completion_1_load.eval()  


model_completion_2 = CompletionN()    
model_completion_2.load_state_dict(torch.load(path_lr + "/final_model" + ".pt"))
model_completion_2.eval()     


losses_1_c_2 = []
losses_1_c_test_2 = []
test_losses_2 = []

my_mean_tensor_2 = torch.unsqueeze(torch.load(path_mean_std_2 + "/mean_tensor.pt")[:,6, :,:,:], 1).to(device)
my_std_tensor_2 = torch.unsqueeze(torch.load(path_mean_std_2 + "/std_tensor.pt")[:,6,:,:,:], 1).to(device)


optimizer_completion_2 =torch.optim.Adam(model_completion_2.parameters(), lr=lr_c_2)   #questo è l'optimizer che uso per questa rete --> che implementa praticamente una stochastic gradient descend

f_2, f_test_2 = open(path_losses_2 + "/train_loss.txt", "w+"), open(path_losses_2 + "/test_loss.txt", "w+")

for ep in range(epoch_c_2):
    if ep > 400:     
        lr_c_2 = 0.0001

    #PHASE OF TRAINING
    for i in range(len(train_dataset_2)):
        training_x = train_dataset_2[i]
        training_x = training_x.to(device)
        input = training_x  
        model_completion_2 = model_completion_2.to(device)
        output = model_completion_2(input.float())


        #ora calcolo la loss solo sui float --> solo sui profili campionati 
        denormalized_output = Denormalization(output, my_mean_tensor_2, my_std_tensor_2).to(device)
        float_tensor_input = denorm_old_float_train_dataset[index_training_2[i]].to(device)
        float_coord_mask = generate_float_mask(sampled_list_float_profile_coordinates[index_training_2[i]]).to(device)
        loss_completion_2 = convolutional_network_float_exp_weighted_loss(float_tensor_input.float(), denormalized_output.float(), land_sea_masks, float_coord_mask, exp_weights.to(device))
        print("loss completion", loss_completion_2)
        losses_1_c.append(loss_completion_2.item())

        print(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_completion_2.item():.12f}")
        f_2.write(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_completion_2.item():.12f} \n")

        optimizer_completion_2.zero_grad()
        loss_completion_2.backward()
        optimizer_completion_2.step()

        del training_x
        del denormalized_output
        del float_tensor_input
        del float_coord_mask
        torch.cuda.empty_cache()


    #PHASE OF TESTING
    if (ep+1) % snaperiod_2 == 0:
        model_completion_2.eval()

        losses_1_c_test_2 = []
        with torch.no_grad():

            for i_test in range(len(internal_test_dataset_2)):
                test_data = internal_test_dataset_2[i_test]  
                test_data = test_data.to(device)   
                testing_input = test_data  
                testing_output = model_completion_2(testing_input.float())

                denormalized_output = Denormalization(testing_output, my_mean_tensor_2, my_std_tensor_2).to(device)
                float_tensor_input = denorm_old_float_train_dataset[index_internal_testing_2[i_test]].to(device)
                float_coord_mask = generate_float_mask(sampled_list_float_profile_coordinates[index_internal_testing_2[i_test]]).to(device)
                loss_1c_test_2 = convolutional_network_float_exp_weighted_loss(float_tensor_input.float(), denormalized_output.float(), land_sea_masks, float_coord_mask, exp_weights.to(device))
                losses_1_c_test_2.append(loss_1c_test_2.cpu())

                print(f"[EPOCH]: {ep + 1}, [TEST LOSS]: {loss_1c_test_2.item():.12f}")
                f_test_2.write(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_1c_test_2.item():.12f} \n")

                del test_data
                del denormalized_output
                del float_tensor_input
                del float_coord_mask
                torch.cuda.empty_cache()

            test_loss_2 = np.mean(np.array(losses_1_c_test_2))      
            test_losses_2.append(test_loss_2)


    if (ep+1) % snaperiod_2 == 0:  
        torch.save(model_completion_2.state_dict(), path_model_2 + "/ep_" + str(ep + epoch_pretrain_2) + ".pt")

f_2.close()
f_test_2.close()

Plot_Error(test_losses_2, "2p", path_losses_2 + "/")

torch.save(model_completion_2.state_dict(), path_lr_2 + "/final_model" + ".pt")



#Plots of the 2 phase
path_profiles_with_NN_1 = path_plots_2 + "/profiles_2p_NN_1"
path_NN_reconstruction = path_plots_2 + "/NN_maps"
path_NN_phases_diff = path_plots_2 + "/NN_phases_diff"
model_completion_2.eval()
with torch.no_grad():
    for i in range(len(external_test_dataset_2)):
        print("new test data")
        test_data_2 = external_test_dataset_2[i]
        test_data_2 = test_data_2.to(device)
        testing_input_2 = test_data_2 
        testing_output_2 = model_completion_2(testing_input_2.float())

        denorm_testing_output_2 = Denormalization(testing_output_2, my_mean_tensor_2, my_std_tensor_2)
        norm_testing_output_1 = model_completion_1_load(testing_input_2.cpu().float()).to(device) 
        denorm_testing_output_1 = Denormalization(norm_testing_output_1, my_mean_tensor_2, my_std_tensor_2) 

        path_profiles_test_data_NN_1 = path_profiles_with_NN_1 + "/test_data_" + str(i) + "_week_" + str(float_list_weeks[index_external_testing_2[i]])
        if not os.path.exists(path_profiles_test_data_NN_1):
            os.makedirs(path_profiles_test_data_NN_1)
        path_NN_reconstruction_test_data = path_NN_reconstruction + "/test_data_" + str(i) + "_week_" + str(float_list_weeks[index_external_testing_2[i]])
        if not os.path.exists(path_NN_reconstruction_test_data):
            os.makedirs(path_NN_reconstruction_test_data)
        path_NN_diff_test_data = path_NN_phases_diff + "/test_data_" + str(i) + "_week_" + str(float_list_weeks[index_external_testing_2[i]])
        if not os.path.exists(path_NN_diff_test_data):
            os.makedirs(path_NN_diff_test_data)
        
        comparison_profiles_1_2_phases(torch.unsqueeze(denorm_old_float_train_dataset[index_external_testing_2[i]][:, 6, :, :, :], 1) , denorm_testing_output_2, biogeoch_train_dataset[index_external_testing_2[i]][:, :, :-1, :, 1:-1], denorm_testing_output_1,
                             var, path_profiles_test_data_NN_1)
        plot_NN_maps(denorm_testing_output_2, land_sea_masks, var, path_NN_reconstruction_test_data)
        plot_difference_NN_phases(denorm_testing_output_1, denorm_testing_output_2, land_sea_masks, var, path_NN_diff_test_data, list_float_profiles_coordinates[index_external_testing_2[i]])  #float_locations_coord[index_testing_2[i]])

        del test_data_2
        torch.cuda.empty_cache()


#Remove the last tensors on gpu
del my_mean_tensor_2
del my_std_tensor_2
torch.cuda.empty_cache()




print("model state dict", model_completion_2.state_dict().keys())
print("model conv1 weights", model_completion_2.state_dict()['conv1.weight'].size())
print("model conv1 weights", torch.sum(model_completion_2.state_dict()['conv1.weight'].isnan() == True))
print("model conv1 bias", model_completion_2.state_dict()['conv1.bias'].size())
print("model conv1 bias", torch.sum(model_completion_2.state_dict()['conv1.bias'].isnan() == True))
print("losses_2c_test", losses_1_c_test_2)
print("len loss 2c test", len(losses_1_c_test_2))
print("end")