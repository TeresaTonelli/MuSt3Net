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


#PHASE 1  --> MODEL data
#generate the new train dataset --> merging physical and biogeoch variable
physics_train_dataset, list_physics_weeks = get_list_model_tensor_weeks("physics_vars")
biogeoch_train_dataset, list_biogeoch_weeks = get_list_model_tensor_weeks(list_biogeoch_vars[0])

physics_train_dataset, list_physics_weeks = re_order_weeks(list_physics_weeks, physics_train_dataset)
biogeoch_train_dataset, list_biogeoch_weeks = re_order_weeks(list_biogeoch_weeks, biogeoch_train_dataset)

#Generation of land - sea masks
land_sea_masks = []
for i_d in range(int((depth_interval[1] - depth_interval[0]) / resolution[2] + 1) - 1):
    land_sea_mask = generate_sea_land_mask(biogeoch_train_dataset[0], i_d)
    torch.save(land_sea_mask, path_land_sea_masks + "/mask_depth_" + str(i_d) + ".pt")
    land_sea_masks.append(land_sea_mask)

#land_sea_masks =[land_sea_mask.to(device) for land_sea_mask in land_sea_masks]

old_biogeoch_train_dataset = biogeoch_train_dataset.copy()   
old_total_dataset = [concatenate_tensors(physics_train_dataset[i], old_biogeoch_train_dataset[i], axis=1) for i in range(len(physics_train_dataset))]



#sottocampiono i tensori biogeochimici per arrivare ad una cardinalità simile a quella dei dati float (ovviamente calcolati a livello settimanale)
duplicates_biogeoch_train_dataset, duplicates_biogeoch_coordinates = generate_list_sample_tensors(biogeoch_train_dataset, land_sea_masks[0], n_samples_biogeoch, n_duplicates_biogeoch)   #ovviamente questi parametri sono da risettare pois

duplicates_physics_train_dataset = extend_list(physics_train_dataset, n_duplicates_biogeoch)

old_duplicates_total_dataset = [concatenate_tensors(duplicates_physics_train_dataset[i], duplicates_biogeoch_train_dataset[i], axis=1) for i in range(len(duplicates_physics_train_dataset))]
old_duplicates_total_dataset = [old_duplicate_tensor[:, :, :-1, :, 1:-1] for old_duplicate_tensor in old_duplicates_total_dataset]
transposed_duplicates_biogeoch_coordinates = transpose_latitudes(duplicates_biogeoch_coordinates)
list_duplicates_biogeoch_weeks = create_list_duplicates(list_biogeoch_weeks, n_duplicates_biogeoch)



#fill missing values of biogeoch tensor with standard values
biogeoch_train_dataset_fill = [fill_tensor_opt(biogeoch_tensor, land_sea_masks, standard_mean_values[list_biogeoch_vars[0]]) for biogeoch_tensor in duplicates_biogeoch_train_dataset]
print("end opt fill")


#merging part
total_dataset = [concatenate_tensors(duplicates_physics_train_dataset[i], biogeoch_train_dataset_fill[i], axis=1) for i in range(len(duplicates_physics_train_dataset))]


#generate test daatset
index_testing = random.sample(range(len(total_dataset)), int(len(total_dataset) * 0.2))  #Qua bisogna cambiare il modo in cui fare il test
index_training = [i for i in range(len(total_dataset)) if i not in index_testing]

denormalized_test_dataset = [total_dataset[i] for i in index_testing]
denormalized_train_dataset = [total_dataset[i] for i in index_training]

total_dataset_norm, _, _ = Normalization(total_dataset, "1p", path_results)    
test_dataset = [total_dataset_norm[i] for i in index_testing]                         
train_dataset = [total_dataset_norm[i] for i in index_training]



# HYPERPARAMETERS
pretrain = 1 
pretrain = 0
epoch_pretrain = 0


model_completion = CompletionN()    


alpha = torch.tensor(4e-4)         
lr_c = 0.001           
epoch_c = 4
snaperiod = 3

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
        
        my_mean_tensor = torch.unsqueeze(torch.load(path_mean_std + "/mean_tensor.pt")[:, 6, :, :, :], 1).to(device)
        my_std_tensor = torch.unsqueeze(torch.load(path_mean_std + "/std_tensor.pt")[:, 6, :, :, :], 1).to(device)
        denormalized_NN_output = Denormalization(output, torch.unsqueeze(torch.load(path_mean_std + "/mean_tensor.pt")[:, 6, :, :, :], 1).to(device), torch.unsqueeze(torch.load(path_mean_std + "/std_tensor.pt")[:, 6, :, :, :], 1).to(device)).to(device)

        loss_completion = convolutional_network_exp_weighted_loss(old_total_dataset[int(index_training[i] / n_duplicates_biogeoch)][:, :, :-1, :, 1:-1].float(), denormalized_NN_output.float(), land_sea_masks, exp_weights)
        print("loss completion", loss_completion)
        losses_1_c.append(loss_completion.item())

        print(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_completion.item():.12f}")
        f.write(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_completion.item():.12f} \n")

        optimizer_completion.zero_grad()
        loss_completion.backward()
        optimizer_completion.step()
    

    #PHASE OF TESTING
    if ep % snaperiod == 0:
        model_completion.eval()

        losses_1_c_test = []
        with torch.no_grad():

            for i_test in range(len(test_dataset)): 
                test_data = test_dataset[i_test] 
                test_data = test_data.to(device)   
                testing_input = test_data
                testing_output = model_completion(testing_input.float())

                denormalized_testing_output = Denormalization(testing_output, torch.unsqueeze(torch.load(path_mean_std + "/mean_tensor.pt")[:,6, :,:,:], 1).to(device), 
                                                              torch.unsqueeze(torch.load(path_mean_std + "/std_tensor.pt")[:,6,:,:,:],1).to(device)).to(device)
                loss_1c_test = convolutional_network_exp_weighted_loss(old_total_dataset[int(index_testing[i_test] / n_duplicates_biogeoch)][:, :, :-1, :, 1:-1].float(), denormalized_testing_output.float(), land_sea_masks, exp_weights)
                losses_1_c_test.append(loss_1c_test.cpu())

                print(f"[EPOCH]: {ep + 1}, [TEST LOSS]: {loss_1c_test.item():.12f}")
                f_test.write(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_1c_test.item():.12f} \n")

            print("len test_1_c_", len(losses_1_c_test))
            test_loss = np.mean(np.array(losses_1_c_test))      
            test_losses.append(test_loss)


    if ep % snaperiod == 0: 
        torch.save(model_completion.state_dict(), path_model + "/ep_" + str(ep + epoch_pretrain) + ".pt")

f.close()
f_test.close()

Plot_Error(test_losses, "1p", path_losses + "/")

torch.save(model_completion.state_dict(), path_lr + "/final_model" + ".pt")


#Save maps of 1 phase: difference between output of the NN model (prediction of biogeoch var) and var biogeoch computed by numerical model
path_difference_maps_1p = path_plots + "/maps_1p"
var = biogeoch_var_to_predict
model_completion.eval()
with torch.no_grad():
    for i in range(len(test_dataset[0:5])):
        test_data = test_dataset[i]
        test_data = test_data.to(device)
        testing_input = test_data 
        testing_output = model_completion(testing_input.float())

        denorm_testing_output = Denormalization(testing_output, torch.unsqueeze(torch.load(path_mean_std + "/mean_tensor.pt")[:,6, :,:,:], 1).to(device), 
                                                torch.unsqueeze(torch.load(path_mean_std + "/std_tensor.pt")[:,6,:,:,:], 1).to(device))

        path_difference_map_1p = path_difference_maps_1p + "/test_data_week_" + str(list_duplicates_biogeoch_weeks[index_testing[i]])
        if not os.path.exists(path_difference_map_1p):
            os.makedirs(path_difference_map_1p)
        plot_difference_maps(torch.unsqueeze(old_total_dataset[int(index_testing[i] / n_duplicates_biogeoch)][:, :, :-1, :, 1:-1][:, 6, :, :, :], 1).to(device), denorm_testing_output, land_sea_masks, var, path_difference_map_1p)

#Create profiles of phase 1
path_profiles = path_plots + "/profiles_1p"
model_completion.eval()
with torch.no_grad():
    for i in range(len(test_dataset[0:5])):
        test_data = test_dataset[i]
        test_data = test_data.to(device)
        testing_input = test_data
        testing_output = model_completion(testing_input.float())

        denorm_testing_input = Denormalization(testing_input, torch.load(path_mean_std + "/mean_tensor.pt").to(device), torch.load(path_mean_std + "/std_tensor.pt").to(device))
        denorm_testing_output = Denormalization(testing_output, torch.unsqueeze(torch.load(path_mean_std + "/mean_tensor.pt")[:,6,:,:,:],1).to(device), 
                                                torch.unsqueeze(torch.load(path_mean_std + "/std_tensor.pt")[:,6,:,:,:],1).to(device))

        path_profiles_test_data = path_profiles + "/test_data_week_" + str(list_duplicates_biogeoch_weeks[index_testing[i]])
        if not os.path.exists(path_profiles_test_data):
            os.makedirs(path_profiles_test_data)
        
        plot_models_profiles_1p(torch.unsqueeze(denorm_testing_input[:, 6, :, :, :], 1), denorm_testing_output, torch.unsqueeze(old_total_dataset[int(index_testing[i] / n_duplicates_biogeoch)][:, :, :-1, :, 1:-1][:, 6, :, :, :], 1),  
                             var, path_profiles_test_data, transposed_duplicates_biogeoch_coordinates[index_testing[i]])    




#Plot results of NN_reconstruction for test data
path_NN_reconstruction = path_plots + "/NN_maps"
model_completion.eval()
with torch.no_grad():
    for i in range(len(test_dataset[0:5])):
        test_data = test_dataset[i]
        test_data = test_data.to(device)
        testing_input = test_data 
        testing_output = model_completion(testing_input.float())

        denorm_testing_output = Denormalization(testing_output, torch.unsqueeze(torch.load(path_mean_std + "/mean_tensor.pt")[:,6,:,:,:],1).to(device), 
                                                  torch.unsqueeze(torch.load(path_mean_std + "/std_tensor.pt")[:,6,:,:,:],1).to(device))
        
        path_NN_reconstruction_test_data = path_NN_reconstruction + "/test_data_week_" + str(list_duplicates_biogeoch_weeks[index_testing[i]]) 
        if not os.path.exists(path_NN_reconstruction_test_data):
            os.makedirs(path_NN_reconstruction_test_data)

        plot_NN_maps(denorm_testing_output, land_sea_masks, var, path_NN_reconstruction_test_data)


#Plot results of BFM_reconstruction for test data
path_BFM_reconstruction = path_plots + "/BFM_maps"
model_completion.eval()
with torch.no_grad():
    for i in range(len(test_dataset[0:5])):
        test_data = test_dataset[i]
        test_data = test_data.to(device)
        testing_input = test_data 
        
        path_BFM_reconstruction_test_data = path_BFM_reconstruction + "/test_data_week_" + str(list_duplicates_biogeoch_weeks[index_testing[i]])
        if not os.path.exists(path_BFM_reconstruction_test_data):
            os.makedirs(path_BFM_reconstruction_test_data)

        plot_NN_maps(torch.unsqueeze(old_total_dataset[int(index_testing[i] / n_duplicates_biogeoch)][:, :, :-1, :, 1:-1][:, 6, :, :, :], 1), land_sea_masks, var, path_BFM_reconstruction_test_data)






#PHASE 2

#SCRIVIAMO I NUOVI PATH IN CUI SALVARE TUTTI I RISULTATI CHE OTTENIAMO
# HYPERPARAMETERS
pretrain_2 = 1  # 0 means that we don"t use pretrained model to fine tuning
pretrain_2 = 0
epoch_pretrain_2 = 0

alpha_2 = torch.tensor(4e-4)          
lr_c_2 = 0.001           
epoch_c_2 = 2 
snaperiod_2 = 2
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


physics_train_dataset_2, list_physics_weeks_2 = get_list_model_tensor_weeks("physics_vars")
biogeoch_float_train_dataset, float_list_weeks = get_list_float_tensor_weeks(biogeoch_var_to_predict) 
print("sum biogeoch tensors", [torch.sum(bf) for bf in biogeoch_float_train_dataset])
print("list float", float_list_weeks)

physics_train_dataset_2, list_physics_weeks_2 = re_order_weeks(list_physics_weeks_2, physics_train_dataset_2)
biogeoch_float_train_dataset, float_list_weeks = re_order_weeks(float_list_weeks, biogeoch_float_train_dataset)
print("sum biogeoch tensors", [torch.sum(bf) for bf in biogeoch_float_train_dataset])
print("list_float_weeks", float_list_weeks)

physics_train_dataset_2.pop(-1)  
list_physics_weeks_2.pop(-1)  
biogeoch_float_train_dataset.pop(-1)
float_list_weeks.pop(-1)

print("sum biogeoch tensors", [torch.sum(bf) for bf in biogeoch_float_train_dataset])
print("list_float_weeks", float_list_weeks)

denorm_old_float_train_dataset = [concatenate_tensors(physics_train_dataset_2[i], biogeoch_float_train_dataset[i][:, 0:1, :, :, :], axis=1) for i in range(len(physics_train_dataset_2))]
denorm_old_float_train_dataset = [denorm_old_float_data[:, :, :-1, :, 1:-1] for denorm_old_float_data in denorm_old_float_train_dataset]
print("Nan in denorm float", [torch.nonzero(torch.isnan(d_o_tens)) for d_o_tens in denorm_old_float_train_dataset])


#find the coordinates of profiles of float, for each week 
list_float_profiles_coordinates = [compute_profile_coordinates(float_tensor[:, 0:1, :, :, :]) for float_tensor in biogeoch_float_train_dataset]
for f in list_float_profiles_coordinates:
    print("f", f)
sampled_list_float_profile_coordinates = [random.sample(float_profile_coord, int(0.4 * len(float_profile_coord))) for float_profile_coord in list_float_profiles_coordinates]

reduced_biogeoch_float_train_dataset = [remove_float(biogeoch_float_train_dataset[j], sampled_list_float_profile_coordinates[j]) for j in range(len(biogeoch_float_train_dataset))]

fill_biogeoch_float_train_dataset = [fill_tensor_opt(reduced_biogeoch_float_train_dataset[i_float][:, 0:1, :, :, :], land_sea_masks, standard_mean_values[list_biogeoch_vars[0]]) for i_float in range(len(reduced_biogeoch_float_train_dataset))]

#merging part
total_dataset_2 = [concatenate_tensors(physics_train_dataset_2[i], fill_biogeoch_float_train_dataset[i][:, 0:1, :, :, :], axis=1) for i in range(len(physics_train_dataset_2))]


#generate test daatset
index_testing_2 = random.sample(range(len(total_dataset_2)), int(len(total_dataset_2) * 0.2))  
index_training_2 = [i for i in range(len(total_dataset_2)) if i not in index_testing_2]

total_dataset_2_norm, _, _ = Normalization(total_dataset_2, "2p", path_results_2)    
test_dataset_2 = [total_dataset_2_norm[i] for i in index_testing_2]                         
train_dataset_2 = [total_dataset_2_norm[i] for i in index_training_2]



model_completion_1_load = CompletionN()    
model_completion_1_load.load_state_dict(torch.load(path_lr + "/final_model" + ".pt", map_location=torch.device('cpu')))
model_completion_1_load.eval()  


model_completion_2 = CompletionN()    
model_completion_2.load_state_dict(torch.load(path_lr + "/final_model" + ".pt"))
model_completion_2.eval()     


losses_1_c_2 = []
losses_1_c_test_2 = []
test_losses_2 = []


optimizer_completion_2 =torch.optim.Adam(model_completion_2.parameters(), lr=lr_c_2)   #questo è l'optimizer che uso per questa rete --> che implementa praticamente una stochastic gradient descend

f_2, f_test_2 = open(path_losses_2 + "/train_loss.txt", "w+"), open(path_losses_2 + "/test_loss.txt", "w+")

for ep in range(epoch_c_2):
    if ep > 400:     
        lr_c_2 = 0.0001

    #PHASE OF TRAINING
    for i in range(len(train_dataset_2)):
        training_x = train_dataset_2[i]
        print("n zeros", torch.count_nonzero(torch.isnan(training_x)))
        training_x = training_x.to(device)
        input = training_x  
        model_completion_2 = model_completion_2.to(device)
        output = model_completion_2(input.float())
        print("output", torch.nonzero(torch.isnan(output)))
        print("shapes torch isnan output", torch.nonzero(torch.isnan(output)).shape)


        #ora calcolo la loss solo sui float --> solo sui profili campionati 
        denormalized_output = Denormalization(output, torch.unsqueeze(torch.load(path_mean_std_2 + "/mean_tensor.pt")[:,6, :,:,:], 1).to(device), 
                                              torch.unsqueeze(torch.load(path_mean_std_2 + "/std_tensor.pt")[:,6,:,:,:], 1).to(device)).to(device)
        print("denorm output", torch.nonzero(torch.isnan(denormalized_output)))
        loss_completion_2 = convolutional_network_float_exp_weighted_loss(denorm_old_float_train_dataset[index_training_2[i]].float(), denormalized_output.float(), land_sea_masks, generate_float_mask(sampled_list_float_profile_coordinates[index_training_2[i]]).to(device), exp_weights)
        print("loss completion", loss_completion_2)
        losses_1_c.append(loss_completion_2.item())

        print(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_completion_2.item():.12f}")
        f_2.write(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_completion_2.item():.12f} \n")

        optimizer_completion_2.zero_grad()
        loss_completion_2.backward()
        optimizer_completion_2.step()


    #PHASE OF TESTING
    if ep % snaperiod_2 == 0:
        model_completion_2.eval()

        losses_1_c_test_2 = []
        with torch.no_grad():

            for i_test in range(len(test_dataset_2)):
                test_data = test_dataset_2[i_test]  
                test_data = test_data.to(device)   
                testing_input = test_data  
                testing_output = model_completion_2(testing_input.float())

                denormalized_output = Denormalization(testing_output, torch.unsqueeze(torch.load(path_mean_std_2 + "/mean_tensor.pt")[:,6, :,:,:],1).to(device), 
                                                      torch.unsqueeze(torch.load(path_mean_std_2 + "/std_tensor.pt")[:,6,:,:,:],1).to(device)).to(device)
                loss_1c_test_2 = convolutional_network_float_exp_weighted_loss(denorm_old_float_train_dataset[index_testing_2[i_test]].float(), denormalized_output.float(), land_sea_masks, generate_float_mask(sampled_list_float_profile_coordinates[index_testing_2[i_test]]).to(device), exp_weights)
                losses_1_c_test_2.append(loss_1c_test_2.cpu())

                print(f"[EPOCH]: {ep + 1}, [TEST LOSS]: {loss_1c_test_2.item():.12f}")
                f_test_2.write(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_1c_test_2.item():.12f} \n")

            test_loss_2 = np.mean(np.array(losses_1_c_test_2))      
            test_losses_2.append(test_loss_2)


    if ep % snaperiod_2 == 0:  
        torch.save(model_completion_2.state_dict(), path_model_2 + "/ep_" + str(ep + epoch_pretrain_2) + ".pt")

f_2.close()
f_test_2.close()

Plot_Error(test_losses_2, "2p", path_losses_2 + "/")

torch.save(model_completion_2.state_dict(), path_lr_2 + "/final_model" + ".pt")


#Save maps of 2 phase: difference between output of the NN model (prediction of biogeoch var) and var biogeoch computed by numerical model
path_difference_maps_2p = path_plots_2 + "/maps_2p"
var = biogeoch_var_to_predict
model_completion_2.eval()
with torch.no_grad():
    for i in range(len(test_dataset_2)):
        test_data_2 = test_dataset_2[i]
        test_data_2 = test_data_2.to(device)
        testing_input_2 = test_data_2   
        testing_output_2 = model_completion_2(testing_input_2.float())

        denorm_testing_output_2 = Denormalization(testing_output_2, torch.unsqueeze(torch.load(path_mean_std_2 + "/mean_tensor.pt")[:,6,:,:,:],1).to(device), 
                                                  torch.unsqueeze(torch.load(path_mean_std_2 + "/std_tensor.pt")[:,6,:,:,:],1).to(device))

        path_difference_map_2p = path_difference_maps_2p + "/test_data_" + str(i) + "_week_" + str(float_list_weeks[index_testing_2[i]])
        if not os.path.exists(path_difference_map_2p):
            os.makedirs(path_difference_map_2p)

        plot_difference_maps(torch.unsqueeze(old_total_dataset[int(index_testing_2[i] / n_duplicates_biogeoch)][:, :, :-1, :, 1:-1][:, 6, :, :, :], 1).to(device), denorm_testing_output_2, land_sea_masks, var, path_difference_map_2p)
    

##Save profiles of 2 phase: comparison between the profile of float vs the extraction of the SAME profile of NN model vs the extraction of the SAME profile of num model 
#path_profiles = path_plots_2 + "/profiles_2p"
#model_completion_2.eval()
#with torch.no_grad():
 #   for i in range(len(test_dataset_2)):
  #      print("new test data")
   #     test_data_2 = test_dataset_2[i]
    #    test_data_2 = test_data_2.to(device)
     #   testing_input_2 = test_data_2   
      #  testing_output_2 = model_completion_2(testing_input_2.float())

       # denorm_testing_output_2 = Denormalization(testing_output_2, torch.unsqueeze(torch.load(path_mean_std_2 + "/mean_tensor.pt")[:,6,:,:,:],1).to(device), 
        #                                          torch.unsqueeze(torch.load(path_mean_std_2 + "/std_tensor.pt")[:,6,:,:,:],1).to(device))

        #path_profiles_test_data = path_profiles + "/test_data_" + str(i) + "_week_" + str(float_list_weeks[index_testing_2[i]])
        #if not os.path.exists(path_profiles_test_data):
         #   os.makedirs(path_profiles_test_data)
        
        #plot_models_profiles(torch.unsqueeze(denorm_old_float_train_dataset[index_testing_2[i]][:, 6, :, :, :], 1) , denorm_testing_output_2, biogeoch_train_dataset[index_testing_2[i]][:, :, :-1, :, 1:-1], 
         #                    var, path_profiles_test_data, sampled_list_float_profile_coordinates[index_testing_2[i]])  



#Save profiles of 2 phase wrt 1 NN: comparison between the profile of float vs the extraction of the SAME profile of NN model,in its 1 phase vs the extraction of the SAME profile of num model 
path_profiles_with_NN_1 = path_plots_2 + "/profiles_2p_NN_1"
model_completion_2.eval()
with torch.no_grad():
    for i in range(len(test_dataset_2)):
        print("new test data")
        test_data_2 = test_dataset_2[i]
        test_data_2 = test_data_2.to(device)
        testing_input_2 = test_data_2   
        testing_output_2 = model_completion_2(testing_input_2.float())

        denorm_testing_output_2 = Denormalization(testing_output_2, torch.unsqueeze(torch.load(path_mean_std_2 + "/mean_tensor.pt")[:,6,:,:,:],1).to(device), 
                                                  torch.unsqueeze(torch.load(path_mean_std_2 + "/std_tensor.pt")[:,6,:,:,:],1).to(device))
        norm_testing_output_1 = model_completion_1_load(testing_input_2.cpu().float()).to(device) 
        denorm_testing_output_1 = Denormalization(norm_testing_output_1, torch.unsqueeze(torch.load(path_mean_std_2 + "/mean_tensor.pt")[:,6,:,:,:],1).to(device), 
                                                  torch.unsqueeze(torch.load(path_mean_std_2 + "/std_tensor.pt")[:,6,:,:,:],1).to(device)) 

        path_profiles_test_data_NN_1 = path_profiles_with_NN_1 + "/test_data_" + str(i) + "_week_" + str(float_list_weeks[index_testing_2[i]])
        if not os.path.exists(path_profiles_test_data_NN_1):
            os.makedirs(path_profiles_test_data_NN_1)
        
        plot_models_profiles(torch.unsqueeze(denorm_old_float_train_dataset[index_testing_2[i]][:, 6, :, :, :], 1) , denorm_testing_output_1, biogeoch_train_dataset[index_testing_2[i]][:, :, :-1, :, 1:-1], 
                             var, path_profiles_test_data_NN_1, sampled_list_float_profile_coordinates[index_testing_2[i]])



#Plot results of NN_reconstruction for test data
path_NN_reconstruction = path_plots_2 + "/NN_maps"
model_completion_2.eval()
with torch.no_grad():
    for i in range(len(test_dataset_2)):
        test_data_2 = test_dataset_2[i]
        test_data_2 = test_data_2.to(device)
        testing_input_2 = test_data_2   
        testing_output_2 = model_completion_2(testing_input_2.float())

        denorm_testing_output_2 = Denormalization(testing_output_2, torch.unsqueeze(torch.load(path_mean_std_2 + "/mean_tensor.pt")[:,6,:,:,:],1).to(device), 
                                                  torch.unsqueeze(torch.load(path_mean_std_2 + "/std_tensor.pt")[:,6,:,:,:],1).to(device))
        
        path_NN_reconstruction_test_data = path_NN_reconstruction + "/test_data_" + str(i) + "_week_" + str(float_list_weeks[index_testing_2[i]])
        if not os.path.exists(path_NN_reconstruction_test_data):
            os.makedirs(path_NN_reconstruction_test_data)

        plot_NN_maps(denorm_testing_output_2, land_sea_masks, var, path_NN_reconstruction_test_data)


#Plot differences between 1 phase output and 2 phase output of NN
path_NN_phases_diff = path_plots_2 + "/NN_phases_diff"
model_completion_2.eval()
with torch.no_grad():
    for i in range(len(test_dataset_2)):
        test_data_2 = test_dataset_2[i]
        test_data_2 = test_data_2.to(device)
        testing_input_2 = test_data_2  
        testing_output_2 = model_completion_2(testing_input_2.float())

        denorm_testing_output_2 = Denormalization(testing_output_2, torch.unsqueeze(torch.load(path_mean_std_2 + "/mean_tensor.pt")[:,6,:,:,:],1).to(device), 
                                                  torch.unsqueeze(torch.load(path_mean_std_2 + "/std_tensor.pt")[:,6,:,:,:],1).to(device))
        norm_testing_output_1 = model_completion_1_load(testing_input_2.cpu().float()).to(device) 
        denorm_testing_output_1 = Denormalization(norm_testing_output_1, torch.unsqueeze(torch.load(path_mean_std_2 + "/mean_tensor.pt")[:,6,:,:,:],1).to(device), 
                                                  torch.unsqueeze(torch.load(path_mean_std_2 + "/std_tensor.pt")[:,6,:,:,:],1).to(device)) 

        path_NN_diff_test_data = path_NN_phases_diff + "/test_data_" + str(i) + "_week_" + str(float_list_weeks[index_testing_2[i]])
        if not os.path.exists(path_NN_diff_test_data):
            os.makedirs(path_NN_diff_test_data)

        plot_difference_NN_phases(denorm_testing_output_1, denorm_testing_output_2, land_sea_masks, var, path_NN_diff_test_data, list_float_profiles_coordinates[index_testing_2[i]])  #float_locations_coord[index_testing_2[i]])



#Comparison between 1 phase and 2 phase profiles in the same plot 


#Plot differences between 1 phase output and 2 phase output of NN, wrt layers




print("model state dict", model_completion_2.state_dict().keys())
print("model conv1 weights", model_completion_2.state_dict()['conv1.weight'].size())
print("model conv1 weights", torch.sum(model_completion_2.state_dict()['conv1.weight'].isnan() == True))
print("model conv1 bias", model_completion_2.state_dict()['conv1.bias'].size())
print("model conv1 bias", torch.sum(model_completion_2.state_dict()['conv1.bias'].isnan() == True))
print("losses_2c_test", losses_1_c_test_2)
print("len loss 2c test", len(losses_1_c_test_2))
print("end")