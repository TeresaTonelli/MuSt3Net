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
from losses import completion_network_loss, convolutional_network_loss, convolutional_network_float_loss
from mean_pixel_value import MV_pixel
from utils_mask import generate_input_mask, generate_sea_land_mask
from normalization import Normalization
from denormalization import Denormalization
from get_dataset import *
from plot_error import Plot_Error
from plot_results import *
from utils_function import *
from utils_mask import generate_float_mask

num_channel = number_channel  
name_datetime_folder = str(datetime.datetime.utcnow())

path_job = "results_job_" + name_datetime_folder
if not os.path.exists(path_job):
    os.mkdir(path_job)

path_results = path_job + "/results_training_1"     #"results_training" + "_" + name_datetime_folder      
if not os.path.exists(path_results):                   
    os.mkdir(path_results)

path_mean_std = path_results + "/mean_and_std_tensors"
if not os.path.exists(path_mean_std):
    os.makedirs(path_mean_std)

#creo il device per le GPU e porto i dati su GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device", device)


biogeoch_var_to_predict = "O2o"


#PHASE 1  --> MODEL data
#generate the new train dataset --> merging physical and biogeoch variable
physics_train_dataset = get_list_model_tensor("physics_vars")[:10]  #Perchè loro hanno lunghezza 52, ma dopo i primi 10 ho tensori nulli
biogeoch_train_dataset = get_list_model_tensor("O2o")[:10]

physics_train_dataset = [physics_train_data.to(device) for physics_train_data in physics_train_dataset]  #physics_train_dataset.to(device)
biogeoch_train_dataset = [biogeoch_train_data.to(device) for biogeoch_train_data in biogeoch_train_dataset]


#Generation of land - sea masks
land_sea_masks = []
for i_d in range(int((depth_interval[1] - depth_interval[0]) / resolution[2] + 1) - 1):
    land_sea_mask = generate_sea_land_mask(biogeoch_train_dataset[0], i_d)
    land_sea_masks.append(land_sea_mask)

land_sea_masks =[land_sea_mask.to(device) for land_sea_mask in land_sea_masks]

old_biogeoch_train_dataset = biogeoch_train_dataset.copy()   #sta roba mi serve solo per ritrovare gli indici dei tensori per ottenere i profili del modello numerico della var biogeoch
old_total_dataset = [concatenate_tensors(physics_train_dataset[i], old_biogeoch_train_dataset[i], axis=1) for i in range(len(physics_train_dataset))]
#old_total_dataset, _, _ = Normalization(old_total_dataset, "1p", path_results)   #tolot perchè ora sto lavorando con la denormalization

#old_biogeoch_train_dataset, _, _ = Normalization(old_biogeoch_train_dataset, "1p", path_results, number_channel=1)   #questa phase è da cambiare, RICORDA



#sottocampiono i tensori biogeochimici per arrivare ad una cardinalità simile a quella dei dati float (ovviamente calcolati a livello settimanale)
duplicates_biogeoch_train_dataset = generate_list_sample_tensors(biogeoch_train_dataset, land_sea_masks[0], 100, 10)   #ovviamente questi parametri sono da risettare pois
duplicates_physics_train_dataset = extend_list(physics_train_dataset, 10)
old_duplicates_total_dataset = [concatenate_tensors(duplicates_physics_train_dataset[i], duplicates_biogeoch_train_dataset[i], axis=1) for i in range(len(duplicates_physics_train_dataset))]
old_duplicates_total_dataset = [old_duplicate_tensor[:, :, :-1, :, 1:-1] for old_duplicate_tensor in old_duplicates_total_dataset]

#create the tensor that we will use for the loss --> No mean value and no normalization

#fill missing values of biogeoch tensor with standard values
biogeoch_train_dataset_fill = [fill_tensor_with_standard(biogeoch_tensor, land_sea_masks, 200) for biogeoch_tensor in duplicates_biogeoch_train_dataset]

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


#implement the mean pixel value technique
mean_value_pixel = MV_pixel(total_dataset)  # compute the mean of the channel of the training set
mean_value_pixel = torch.tensor(mean_value_pixel.reshape(1, num_channel, 1, 1, 1))     #Questa era un'operazione fatta nel codice che Gloria ha copiato --> dovrebbe servire per delle cose, quindi la lascio
mean_value_pixel = mean_value_pixel.to(device)


# HYPERPARAMETERS
pretrain = 1  # 0 means that we don"t use pretrained model to fine tuning
#ora non ho il pretrain quindi metto pretrain = 0
pretrain = 0
#e aggiungo anceh epoch_pretrain 
epoch_pretrain = 0


model_completion = CompletionN()     #modo per istanziare un oggetto della classe Completion --> quindi ora ho un oggetto che è una rete
#if pretrain:                            #da usare solo se uso il pretrain, ovvero se devo ripartire da risultati parziali, magari perchè mi si è bloccata la compilazione
 #   path_pretrain = os.getcwd() + "/starting_model/"
  #  epoch_pretrain = 5000
   # model_name = "model_step1_ep_" + str(epoch_pretrain) + ".pt"
    #model_completion.load_state_dict(torch.load(path_pretrain + model_name))
    #model_completion.eval()

alpha = torch.tensor(4e-4)         
lr_c = 0.001           
epoch_c = 20 
snaperiod = 5



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

for ep in range(epoch_c):
    if ep > 400:     #proposto nel paper da cui sto prendendo l'architettura
        lr_c = 0.0001

    #PHASE OF TRAINING
    for i in range(len(train_dataset)):
        training_x = train_dataset[i]
        training_x = training_x.to(device)
        input = training_x
        model_completion = model_completion.to(device)
        output = model_completion(input.float())

        #find denormalized tensors to compute the loss
        #denormalized_training_x = denormalized_train_dataset[i]
        my_mean_tensor = torch.unsqueeze(torch.load(path_mean_std + "/mean_tensor.pt")[:, 6, :, :, :], 1)
        my_std_tensor = torch.unsqueeze(torch.load(path_mean_std + "/std_tensor.pt")[:, 6, :, :, :], 1)
        denormalized_NN_output = Denormalization(output, torch.unsqueeze(torch.load(path_mean_std + "/mean_tensor.pt")[:, 6, :, :, :], 1), torch.unsqueeze(torch.load(path_mean_std + "/std_tensor.pt")[:, 6, :, :, :], 1))

        loss_completion = convolutional_network_loss(old_total_dataset[int(index_training[i] / 10)][:, :, :-1, :, 1:-1].float(), denormalized_NN_output.float(), land_sea_masks)
        #loss_completion = convolutional_network_loss(old_duplicates_total_dataset[index_training[i]].float(), denormalized_NN_output.float(), land_sea_masks)
        print("loss completion", loss_completion)
        losses_1_c.append(loss_completion.item())

        print(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_completion.item():.12f}")
        f.write(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_completion.item():.12f} \n")

        optimizer_completion.zero_grad()
        loss_completion.backward()
        optimizer_completion.step()
    

    # test       #faccio il test ogni a intervalli di snaperiod 
    #PHASE OF TESTING
    if ep % snaperiod == 0:
        model_completion.eval()

        losses_1_c_test = []
        with torch.no_grad():

            for i_test in range(len(test_dataset)): 
                test_data = test_dataset[i_test] 
                test_data.to(device)   
                testing_input = test_data
                testing_output = model_completion(testing_input.float())

                denormalized_testing_output = Denormalization(testing_output, torch.unsqueeze(torch.load(path_mean_std + "/mean_tensor.pt")[:,6, :,:,:], 1), 
                                                              torch.unsqueeze(torch.load(path_mean_std + "/std_tensor.pt")[:,6,:,:,:],1))
                loss_1c_test = convolutional_network_loss(old_total_dataset[int(index_testing[i_test] / 10)][:, :, :-1, :, 1:-1].float(), denormalized_testing_output.float(), land_sea_masks)
                #loss_1c_test = convolutional_network_loss(old_duplicates_total_dataset[index_testing[i_test]].float(), denormalized_testing_output.float(), land_sea_masks)  
                losses_1_c_test.append(loss_1c_test)

                print(f"[EPOCH]: {ep + 1}, [TEST LOSS]: {loss_1c_test.item():.12f}")
                f_test.write(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_1c_test.item():.12f} \n")

            print("len test_1_c_", len(losses_1_c_test))
            test_loss = np.mean(np.array(losses_1_c_test))      
            test_losses.append(test_loss)


    if ep % snaperiod == 0:  # save the partial model
        torch.save(model_completion.state_dict(), path_model + "/ep_" + str(ep + epoch_pretrain) + ".pt")

f.close()
f_test.close()

Plot_Error(test_losses, "1p", path_losses + "/")

torch.save(model_completion.state_dict(), path_lr + "/final_model" + ".pt")


#Save maps of 1 phase: difference between output of the NN model (prediction of biogeoch var) and var biogeoch computed by numerical model
path_difference_maps_1p = path_plots + "/maps_1p"
var = "O2o"
model_completion.eval()
with torch.no_grad():
    for i in range(len(test_dataset)):
        test_data = test_dataset[i]
        testing_input = test_data  
        testing_output = model_completion(testing_input.float())

        denorm_testing_output = Denormalization(testing_output, torch.unsqueeze(torch.load(path_mean_std + "/mean_tensor.pt")[:,6, :,:,:], 1), 
                                                torch.unsqueeze(torch.load(path_mean_std + "/std_tensor.pt")[:,6,:,:,:], 1))

        path_difference_map_1p = path_difference_maps_1p + "/test_data_" + str(i)
        if not os.path.exists(path_difference_map_1p):
            os.makedirs(path_difference_map_1p)
        #plot_difference_maps(torch.unsqueeze(testing_input[:, 6, :, :, :], 1), testing_output, land_sea_masks, var, path_difference_map_1p)
        plot_difference_maps(torch.unsqueeze(old_total_dataset[int(index_testing[i] / 10)][:, :, :-1, :, 1:-1][:, 6, :, :, :], 1), denorm_testing_output, land_sea_masks, var, path_difference_map_1p)

#Create profiles of phase 1
#path_profiles = path_plots + "/profiles_1p"
#model_completion.eval()
#with torch.no_grad():
 #   for i in range(len(test_dataset)):
  #      print("index testing i", index_testing[i])
   #     test_data = test_dataset[i]
    #    testing_input = test_data
     #   testing_output = model_completion(testing_input.float())
#
 #       denorm_testing_input = Denormalization(testing_input, torch.load(path_mean_std + "/mean_tensor.pt"), torch.load(path_mean_std + "/std_tensor.pt"))
  #      denorm_testing_output = Denormalization(testing_output, torch.unsqueeze(torch.load(path_mean_std + "/mean_tensor.pt")[:,6,:,:,:],1), 
   #                                             torch.unsqueeze(torch.load(path_mean_std + "/std_tensor.pt")[:,6,:,:,:],1))
#
 #       path_profiles_test_data = path_profiles + "/test_data_" + str(i)
  #      if not os.path.exists(path_profiles_test_data):
   #         os.makedirs(path_profiles_test_data)
    #    
     #   plot_models_profiles(torch.unsqueeze(old_total_dataset[int(index_testing[i] / 10)][:, :, :-1, :, 1:-1][:, 6, :, :, :], 1), denorm_testing_output, torch.unsqueeze(denorm_testing_input[:, 6, :, :, :], 1),  
      #                       var, path_profiles_test_data, [(lon_ind, lat_ind) for lon_ind in range(0, h) for lat_ind in range(0, w)])  













#PHASE 2

#SCRIVIAMO I NUOVI PATH IN CUI SALVARE TUTTI I RISULTATI CHE OTTENIAMO
# HYPERPARAMETERS
pretrain_2 = 1  # 0 means that we don"t use pretrained model to fine tuning
#ora non ho il pretrain quindi metto pretrain = 0
pretrain_2 = 0
#e aggiungo anceh epoch_pretrain 
epoch_pretrain_2 = 0
alpha_2 = torch.tensor(4e-4)          
lr_c_2 = 0.001           
epoch_c_2 = 10 
snaperiod_2 = 2
path_results_2 = path_job + "/results_training_2"  #"results_training_2" + "_" + str(datetime.datetime.utcnow())
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


physics_train_dataset_2 = get_list_model_tensor("physics_vars")[:10]
biogeoch_float_train_dataset = get_list_float_tensor("O2o")[:10]   #DA CONTROLLARE PERCHE LUI è UN FLOAT, E FORSE SI COMPORTA DIVERSAMENTE
denorm_old_float_train_dataset = [concatenate_tensors(physics_train_dataset_2[i], biogeoch_float_train_dataset[i][:, 0:1, :, :, :], axis=1) for i in range(len(physics_train_dataset_2))]
denorm_old_float_train_dataset = [denorm_old_float_data[:, :, :-1, :, 1:-1] for denorm_old_float_data in denorm_old_float_train_dataset]
#old_float_train_dataset, _, _ = Normalization(denorm_old_float_train_dataset, "2p", path_results_2)

#find the coordinates of profiles of float, for each week 
list_float_profiles_coordinates = [compute_profile_coordinates(float_tensor[:, 0:1, :, :, :]) for float_tensor in biogeoch_float_train_dataset]
print("list float profile coord", list_float_profiles_coordinates)
sampled_list_float_profile_coordinates = [random.sample(float_profile_coord, int(0.4 * len(float_profile_coord))) for float_profile_coord in list_float_profiles_coordinates]
print("sampled list float profiles coord", sampled_list_float_profile_coordinates)

#remove sampled_coordinates profiles from float tensor
#sampled_float_coordinates_tensors = [generate_sampled_profiles_tensor(biogeoch_float_train_dataset[j], sampled_list_float_profile_coordinates[j]) for j in range(len(biogeoch_float_train_dataset))]
reduced_biogeoch_float_train_dataset = [remove_float(biogeoch_float_train_dataset[j], sampled_list_float_profile_coordinates[j]) for j in range(len(biogeoch_float_train_dataset))]

#fill in the missing values of float data with standard value
#biogeoch_float_train_dataset = [fill_tensor_with_standard(biogeoch_float_train_dataset[i_float][:, 0:1, :, :, :] - sampled_float_coordinates_tensors[i_float][:, 0:1, :, :, :], land_sea_masks, 200) for i_float in range(len(biogeoch_float_train_dataset))]
fill_biogeoch_float_train_dataset = [fill_tensor_with_standard(reduced_biogeoch_float_train_dataset[i_float][:, 0:1, :, :, :], land_sea_masks, 200) for i_float in range(len(reduced_biogeoch_float_train_dataset))]

#merging part
#QUESTA COSA VA BENE ORA PERCHE HO I FLOAT CON LA dim 1 SBAGLIATA --> poi sta roba si deve togliere e rimettere normale
total_dataset_2 = [concatenate_tensors(physics_train_dataset_2[i], fill_biogeoch_float_train_dataset[i][:, 0:1, :, :, :], axis=1) for i in range(len(physics_train_dataset_2))]


#generate test daatset
index_testing_2 = random.sample(range(len(total_dataset_2)), int(len(total_dataset_2) * 0.2))  #Qua bisogna cambiare il modo in cui fare il test
index_training_2 = [i for i in range(len(total_dataset_2)) if i not in index_testing_2]

total_dataset_2_norm, _, _ = Normalization(total_dataset_2, "2p", path_results_2)    
test_dataset_2 = [total_dataset_2_norm[i] for i in index_testing_2]                         
train_dataset_2 = [total_dataset_2_norm[i] for i in index_training_2]




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
        training_x = training_x.to(device)
        input = training_x  
        model_completion_2 = model_completion_2.to(device)
        output = model_completion_2(input.float())


        #ora calcolo la loss solo sui float --> solo sui profili campionati 
        denormalized_output = Denormalization(output, torch.unsqueeze(torch.load(path_mean_std_2 + "/mean_tensor.pt")[:,6, :,:,:], 1), 
                                              torch.unsqueeze(torch.load(path_mean_std_2 + "/std_tensor.pt")[:,6,:,:,:], 1))
        loss_completion_2 = convolutional_network_float_loss(denorm_old_float_train_dataset[index_training_2[i]].float(), denormalized_output.float(), land_sea_masks, generate_float_mask(sampled_list_float_profile_coordinates[index_training_2[i]]))
        print("loss completion", loss_completion_2)
        losses_1_c.append(loss_completion_2.item())

        print(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_completion_2.item():.12f}")
        f_2.write(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_completion_2.item():.12f} \n")

        optimizer_completion_2.zero_grad()
        loss_completion_2.backward()
        optimizer_completion_2.step()

    # test       #faccio il test ogni a intervalli di snaperiod 
    #PHASE OF TESTING
    if ep % snaperiod_2 == 0:
        model_completion_2.eval()

        losses_1_c_test_2 = []
        with torch.no_grad():

            for i_test in range(len(test_dataset_2)):
                test_data = test_dataset_2[i_test]  
                test_data.to(device)   
                testing_input = test_data  
                testing_output = model_completion_2(testing_input.float())

                denormalized_output = Denormalization(testing_output, torch.unsqueeze(torch.load(path_mean_std_2 + "/mean_tensor.pt")[:,6, :,:,:],1), 
                                                      torch.unsqueeze(torch.load(path_mean_std_2 + "/std_tensor.pt")[:,6,:,:,:],1))
                loss_1c_test_2 = convolutional_network_float_loss(denorm_old_float_train_dataset[index_testing_2[i_test]].float(), denormalized_output.float(), land_sea_masks, generate_float_mask(sampled_list_float_profile_coordinates[index_testing_2[i_test]]))  #convolutional_network_loss(test_data, testing_output, land_sea_masks)
                losses_1_c_test_2.append(loss_1c_test_2)

                print(f"[EPOCH]: {ep + 1}, [TEST LOSS]: {loss_1c_test_2.item():.12f}")
                f_test_2.write(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_1c_test_2.item():.12f} \n")

            test_loss_2 = np.mean(np.array(losses_1_c_test_2))      
            test_losses_2.append(test_loss_2)


    if ep % snaperiod_2 == 0:  # save the partial model
        torch.save(model_completion_2.state_dict(), path_model_2 + "/ep_" + str(ep + epoch_pretrain_2) + ".pt")

f_2.close()
f_test_2.close()

Plot_Error(test_losses_2, "2p", path_losses_2 + "/")

torch.save(model_completion_2.state_dict(), path_lr_2 + "/final_model" + ".pt")


#Save maps of 2 phase: difference between output of the NN model (prediction of biogeoch var) and var biogeoch computed by numerical model
path_difference_maps_2p = path_plots + "/maps_2p"
var = "O2o"
model_completion_2.eval()
with torch.no_grad():
    for i in range(len(test_dataset_2)):
        test_data_2 = test_dataset_2[i]
        testing_input_2 = test_data_2 
        testing_output_2 = model_completion_2(testing_input_2.float())

        denorm_testing_output_2 = Denormalization(testing_output_2, torch.unsqueeze(torch.load(path_mean_std_2 + "/mean_tensor.pt")[:,6,:,:,:],1), 
                                                  torch.unsqueeze(torch.load(path_mean_std_2 + "/std_tensor.pt")[:,6,:,:,:],1))

        path_difference_map_2p = path_difference_maps_2p + "/test_data_" + str(i)
        if not os.path.exists(path_difference_map_2p):
            os.makedirs(path_difference_map_2p)
        #plot_difference_maps(torch.unsqueeze(testing_input_2[:, 6, :, :, :], 1), testing_output_2, land_sea_masks, var, path_difference_map_2p)
        #plot_difference_maps(old_biogeoch_train_dataset[index_testing_2[i]], testing_output_2, land_sea_masks, var, path_difference_map_2p)
        plot_difference_maps(torch.unsqueeze(old_total_dataset[int(index_testing_2[i] / 10)][:, :, :-1, :, 1:-1][:, 6, :, :, :], 1), denorm_testing_output_2, land_sea_masks, var, path_difference_map_2p)


#Save profiles of 2 phase: comparison between the profile of float vs the extraction of the SAME profile of NN model vs the extraction of the SAME profile of num model 
path_profiles = path_plots + "/profiles_2p"
model_completion_2.eval()
with torch.no_grad():
    for i in range(len(test_dataset_2)):
        print("new test data")
        test_data_2 = test_dataset_2[i]
        testing_input_2 = test_data_2 
        testing_output_2 = model_completion_2(testing_input_2.float())

        denorm_testing_output_2 = Denormalization(testing_output_2, torch.unsqueeze(torch.load(path_mean_std_2 + "/mean_tensor.pt")[:,6,:,:,:],1), 
                                                  torch.unsqueeze(torch.load(path_mean_std_2 + "/std_tensor.pt")[:,6,:,:,:],1))

        path_profiles_test_data = path_profiles + "/test_data_" + str(i)
        if not os.path.exists(path_profiles_test_data):
            os.makedirs(path_profiles_test_data)
        
        plot_models_profiles(torch.unsqueeze(denorm_old_float_train_dataset[index_testing_2[i]][:, 6, :, :, :], 1) , denorm_testing_output_2, biogeoch_train_dataset[index_testing_2[i]][:, :, :-1, :, 1:-1], 
                             var, path_profiles_test_data, sampled_list_float_profile_coordinates[index_testing_2[i]])   #IL TENSORE DEL MODELLO NUMERICO CORRISPONDENTE NON SAPREI COME RITROVARLO




print("model state dict", model_completion_2.state_dict().keys())
print("model conv1 weights", model_completion_2.state_dict()['conv1.weight'].size())
print("model conv1 weights", torch.sum(model_completion_2.state_dict()['conv1.weight'].isnan() == True))
print("model conv1 bias", model_completion_2.state_dict()['conv1.bias'].size())
print("model conv1 bias", torch.sum(model_completion_2.state_dict()['conv1.bias'].isnan() == True))
print("losses_2c_test", losses_1_c_test_2)
print("len loss 2c test", len(losses_1_c_test_2))
#Plot_Error(losses_1_c_test[0].cpu(), "1c", path_lr + "/")   #modificato per tornare alle cpu, prima era senza [0].cpu() --> non sono convinta che funzioni
print("end")