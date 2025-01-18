"""
Implementation of the training routine for the 3D CNN with GAN
- train_dataset : list/array of 5D (or 5D ?) tensor in form (bs, input_channels, D_in, H_in, W_in)
"""
import numpy as np
import torch.nn as nn
from torch.optim import Adadelta
import matplotlib.pyplot as plt
import random
#from IPython import display
import datetime

#from completion import CompletionN
from convolutional_network import CompletionN
from losses import completion_network_loss, convolutional_network_loss, convolutional_network_float_loss
from mean_pixel_value import MV_pixel
from utils_mask import generate_input_mask, generate_sea_land_mask
from normalization import Normalization
#from plot_save_tensor import Plot_Error
from get_dataset import *
from plot_error import Plot_Error
from plot_results import *
from utils_function import *
from utils_mask import generate_float_mask

num_channel = number_channel  
name_datetime_folder = str(datetime.datetime.utcnow())

path_results = "results_training" + "_" + name_datetime_folder      
if not os.path.exists(path_results):                   
    os.mkdir(path_results)

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
#old_biogeoch .. e biogeoch_train_dataset DOVREBBERO essere due cose diverse --> e se modifico una modifico l'altra?? --> sembra di no, ma non sono sicura
#e poi lo devo anche normalizzare
old_biogeoch_train_dataset, _, _ = Normalization(old_biogeoch_train_dataset, "1p")   #questa phase è da cambiare, RICORDA



#sottocampiono i tensori biogeochimici per arrivare ad una cardinalità simile a quella dei dati float (ovviamente calcolati a livello settimanale)
#biogeoch_train_dataset = [generate_sample_tensors(biogeoch_train_dataset[i], 10, 1000) for i in range(len(biogeoch_train_dataset))]
biogeoch_train_dataset = generate_list_sample_tensors(biogeoch_train_dataset, land_sea_masks[0], 100, 10)   #ovviamente questi parametri sono da risettare pois


#fill missing values of biogeoch tensor with standard values
biogeoch_train_dataset = [fill_tensor_with_standard(biogeoch_tensor, land_sea_masks, 200) for biogeoch_tensor in biogeoch_train_dataset]

#merging part
train_dataset = [concatenate_tensors(physics_train_dataset[i], biogeoch_train_dataset[i], axis=1) for i in range(len(physics_train_dataset))]

#train_dataset = get_list_model_tensor()      #Si trova in get_datasets.py --> Partendo dal dataset crea il training dataset
#print("get list model tensor done")

#generate test daatset
index_testing = random.sample(range(len(train_dataset)), int(len(train_dataset) * 0.2))  #Qua bisogna cambiare il modo in cui fare il test

train_dataset, _, _ = Normalization(train_dataset, "1p")    
test_dataset = [train_dataset[i] for i in index_testing]                         
for i in range(len(index_testing)-1, -1, -1):
    #train_index = train_dataset.index(test_dataset[i])
    #train_dataset.remove(test_dataset[i])
    train_dataset.pop(i)



#implement the mean pixel value technique
mean_value_pixel = MV_pixel(train_dataset)  # compute the mean of the channel of the training set
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

alpha = torch.tensor(4e-4)          #qua ho vari parametri, in caso ne dovrò solo cambiare i valori

lr_c = 0.001            #learning rate per la rete Completion

epoch_c = 500  # number of step for the first phase of training  --> riguarda la rete Completion
snaperiod = 5



#SCRIVIAMO I NUOVI PATH IN CUI SALVARE TUTTI I RISULTATI CHE OTTENIAMO
# make directory      #partendo dal path già creato ad inizio script, vado avanti a completarlo, per poter salvare in una directory appropriata i risultati di ogni epoch
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
    for training_x in train_dataset:
        training_x = training_x.to(device)
        #mask = generate_input_mask(
         #   shape=(training_x.shape[0], 1, training_x.shape[2], training_x.shape[3], training_x.shape[4]), n_points=1900)  #,
          #  #hole_size=(hole_min_d1, hole_max_d1, hole_min_h1, hole_max_h1, hole_min_w1, hole_max_w1))
        #mask = mask.to(device)
        #training_x_masked = training_x - training_x * mask + mean_value_pixel * mask  # mask the training tensor with  --> STA ROBA QUI LA TOGLIEREI 
        #input = torch.cat((training_x_masked, mask), dim=1)     #qua sto aumentanod la dimensione del tensore se aggiungo così la maschera
        input = training_x
        model_completion = model_completion.to(device)
        output = model_completion(input.float())


        #porto training_x su GPU()
        #loss_completion = completion_network_loss(training_x, output, mask)  #QUA PRIMA C'ERA ANCHE MASK COME INPUT DELLA FUNZIONE LOSS
        loss_completion = convolutional_network_loss(training_x, output, land_sea_masks)
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

        with torch.no_grad():

            for test_data in test_dataset:  
                test_data.to(device)   
                #training_mask = generate_input_mask(
                 #   shape=(test_data.shape[0], 1, test_data.shape[2], test_data.shape[3], test_data.shape[4]) , n_points=1900)  #,
                # # #hole_size=(hole_min_d1, hole_max_d1, hole_min_h1, hole_max_h1, hole_min_w1, hole_max_w1))
                #training_mask = training_mask.to(device)       
                #testing_x_mask = test_data - test_data * training_mask + mean_value_pixel * training_mask
                testing_input = test_data  #torch.cat((testing_x_mask, training_mask), dim=1)
                testing_output = model_completion(testing_input.float())

                loss_1c_test = convolutional_network_loss(test_data, testing_output, land_sea_masks)  #completion_network_loss(test_data, testing_output, training_mask)
                losses_1_c_test.append(loss_1c_test)

                print(f"[EPOCH]: {ep + 1}, [TEST LOSS]: {loss_1c_test.item():.12f}")
                f_test.write(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_completion.item():.12f} \n")

            test_loss = np.mean(np.array(losses_1_c_test))      #test loss media di questa epoca
            test_losses.append(test_loss)


    if ep % snaperiod == 0:  # save the partial model
        torch.save(model_completion.state_dict(), path_model + "/ep_" + str(ep + epoch_pretrain) + ".pt")

f.close()
f_test.close()

Plot_Error(test_losses, "1p", path_losses + "/")

#torch.save(model_completion.state_dict(), path_lr + "/ep_" + str(epoch_c + epoch_pretrain) + ".pt")
torch.save(model_completion.state_dict(), path_lr + "/final_model" + ".pt")


#Save maps of 1 phase: difference between output of the NN model (prediction of biogeoch var) and var biogeoch computed by numerical model
path_difference_maps_1p = path_plots + "/maps_1p"
var = "O2o"
model_completion.eval()
with torch.no_grad():
    for i in range(len(test_dataset)):
        test_data = test_dataset[i]
        #training_mask = generate_input_mask(
        #        shape=(test_data.shape[0], 1, test_data.shape[2], test_data.shape[3], test_data.shape[4]) , n_points=1900)  
        #training_mask = training_mask.to(device)       
        #testing_x_mask = test_data - test_data * training_mask + mean_value_pixel * training_mask
        testing_input = test_data  #torch.cat((testing_x_mask, training_mask), dim=1)
        testing_output = model_completion(testing_input.float())

        path_difference_map_1p = path_difference_maps_1p + "/test_data_" + str(i)
        if not os.path.exists(path_difference_map_1p):
            os.makedirs(path_difference_map_1p)
        plot_difference_maps(torch.unsqueeze(testing_input[:, 6, :, :, :], 1), testing_output, land_sea_masks, var, path_difference_map_1p)




#PHASE 2
physics_train_dataset_2 = get_list_model_tensor("physics_vars")[:10]
biogeoch_float_train_dataset = get_list_float_tensor("O2o")[:10]   #DA CONTROLLARE PERCHE LUI è UN FLOAT, E FORSE SI COMPORTA DIVERSAMENTE

#find the coordinates of profiles of float, for each week 
list_float_profiles_coordinates = [compute_profile_coordinates(float_tensor[:, 0:1, :, :, :]) for float_tensor in biogeoch_float_train_dataset]

#fill in the missing values of float data with standard value
biogeoch_float_train_dataset = [fill_tensor_with_standard(float_tensor[:, 0:1, :, :, :], land_sea_masks, 200) for float_tensor in biogeoch_float_train_dataset]


#merging part
#QUESTA COSA VA BENE ORA PERCHE HO I FLOAT CON LA dim 1 SBAGLIATA --> poi sta roba si deve togliere e rimettere normale
train_dataset_2 = [concatenate_tensors(physics_train_dataset_2[i], biogeoch_float_train_dataset[i][:, 0:1, :, :, :], axis=1) for i in range(len(physics_train_dataset_2))]

#train_dataset = get_list_model_tensor()      #Si trova in get_datasets.py --> Partendo dal dataset crea il training dataset
#print("get list model tensor done")

#generate test daatset
index_testing_2 = random.sample(range(len(train_dataset_2)), int(len(train_dataset_2) * 0.2))  #Qua bisogna cambiare il modo in cui fare il test
index_training_2 = [i for i in range(len(train_dataset_2)) if i not in index_testing_2]

train_dataset_2, _, _ = Normalization(train_dataset_2, "2p")    
test_dataset_2 = [train_dataset_2[i] for i in index_testing_2]                         
for i in range(len(index_testing_2)-1, -1, -1):
    train_dataset_2.pop(i)



# HYPERPARAMETERS
pretrain_2 = 1  # 0 means that we don"t use pretrained model to fine tuning
#ora non ho il pretrain quindi metto pretrain = 0
pretrain_2 = 0
#e aggiungo anceh epoch_pretrain 
epoch_pretrain_2 = 0


model_completion_2 = CompletionN()     #modo per istanziare un oggetto della classe Completion --> quindi ora ho un oggetto che è una rete
model_completion_2.load_state_dict(torch.load(path_lr + "/final_model" + ".pt"))
model_completion_2.eval()     #lui non mi convince !! --> QUANDO METTO EVAL IL MODELLO PUò ANCORA ESSERE MODIFICATO?  ma Gloria lo aveva messo così 

alpha_2 = torch.tensor(4e-4)          
lr_c_2 = 0.001           
epoch_c_2 = 500 
snaperiod_2 = 5



#SCRIVIAMO I NUOVI PATH IN CUI SALVARE TUTTI I RISULTATI CHE OTTENIAMO
# make directory      #partendo dal path già creato ad inizio script, vado avanti a completarlo, per poter salvare in una directory appropriata i risultati di ogni epoch
path_results_2 = "results_training_2" + "_" + str(datetime.datetime.utcnow())
path_configuration_2 = path_results_2 + "/" + str(biogeoch_var_to_predict) + "/" + str(epoch_c_2 + epoch_pretrain_2) 
if not os.path.exists(path_configuration_2):
    os.makedirs(path_configuration_2)
path_lr_2 = path_configuration_2 + "/lrc_" + str(lr_c_2) 
if not os.path.exists(path_lr_2):
    os.makedirs(path_lr_2)
path_losses_2 = path_lr_2 + "/losses"
if not os.path.exists(path_losses_2):
    os.makedirs(path_losses_2)
path_model_2 = path_lr_2 + "/partial_models/"
if not os.path.exists(path_model_2):
    os.mkdir(path_model_2)

losses_1_c_2 = []
losses_1_c_test_2 = []
test_losses_2 = []

# PHASE 1   #la phase 1 del training consiste nel trainare solo la Completion

optimizer_completion_2 =torch.optim.Adam(model_completion_2.parameters(), lr=lr_c_2)   #questo è l'optimizer che uso per questa rete --> che implementa praticamente una stochastic gradient descend

f_2, f_test_2 = open(path_losses_2 + "/train_loss.txt", "w+"), open(path_losses_2 + "/test_loss.txt", "w+")

for ep in range(epoch_c_2):
    if ep > 400:     #proposto nel paper da cui sto prendendo l'architettura
        lr_c_2 = 0.0001

    #PHASE OF TRAINING
    for training_x in train_dataset_2:
        training_x = training_x.to(device)
        #mask = generate_input_mask(
         #   shape=(training_x.shape[0], 1, training_x.shape[2], training_x.shape[3], training_x.shape[4]), n_points=1900)  #,
          #  #hole_size=(hole_min_d1, hole_max_d1, hole_min_h1, hole_max_h1, hole_min_w1, hole_max_w1))
        #mask = mask.to(device)
        #training_x_masked = training_x - training_x * mask + mean_value_pixel * mask  # mask the training tensor with
        input = training_x   #torch.cat((training_x_masked, mask), dim=1)     #qua sto aumentanod la dimensione del tensore se aggiungo così la maschera
        model_completion_2 = model_completion_2.to(device)
        output = model_completion_2(input.float())


        #porto training_x su GPU() 
        loss_completion_2 = convolutional_network_loss(input, output, land_sea_masks)
        #float_coord_mask = generate_float_mask(compute_profile_coordinates(training_x))
        #loss_completion_2 = convolutional_network_float_loss(input, output, land_sea_masks, float_coord_mask)
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

        with torch.no_grad():

            for test_data in test_dataset_2:  
                test_data.to(device)   
                #training_mask = generate_input_mask(
                 #   shape=(test_data.shape[0], 1, test_data.shape[2], test_data.shape[3], test_data.shape[4]) , n_points=1900)  #,
                # # #hole_size=(hole_min_d1, hole_max_d1, hole_min_h1, hole_max_h1, hole_min_w1, hole_max_w1))
                #training_mask = training_mask.to(device)       
                #testing_x_mask = test_data - test_data * training_mask + mean_value_pixel * training_mask
                testing_input = test_data   #torch.cat((testing_x_mask, training_mask), dim=1)
                testing_output = model_completion_2(testing_input.float())

                #loss_1c_test_2 = completion_network_loss(test_data, testing_output, training_mask)
                loss_1c_test_2 = convolutional_network_loss(test_data, testing_output, land_sea_masks)
                #float_coord_mask = generate_float_mask(compute_profile_coordinates(test_data))    #list_float_profiles_coordinates[index_testing_2[i]])
                #loss_1c_test_2 = convolutional_network_float_loss(test_data, testing_output, land_sea_masks, float_coord_mask)
                losses_1_c_test_2.append(loss_1c_test_2)

                print(f"[EPOCH]: {ep + 1}, [TEST LOSS]: {loss_1c_test_2.item():.12f}")
                f_test_2.write(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_1c_test_2.item():.12f} \n")

            test_loss_2 = np.mean(np.array(losses_1_c_test_2))      #test loss media di questa epoca
            test_losses_2.append(test_loss_2)


    if ep % snaperiod_2 == 0:  # save the partial model
        torch.save(model_completion_2.state_dict(), path_model_2 + "/ep_" + str(ep + epoch_pretrain_2) + ".pt")

f_2.close()
f_test_2.close()

Plot_Error(test_losses_2, "2p", path_losses_2 + "/")

#torch.save(model_completion.state_dict(), path_lr + "/ep_" + str(epoch_c + epoch_pretrain) + ".pt")
torch.save(model_completion_2.state_dict(), path_lr_2 + "/final_model" + ".pt")


#Save maps of 2 phase: difference between output of the NN model (prediction of biogeoch var) and var biogeoch computed by numerical model
path_difference_maps_2p = path_plots + "/maps_2p"
var = "O2o"
model_completion_2.eval()
with torch.no_grad():
    for i in range(len(test_dataset_2)):
        test_data_2 = test_dataset_2[i]
        #training_mask = generate_input_mask(
         #       shape=(test_data_2.shape[0], 1, test_data_2.shape[2], test_data_2.shape[3], test_data_2.shape[4]) , n_points=1900)  
        #training_mask = training_mask.to(device)       
        #testing_x_mask = test_data_2 - test_data_2 * training_mask + mean_value_pixel * training_mask
        testing_input_2 = test_data_2  #torch.cat((testing_x_mask, training_mask), dim=1)
        testing_output_2 = model_completion_2(testing_input_2.float())

        path_difference_map_2p = path_difference_maps_2p + "/test_data_" + str(i)
        if not os.path.exists(path_difference_map_2p):
            os.makedirs(path_difference_map_2p)
        plot_difference_maps(torch.unsqueeze(testing_input_2[:, 6, :, :, :], 1), testing_output_2, land_sea_masks, var, path_difference_map_2p)


#Save profiles of 2 phase: comparison between the profile of float vs the extraction of the SAME profile of NN model vs the extraction of the SAME profile of num model 
path_profiles = path_plots + "/profiles"
model_completion_2.eval()
with torch.no_grad():
    for i in range(len(test_dataset_2)):
        test_data_2 = test_dataset_2[i]
        #training_mask = generate_input_mask(
         #       shape=(test_data_2.shape[0], 1, test_data_2.shape[2], test_data_2.shape[3], test_data_2.shape[4]) , n_points=1900)  
        #training_mask = training_mask.to(device)       
        #testing_x_mask = test_data_2 - test_data_2 * training_mask + mean_value_pixel * training_mask
        testing_input_2 = test_data_2  #torch.cat((testing_x_mask, training_mask), dim=1)
        testing_output_2 = model_completion_2(testing_input_2.float())

        path_profiles_test_data = path_profiles + "/test_data_" + str(i)
        if not os.path.exists(path_profiles_test_data):
            os.makedirs(path_profiles_test_data)
        
        plot_models_profiles(torch.unsqueeze(testing_input_2[:, 6, :, :, :], 1), testing_output_2, old_biogeoch_train_dataset[index_testing_2[i]], var, path_profiles_test_data)   #IL TENSORE DEL MODELLO NUMERICO CORRISPONDENTE NON SAPREI COME RITROVARLO





print("model state dict", model_completion_2.state_dict().keys())
print("model conv1 weights", model_completion_2.state_dict()['conv1.weight'].size())
print("model conv1 weights", torch.sum(model_completion_2.state_dict()['conv1.weight'].isnan() == True))
print("model conv1 bias", model_completion_2.state_dict()['conv1.bias'].size())
print("model conv1 bias", torch.sum(model_completion_2.state_dict()['conv1.bias'].isnan() == True))
print("losses_2c_test", losses_1_c_test_2)
print("len loss 2c test", len(losses_1_c_test_2))
#Plot_Error(losses_1_c_test[0].cpu(), "1c", path_lr + "/")   #modificato per tornare alle cpu, prima era senza [0].cpu() --> non sono convinta che funzioni
print("end")