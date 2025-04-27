"""
Implementation of the 1 and 2 steps training and testing functions
"""

import torch
from CNN_3DMedSea.convolutional_network import CompletionN
from CNN_3DMedSea.normalization_functions import Denormalization
from CNN_3DMedSea.losses import convolutional_network_exp_weighted_loss, convolutional_network_float_exp_weighted_loss
from CNN_plots.plot_error import Plot_Error
from CNN_plots.plot_results import plot_models_profiles_1p, plot_NN_maps, comparison_profiles_1_2_phases, plot_NN_maps_layer_mean
from utils.utils_general import *
from utils.utils_dataset_generation import write_list, read_list
from utils.utils_mask import generate_float_mask
from utils.utils_training import load_old_total_tensor


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def training_1p(n_epochs_1p, snaperiod, l_r, years_week_dupl_indexes,  my_mean_tensor, my_std_tensor,train_dataset, internal_test_dataset, index_training, index_internal_test, land_sea_masks, exp_weights, f, f_test, f_job_dev, losses_1p, train_losses_1p, test_losses_1p, model_save_path, path_model, path_losses, path_lr, transposed_lat_coordinates):
    """this function describes the procedure of training 1p"""
    model_1p = CompletionN()
    try:
        checkpoint = torch.load(model_save_path + "/" + 'model_checkpoint.pth', map_location=device)
        model_1p.load_state_dict(checkpoint['model_state_dict'])
        optimizer_1p = torch.optim.Adam(model_1p.parameters(), lr=l_r)    
        optimizer_1p.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer_1p.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_epoch = checkpoint['epoch']
        loss_1p = checkpoint['loss']
        #check of weights
        mismatch = False
        for (name1, param1), (name2, param2) in zip(checkpoint['model_state_dict'].items(), model_1p.state_dict().items()):
            if not torch.equal(param1.cpu(), param2.cpu()):
                print(f"Mismatch in parameter: {name1}")
                mismatch = True
        if not mismatch:
            print("All model parameters match exactly!")
        #otehr prints
        train_losses_1p = read_list(path_losses + "/train_losses_1p.txt")
        test_losses_1p = read_list(path_losses + "/test_losses_1p.txt")
    except FileNotFoundError:
        print("No checkpoint found, starting fresh.")
        start_epoch = 0
        optimizer_1p = torch.optim.Adam(model_1p.parameters(), lr=l_r) 
    print("start epoch = ", start_epoch, flush = True)
    # Set model to training mode
    model_1p.train()     
    #training loop
    num_epochs = n_epochs_1p
    for epoch in range(start_epoch, num_epochs):
        losses_1p = []
        for i in range(len(train_dataset)):
            training_x = train_dataset[i]
            input = training_x.to(device)
            model_1p = model_1p.to(device)
            output = model_1p(input.float())
            denormalized_NN_output = Denormalization(output, my_mean_tensor, my_std_tensor).to(device)
            #load the corresponding tensor of old_total_dataset
            biog_input = load_old_total_tensor("dataset_training/old_total_dataset/", index_training[i], years_week_dupl_indexes)
            biog_input = biog_input.to(device)
            exp_weights = exp_weights.to(device)
            land_sea_masks = [mask.to(device) for mask in land_sea_masks]
            loss_1p = convolutional_network_exp_weighted_loss(biog_input[:, :, :-1, :, 1:-1].float(), denormalized_NN_output.float(), land_sea_masks, exp_weights)
            losses_1p.append(loss_1p.item())
            print(f"[EPOCH]: {epoch + 1}, [LOSS]: {loss_1p.item():.12f}")
            f.write(f"[EPOCH]: {epoch + 1}, [LOSS]: {loss_1p.item():.12f} \n")
            #call and update the optimizer
            optimizer_1p.zero_grad()
            loss_1p.backward()
            optimizer_1p.step()
            #remove the data from the gpu
            del training_x
            del output
            del biog_input
            torch.cuda.empty_cache()
        train_loss = np.mean(np.array(losses_1p))      
        train_losses_1p.append(train_loss)
        #internal testing
        if (epoch+1) % snaperiod == 0:
            model_1p.eval()
            losses_1_c_test = []
            with torch.no_grad():
                for i_test in range(len(internal_test_dataset)): 
                    test_data = internal_test_dataset[i_test] 
                    testing_input = test_data.to(device)   
                    testing_output = model_1p(testing_input.float())
                    denormalized_testing_output = Denormalization(testing_output, my_mean_tensor, my_std_tensor).to(device)
                    biog_input = load_old_total_tensor("dataset_training/old_total_dataset/", index_internal_test[i_test], years_week_dupl_indexes).to(device)
                    loss_1c_test = convolutional_network_exp_weighted_loss(biog_input[:, :, :-1, :, 1:-1].float(), denormalized_testing_output.float(), land_sea_masks, exp_weights)
                    losses_1_c_test.append(loss_1c_test.cpu())
                    print(f"[EPOCH]: {epoch + 1}, [TEST LOSS]: {loss_1c_test.item():.12f}")
                    f_test.write(f"[EPOCH]: {epoch + 1}, [LOSS]: {loss_1c_test.item():.12f} \n")
                    #PARTE NUOVA PER VEDERE ALCUNI PROFILI PARZIALI
                    #year, week = years_week_dupl_indexes[index_internal_test[i_test]][0], years_week_dupl_indexes[index_internal_test[i_test]][1]
                    #print("year and week", [year, week])
                    #if week < 21 and year == 2019:  #così non salvo tutti tutti i profili
                    #    path_partial_profiles = path_lr + "/partial_plots"
                    #    if not os.path.exists(path_partial_profiles):
                    #        os.makedirs(path_partial_profiles)
                    #    path_partial_profiles_epoch = path_partial_profiles + "/epoch_" + str(epoch + 1)
                    #    if not os.path.exists(path_partial_profiles_epoch):
                    #        os.makedirs(path_partial_profiles_epoch)
                    #    path_partial_profiles_1p = path_partial_profiles_epoch + "/partial_test_" + str(year) + "_week_" + str(week)
                    #    if not os.path.exists(path_partial_profiles_1p):
                    #        os.makedirs(path_partial_profiles_1p)
                    #    plot_models_profiles_1p(torch.unsqueeze(biog_input[:, :, :-1, :, 1:-1][:, 6, :, :, :], 1), denormalized_testing_output, torch.unsqueeze(biog_input[:, :, :-1, :, 1:-1][:, 6, :, :, :], 1),  
                    #                "P_l", path_partial_profiles_1p, transposed_lat_coordinates[index_internal_test[i_test]]) 
                    #del test_data
                    #del denormalized_testing_output
                    #del biog_input
                    #torch.cuda.empty_cache()
                test_loss = np.mean(np.array(losses_1_c_test))      
                test_losses_1p.append(test_loss)
        if (epoch+1) % snaperiod == 0: 
            torch.save(model_1p.state_dict(), path_model + "/ep_" + str(epoch) + ".pt")
        print("current epoch = ", epoch + 1, flush = True)
        # Save checkpoint after each epoch
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model_1p.state_dict(),
            'optimizer_state_dict': optimizer_1p.state_dict(),
            'loss': loss_1p,
        }, model_save_path + '/' + 'model_checkpoint.pth')
        #Save the list of train and test losses --> useful only to obtain a complete plot of the loss behavior at the end
        write_list(train_losses_1p, path_losses + "/train_losses_1p.txt")
        write_list(test_losses_1p, path_losses + "/test_losses_1p.txt")
    #Plot_Error([train_losses_1p[i] for i in range(len(train_losses_1p)) if (i+1)%10 == 0], n_epochs_1p, int(n_epochs_1p / 10), "1p_train", path_losses + "/")
    #Plot_Error(test_losses_1p, n_epochs_1p, snaperiod, "1p", path_losses + "/")
    return None


def testing_1p(biogeoch_var, path_plots, years_week_dupl_indexes, model_1p, external_test_dataset, index_external_testing, land_sea_masks, transposed_lat_coordinates, my_mean_tensor, my_std_tensor):
    """"this function describes the procedure of training 1p"""
    var = biogeoch_var
    path_profiles = path_plots + "/profiles_1p"
    path_NN_reconstruction = path_plots + "/NN_maps"
    path_BFM_reconstruction = path_plots + "/BFM_maps"
    path_NN_mean_layer = path_plots + "/NN_maps_mean_layer"
    path_BFM_mean_layer = path_plots + "/BFM_maps_mean_layer"
    model_1p.to(device)
    model_1p.eval()
    with torch.no_grad():
        for i in range(len(external_test_dataset)):
            test_data = external_test_dataset[i]
            testing_input = test_data.to(device)
            testing_output = model_1p(testing_input.float())
            denorm_testing_input = Denormalization(testing_input, my_mean_tensor, my_std_tensor)
            denorm_testing_output = Denormalization(testing_output, my_mean_tensor, my_std_tensor)
            year, week = years_week_dupl_indexes[index_external_testing[i]][0], years_week_dupl_indexes[index_external_testing[i]][1]
            path_profiles_test_data = path_profiles + "/test_" + str(year) + "_week_" + str(week)
            if not os.path.exists(path_profiles_test_data):
                os.makedirs(path_profiles_test_data)
            path_NN_reconstruction_test_data = path_NN_reconstruction + "/test_" + str(year) + "_week_" + str(week)
            if not os.path.exists(path_NN_reconstruction_test_data):
                os.makedirs(path_NN_reconstruction_test_data)
            path_BFM_reconstruction_test_data = path_BFM_reconstruction + "/test_" + str(year) + "_week_" + str(week)
            if not os.path.exists(path_BFM_reconstruction_test_data):
                os.makedirs(path_BFM_reconstruction_test_data)
            path_NN_mean_layer_test_data = path_NN_mean_layer + "/test_" + str(year) + "_week_" + str(week)
            if not os.path.exists(path_NN_mean_layer_test_data):
                os.makedirs(path_NN_mean_layer_test_data)
            path_BFM_mean_layer_test_data = path_BFM_mean_layer + "/test_" + str(year) + "_week_" + str(week)
            if not os.path.exists(path_BFM_mean_layer_test_data):
                os.makedirs(path_BFM_mean_layer_test_data)
            plot_models_profiles_1p(torch.unsqueeze(denorm_testing_input[:, 6, :, :, :], 1), denorm_testing_output, torch.unsqueeze(load_old_total_tensor("dataset_training/old_total_dataset/", index_external_testing[i], years_week_dupl_indexes)[:, :, :-1, :, 1:-1][:, 6, :, :, :], 1),  
                                var, path_profiles_test_data, transposed_lat_coordinates[index_external_testing[i]]) 
            #plot_NN_maps(denorm_testing_output, land_sea_masks, var, path_NN_reconstruction_test_data)
            #plot_NN_maps(torch.unsqueeze(load_old_total_tensor("dataset_training/old_total_dataset/", index_external_testing[i], years_week_dupl_indexes)[:, :, :-1, :, 1:-1][:, 6, :, :, :], 1).to(device), land_sea_masks, var, path_BFM_reconstruction_test_data)
            plot_NN_maps_layer_mean(denorm_testing_output, land_sea_masks, var, path_NN_mean_layer_test_data, [0, 40, 80, 120, 180, 300])
            plot_NN_maps_layer_mean(torch.unsqueeze(load_old_total_tensor("dataset_training/old_total_dataset/", index_external_testing[i], years_week_dupl_indexes)[:, :, :-1, :, 1:-1][:, 6, :, :, :], 1), land_sea_masks, var, path_BFM_mean_layer_test_data, [0, 40, 80, 120, 180, 300])
            #remove all the tensors from the gpu 
            del test_data
            del denorm_testing_input
            del denorm_testing_output
            torch.cuda.empty_cache()
    #remove the mean and standard deviation from the gpu
    del my_mean_tensor
    del my_std_tensor
    torch.cuda.empty_cache()
    return None


def training_2p(n_epochs_2p, snaperiod_2, l_r, my_mean_tensor_2, my_std_tensor_2, train_dataset_2, internal_test_dataset_2, index_training_2, index_internal_test_2, land_sea_masks, exp_weights, old_float_total_dataset, sampled_list_float_profile_coordinates, f_2, f_test_2, losses_2p, train_losses_2p, test_losses_2p, model_save_path, model_save_path_2, path_model_2, path_losses_2):
    """this function describes the procedure of training 1p"""
    model_2p = CompletionN()
    checkpoint = torch.load(model_save_path + "/" + 'model_checkpoint.pth', map_location=device)
    model_2p.load_state_dict(checkpoint['model_state_dict'])   
    model_2p.eval() 
    optimizer_2p =torch.optim.Adam(model_2p.parameters(), lr=l_r)
    try:
        checkpoint = torch.load(model_save_path_2 + "/" + 'model_checkpoint_2.pth')
        model_2p.load_state_dict(checkpoint['model_state_dict'])
        optimizer_2p.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer_2p.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_epoch = checkpoint['epoch']
        loss_2p = checkpoint['loss']
        print(f"Resuming from epoch {start_epoch}")
        train_losses_2p = read_list(path_losses_2 + "/train_losses_2p.txt")
        test_losses_2p = read_list(path_losses_2 + "/test_losses_2p.txt")
    except FileNotFoundError:
        print("No checkpoint found, starting fresh.")
        start_epoch = 0
    # Set model to training mode
    model_2p.train()    
    # Example training loop
    num_epochs = n_epochs_2p
    for epoch in range(start_epoch, num_epochs):
        losses_2p = []
        #training loop (with validation)
        for i in range(len(train_dataset_2)):
            training_x = train_dataset_2[i]
            input = training_x.to(device)
            model_2p = model_2p.to(device)
            output = model_2p(input.float())
            #loss computation on float profiles 
            denormalized_output = Denormalization(output, my_mean_tensor_2, my_std_tensor_2).to(device)
            float_tensor_input = old_float_total_dataset[index_training_2[i]].to(device)  
            float_coord_mask = generate_float_mask(sampled_list_float_profile_coordinates[index_training_2[i]]).to(device)    
            loss_2p = convolutional_network_float_exp_weighted_loss(float_tensor_input.float(), denormalized_output.float(), land_sea_masks, float_coord_mask, exp_weights.to(device))
            losses_2p.append(loss_2p.item())
            print(f"[EPOCH]: {epoch + 1}, [LOSS]: {loss_2p.item():.12f}")
            f_2.write(f"[EPOCH]: {epoch + 1}, [LOSS]: {loss_2p.item():.12f} \n")
            #call and update the optimizer
            optimizer_2p.zero_grad()
            loss_2p.backward()
            optimizer_2p.step()
            del training_x
            del denormalized_output
            del float_tensor_input
            del float_coord_mask
            torch.cuda.empty_cache()
        train_loss_2 = np.mean(np.array(losses_2p))      
        train_losses_2p.append(train_loss_2)
        #testing phase
        if (epoch+1) % snaperiod_2 == 0:
            model_2p.eval()
            losses_1_c_test_2 = []
            with torch.no_grad():
                for i_test in range(len(internal_test_dataset_2)):
                    test_data = internal_test_dataset_2[i_test]  
                    test_data = test_data.to(device)   
                    testing_input = test_data  
                    testing_output = model_2p(testing_input.float())
                    denormalized_output = Denormalization(testing_output, my_mean_tensor_2, my_std_tensor_2).to(device)
                    float_tensor_input = old_float_total_dataset[index_internal_test_2[i_test]].to(device)
                    float_coord_mask = generate_float_mask(sampled_list_float_profile_coordinates[index_internal_test_2[i_test]]).to(device)
                    loss_1c_test_2 = convolutional_network_float_exp_weighted_loss(float_tensor_input.float(), denormalized_output.float(), land_sea_masks, float_coord_mask, exp_weights.to(device))
                    losses_1_c_test_2.append(loss_1c_test_2.cpu())
                    print(f"[EPOCH]: {epoch + 1}, [TEST LOSS]: {loss_1c_test_2.item():.12f}")
                    f_test_2.write(f"[EPOCH]: {epoch + 1}, [LOSS]: {loss_1c_test_2.item():.12f} \n")
                    del test_data
                    del denormalized_output
                    del float_tensor_input
                    del float_coord_mask
                    torch.cuda.empty_cache()
                test_loss_2 = np.mean(np.array(losses_1_c_test_2))      
                test_losses_2p.append(test_loss_2)
        if (epoch+1) % snaperiod_2 == 0:  
            torch.save(model_2p.state_dict(), path_model_2 + "/ep_" + str(epoch) + ".pt")
        # Save checkpoint after each epoch
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model_2p.state_dict(),
            'optimizer_state_dict': optimizer_2p.state_dict(),
            'loss': loss_2p,
        }, model_save_path_2 + '/' + 'model_checkpoint_2.pth')
        #Save the list of train and test losses --> useful only to obtain a complete plot of the loss behavior at the end
        write_list(train_losses_2p, path_losses_2 + "/train_losses_2p.txt")
        write_list(test_losses_2p, path_losses_2 + "/test_losses_2p.txt")
    Plot_Error(train_losses_2p, n_epochs_2p, 1, "2p_train", path_losses_2 + "/")
    Plot_Error(test_losses_2p, n_epochs_2p, snaperiod_2, "2p", path_losses_2 + "/")
    return None


def testing_2p(biogeoch_var, path_plots_2, years_week_dupl_indexes, biogeoch_train_dataset, old_float_total_dataset, model_1p, model_2p, external_test_dataset_2, index_external_testing_2, land_sea_masks, list_float_profiles_coordinates, my_mean_tensor_2, my_std_tensor_2):
    """this function describes the procedure of training 2p"""
    path_profiles_with_NN_1 = path_plots_2 + "/profiles_2p_NN_1"
    path_NN_reconstruction = path_plots_2 + "/NN_maps"
    path_NN_phases_diff = path_plots_2 + "/NN_phases_diff"
    path_NN_phases_diff_season = path_plots_2 + "/NN_phases_diff_season"
    model_2p.to(device)
    model_2p.eval()
    with torch.no_grad():
        for i in range(len(external_test_dataset_2)):
            test_data_2 = external_test_dataset_2[i]
            test_data_2 = test_data_2.to(device)
            testing_input_2 = test_data_2 
            testing_output_2 = model_2p(testing_input_2.float())
            denorm_testing_output_2 = Denormalization(testing_output_2, my_mean_tensor_2, my_std_tensor_2)
            exp_weights = torch.ones([1, 1, d-2, h, w-2])
            float_tensor_input = old_float_total_dataset[index_external_testing_2[i]].to(device)
            float_coord_mask = generate_float_mask(compute_profile_coordinates(float_tensor_input[:, 0:1, :, :, :])).to(device)
            l2 = convolutional_network_float_exp_weighted_loss(float_tensor_input.float(), denorm_testing_output_2.float(), land_sea_masks, float_coord_mask, exp_weights.to(device))
            norm_testing_output_1 = model_1p(testing_input_2.cpu().float()).to(device) 
            denorm_testing_output_1 = Denormalization(norm_testing_output_1, my_mean_tensor_2, my_std_tensor_2) 
            l1 = convolutional_network_float_exp_weighted_loss(float_tensor_input.float(), biogeoch_train_dataset[i][:, :, :-1, :, 1:-1].to(device).float(), land_sea_masks, float_coord_mask, exp_weights.to(device))
            path_profiles_test_data_NN_1 = path_profiles_with_NN_1 + "/year_" + str(years_week_dupl_indexes[index_external_testing_2[i]][0]) + "_week_" + str(years_week_dupl_indexes[index_external_testing_2[i]][1])
            if not os.path.exists(path_profiles_test_data_NN_1):
                os.makedirs(path_profiles_test_data_NN_1)
            path_NN_reconstruction_test_data = path_NN_reconstruction + "/year_" + str(years_week_dupl_indexes[index_external_testing_2[i]][0]) + "_week_" + str(years_week_dupl_indexes[index_external_testing_2[i]][1])
            if not os.path.exists(path_NN_reconstruction_test_data):
                os.makedirs(path_NN_reconstruction_test_data)
            path_NN_diff_test_data = path_NN_phases_diff + "/year_" + str(years_week_dupl_indexes[index_external_testing_2[i]][0]) + "_week_" + str(years_week_dupl_indexes[index_external_testing_2[i]][1])
            if not os.path.exists(path_NN_diff_test_data):
                os.makedirs(path_NN_diff_test_data)
            path_NN_diff_season_test_data = path_NN_phases_diff_season + "/year_" + str(years_week_dupl_indexes[index_external_testing_2[i]][0]) + "_week_" + str(years_week_dupl_indexes[index_external_testing_2[i]][1])
            if not os.path.exists(path_NN_diff_season_test_data):
                os.makedirs(path_NN_diff_season_test_data)
            comparison_profiles_1_2_phases(torch.unsqueeze(old_float_total_dataset[index_external_testing_2[i]][:, 6, :, :, :], 1) , denorm_testing_output_2, biogeoch_train_dataset[i][:, :, :-1, :, 1:-1], denorm_testing_output_1,
                                biogeoch_var, path_profiles_test_data_NN_1)
            plot_NN_maps(denorm_testing_output_2, land_sea_masks, biogeoch_var, path_NN_reconstruction_test_data)
            del test_data_2
            torch.cuda.empty_cache()
    del my_mean_tensor_2
    del my_std_tensor_2
    torch.cuda.empty_cache()
    return None


def testing_2p_ensemble(biogeoch_var, path_plots_2, years_week_dupl_indexes, biogeoch_train_dataset, old_float_total_dataset, model_1p, model_2p, external_test_dataset_2, index_external_testing_2, land_sea_masks, list_float_profiles_coordinates, sampled_list_float_profile_coordinates,my_mean_tensor_2, my_std_tensor_2, exp_weights, path_losses_2p):
    """this function describes the procedure of training 2p"""
    path_profiles_with_NN_1 = path_plots_2 + "/profiles_2p_NN_1"
    path_NN_reconstruction = path_plots_2 + "/NN_maps"
    path_NN_phases_diff = path_plots_2 + "/NN_phases_diff"
    path_NN_phases_diff_season = path_plots_2 + "/NN_phases_diff_season"
    model_2p.to(device)
    model_2p.eval()
    test_loss_list = []
    test_loss_list_winter = []
    test_loss_list_summer = []
    with torch.no_grad():
        for i in range(len(external_test_dataset_2)):
            test_data_2 = external_test_dataset_2[i]
            test_data_2 = test_data_2.to(device)
            testing_input_2 = test_data_2 
            testing_output_2 = model_2p(testing_input_2.float())
            denorm_testing_output_2 = Denormalization(testing_output_2, my_mean_tensor_2, my_std_tensor_2)
            norm_testing_output_1 = model_1p(testing_input_2.cpu().float()).to(device) 
            denorm_testing_output_1 = Denormalization(norm_testing_output_1, my_mean_tensor_2, my_std_tensor_2)
            #compute loss and save in 
            float_tensor_input = old_float_total_dataset[index_external_testing_2[i]].to(device)
            float_coord_mask = generate_float_mask(sampled_list_float_profile_coordinates[index_external_testing_2[i]]).to(device)
            test_loss = convolutional_network_float_exp_weighted_loss(float_tensor_input.float(), denorm_testing_output_2.float(), land_sea_masks, float_coord_mask, exp_weights.to(device))
            test_loss_list.append(test_loss)
            #compute the week and write the loss into the file of the corresponding season
            week = years_week_dupl_indexes[i][1]
            if week < 14:
                test_loss_list_winter.append(test_loss)
            else:
                test_loss_list_summer.append(test_loss)
            path_profiles_test_data_NN_1 = path_profiles_with_NN_1 + "/year_" + str(years_week_dupl_indexes[index_external_testing_2[i]][0]) + "_week_" + str(years_week_dupl_indexes[index_external_testing_2[i]][1])
            if not os.path.exists(path_profiles_test_data_NN_1):
                os.makedirs(path_profiles_test_data_NN_1)
            path_NN_reconstruction_test_data = path_NN_reconstruction + "/year_" + str(years_week_dupl_indexes[index_external_testing_2[i]][0]) + "_week_" + str(years_week_dupl_indexes[index_external_testing_2[i]][1])
            if not os.path.exists(path_NN_reconstruction_test_data):
                os.makedirs(path_NN_reconstruction_test_data)
            path_NN_diff_season_test_data = path_NN_phases_diff_season + "/year_" + str(years_week_dupl_indexes[index_external_testing_2[i]][0]) + "_week_" + str(years_week_dupl_indexes[index_external_testing_2[i]][1])
            if not os.path.exists(path_NN_diff_season_test_data):
                os.makedirs(path_NN_diff_season_test_data)
            comparison_profiles_1_2_phases(torch.unsqueeze(old_float_total_dataset[index_external_testing_2[i]][:, 6, :, :, :], 1) , denorm_testing_output_2, biogeoch_train_dataset[i][:, :, :-1, :, 1:-1], denorm_testing_output_1,
                                biogeoch_var, path_profiles_test_data_NN_1)
            plot_NN_maps(denorm_testing_output_2, land_sea_masks, biogeoch_var, path_NN_reconstruction_test_data)
            del test_data_2
            torch.cuda.empty_cache()
    write_list(test_loss_list, path_losses_2p + "/test_loss_final.txt")
    write_list(test_loss_list_winter, path_losses_2p + "/test_loss_final_winter.txt")
    write_list(test_loss_list_summer, path_losses_2p + "/test_loss_final_summer.txt")
    del my_mean_tensor_2
    del my_std_tensor_2
    torch.cuda.empty_cache()
    return None
