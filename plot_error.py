"""
Plotting the error during the different phase of the training
"""
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

#update for GPU

def Plot_Error(losses, n_epochs, snaperiod, flag, path):
    """
    plot of the losses
    flag : what phase of the training we are plotting
    path : where to save the plot
    """
    flag_dict = {'1p': 'model loss with BFM data on test set',
                 '1p_train' : 'model loss with BFM data on train set',
                 '2p': 'model loss with Argo-float data on test set', 
                 '2p_train': 'model loss with Argo-float data on train set'}

    descr = flag_dict[flag]

    figure(figsize=(10, 6))

    label = 'losses'
    plt.plot(losses, 'orange')
    plt.plot(losses, 'm.', label=label)
    my_xticks = np.arange(0, n_epochs // snaperiod)
    my_xticks_label = np.arange(0, n_epochs, snaperiod)
    plt.xticks(my_xticks, my_xticks_label, fontsize=6)
    plt.xlabel('Number of epochs')
    plt.title('Losses of ' + descr)
    plt.legend()
    plt.savefig(path + "loss_" + str(flag) + ".png")
    plt.close()

    figure(figsize=(10, 6))

    label = 'log losses '
    plt.plot(np.log(losses), 'orange')
    plt.plot(np.log(losses), 'm.', label=label)
    my_xticks = np.arange(0, n_epochs // snaperiod)
    my_xticks_label = np.arange(0, n_epochs, snaperiod)
    plt.xticks(my_xticks, my_xticks_label, fontsize=6)
    plt.xlabel('Number of epochs')
    plt.title('Logarithmic Losses of ' + descr)
    plt.legend()
    plt.savefig(path + "LOGloss_" + str(flag) + ".png")
    plt.close()

    return None     #aggiunta di recente, prima non c'era
