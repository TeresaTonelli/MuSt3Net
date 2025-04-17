"""
definition of the COMPLETION NETWORK (fully CNN) that compose the architecture
"""
import torch.nn as nn
from hyperparameter import *

in_channels = number_channel  
out_channels = 1   


class CompletionN(nn.Module):
    def __init__(self):
        super(CompletionN, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, 9, kernel_size=(3,3,3), stride=1, padding=1)    #OSS: se mi da errore sul kernel size, devo mettere solo 3x3x3
        self.bn1 = nn.BatchNorm3d(9)                             #OSS: non penso volgia la normalizzazione nel paper originale, ma per ora la lascio
        self.af1 = nn.ReLU()
        self.do1 = nn.Dropout(0.15)

        self.conv2 = nn.Conv3d(9, 16, kernel_size=(3,3,3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(16)
        self.af2 = nn.ReLU()
        self.do2 = nn.Dropout(0.15)

        self.conv3 = nn.Conv3d(16, 32, kernel_size=(3,3,3), stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(32)
        self.af3 = nn.ReLU()
        self.do3 = nn.Dropout(0.15)

        self.conv4 = nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=1, padding=1)
        self.bn4 = nn.BatchNorm3d(64)
        self.af4 = nn.ReLU()
        self.do4 = nn.Dropout(0.15)

        self.conv5 = nn.Conv3d(64, 128, kernel_size=(3,3,3), stride=1, padding=1)
        self.bn5 = nn.BatchNorm3d(128)
        self.af5 = nn.ReLU()
        self.do5 = nn.Dropout(0.15)

        self.conv6 = nn.Conv3d(128, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1)
        #self.conv6 = nn.ConvTranspose3d(128, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1)
        #self.af6 = nn.ReLU()    # --> torglierla perchè tipicamente non si chiude con una ReLU

        #self.dropout = nn.Dropout(0.15)    #aggiunto dopo, da controllare se funziona   #metterlo dopo ognuno 

    def forward(self, x):
        x = self.bn1(self.af1(self.conv1(x)))
        x = self.do1(x)
        x = self.bn2(self.af2(self.conv2(x)))
        x = self.do2(x)
        x = self.bn3(self.af3(self.conv3(x)))
        x = self.do3(x)
        x = self.bn4(self.af4(self.conv4(x)))
        x = self.do4(x)
        x = self.bn5(self.af5(self.conv5(x)))
        x = self.do5(x)
        #x = self.af6(self.conv6(x))
        x = self.conv6(x)

        return x