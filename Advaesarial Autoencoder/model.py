import pandas as pd 
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, inp, out, hidden_states=1000):
        super(Encoder, self).__init__()
        
        self.fc1 = nn.Linear(inp,hidden_states)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features)
        self.fc3 = nn.Linear(self.fc2.out_features, out)
        
    def forward(self, x):
        x = F.dropout(self.fc1(x),p=0.25,training=self.training)
        x = F.relu(x)
        x = F.dropout(self.fc2(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = self.fc3(x)
                
        return x


class Decoder(nn.Module):
    def __init__(self, inp, out, hidden_states=1000):
        super(Decoder,self).__init__()
        
        self.fc1 = nn.Linear(inp, hidden_states)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features)
        self.fc3 = nn.Linear(self.fc2.out_features, out)
        
    def forward(self, z):
        z = F.dropout(self.fc1(z), p=0.25, training=self.training)
        z = F.relu(z)
        z = F.dropout(self.fc2(z), p=0.25, training=self.training)
        z = torch.sigmoid(self.fc3(z))
        
        return z        


class Discriminator(nn.Module):
    def __init__(self, inp, hidden_states=500):
        super(Discriminator, self).__init__()
        
        self.fc1 = nn.Linear(inp, hidden_states)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features)
        self.fc3 = nn.Linear(self.fc2.out_features, 1)
        
    def forward(self, z):
        z = F.dropout(self.fc1(z), p=0.2, training=self.training)
        z = F.relu(z)
        z = F.dropout(self.fc2(z), p=0.2, training=self.training)
        z = F.relu(z)
        acc_prob = torch.sigmoid(self.fc3(z))
        
        return acc_prob


# Weight initialization with gaussian distribution of std 0.01

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)