import numpy as np
import pandas as pd
import scanpy as sc
import h5py
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.preprocessing import *
import random
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import scipy.sparse
import scipy.io
from scipy import * 
from sklearn.decomposition import PCA
from utils import *

class RNA_seq_ae:
    
    def __init__(self, file, labels):
        self.ann = data_import_mtx(file, labels)
        self.data = self.ann.X
        
    def processed_data(self):
        return self.ann
        
        
    class Autoencoder(nn.Module):
        
        def __init__(self, data_length, hidden_dimension, bottleneck_dimension):
    
            super(RNA_seq_ae.Autoencoder, self).__init__()

            self.encoder = nn.Sequential(
            nn.Linear(data_length, hidden_dimension),
            nn.Softplus(),
            nn.Linear(hidden_dimension,bottleneck_dimension))
            
            self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dimension,hidden_dimension),
            nn.Softplus(),
            nn.Linear(hidden_dimension,data_length),
            nn.Tanhshrink())
            
            
        def forward(self, batch):
            batch = self.encoder(batch)
            batch = self.decoder(batch)
            return batch
        
        
    def train_test(self, bottleneck_size, batch_size, n_epochs, lr):
        
            hidden_dimension = int(self.ann.shape[1]/2)
            
            batches = batchify_autoencoder(self.data, batch_size = batch_size)
            neural_network = RNA_seq_ae.Autoencoder(len(np.ravel(self.data[0])), hidden_dimension, bottleneck_size)
            optimizer = optim.Adagrad(neural_network.parameters(), lr=0.01)
            loss_function = nn.MSELoss()
            neural_network.train()
        
            for i in range(n_epochs):
                error = 0
                for ii in range(len(batches)):
                    optimizer.zero_grad()
                    batch = batches[ii]
                    predictions = neural_network(torch.tensor(np.asarray(batch).astype(np.float32)))
                    loss = loss_function(predictions,torch.tensor(np.asarray(batch).astype(np.float32)))       
                    loss.backward()      
                    optimizer.step()          
                    error += loss.data 
                    

            return neural_network