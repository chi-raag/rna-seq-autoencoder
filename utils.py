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

def data_import_mtx(mtx, labels):
    
        data = sc.read_mtx(mtx)
        labels = pd.read_table(labels, header = None)
        data.obs['treatment'] = np.array(labels[0])
        sc.pp.filter_cells(data, min_genes = 200)
        sc.pp.filter_genes(data, min_cells = 3)
        sc.pp.normalize_total(data)
        sc.pp.scale(data)
        return data
    
    
def batchify_autoencoder(data, batch_size=16):
    
    batches= []
    for n in range(0, len(data),batch_size):
        if n+batch_size < len(data):
            batches.append(data[n:n+batch_size])
            
    if len(data)%batch_size > 0:
        batches.append(data[len(data)-(len(data)%batch_size):len(data)])
  
    return batches

def train_model(model, bot_size=4, epochs=40):
    m = model.train_test(bottleneck_size=bot_size,
                         batch_size=32,
                         n_epochs=epochs,
                         lr=.01)

    return m


def train_models(model, epochs=40, min, max):
    models = []
    model_num = 1
    for bot_size in range(min,max):
        m = model.train_test(bottleneck_size = bot_size,
                              batch_size = 32,
                              n_epochs = epochs,
                              lr = .01)
        models.append(m)
        print("finished model " + str(model_num) + " out of " + str(max-min))
        model_num += 1
   
    return models


def create_embeddings(model, ann):
    embeddings = model.encoder(torch.from_numpy(ann.X))
    em_df = pd.DataFrame()
    for col in range(embeddings.shape[1]):
        em_df["em_" + str(col+1)] = embeddings[:,col].detach().numpy()
        
    return em_df


def kmeans(embeddings, clusters, ann):
    fit = KMeans(n_clusters = clusters, random_state = 0).fit(embeddings)
    embeddings['k'] = fit.labels_
    embeddings['k'] = embeddings['k'].astype('category')
    embeddings['y'] = np.array(ann.obs['treatment'])
    return embeddings


def pca_nmi(comp, ann):
    pca = PCA(n_components = comp)
    fitting = pca.fit_transform(ann.X)
    principalDf = pd.DataFrame(data = fitting)
    kmeans = KMeans(n_clusters=10, random_state=0).fit(principalDf)
    principalDf['kmeans'] = kmeans.labels_
    principalDf['y'] = np.array(ann.obs['treatment'])
    nmi_score = nmi(principalDf.kmeans, principalDf.y)
    return nmi_score


def get_statistics(models, ann):
    stats = pd.DataFrame(columns=['comp', 'type', 'nmi'])
    for m in models:
        em = create_embeddings(m, ann)
        k_output = kmeans(em, 10, ann)
        n = nmi(k_output.k, k_output.y)
        row = pd.DataFrame([[em.shape[1]-2, "ae", n]], columns=['comp', 'type', 'nmi'])
        stats = stats.append(row, ignore_index=True)
        
    for comp in range(4,13):
        row = pd.DataFrame([[comp, "pca", pca_nmi(comp, ann)]], columns=['comp', 'type', 'nmi'])
        stats = stats.append(row, ignore_index=True)
        
    return stats