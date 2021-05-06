from tdc.multi_pred import DTI

import numpy as np
import random
import pandas as pd


def get_split_by_clusters(dataset, num_of_clusters, frac=[0.65, 0.05, 0.30]):

    clusters = np.arange(0, num_of_clusters)
    np.random.shuffle(clusters)
    train_clusters, test_clusters, val_clusters = np.split(clusters, [int(frac[0] * len(clusters)), int((frac[0]+frac[2]) * len(clusters))])
    
    train_dataset = bindingDB_data.loc[bindingDB_data['Cluster'].isin(train_clusters)]
    val_dataset = bindingDB_data.loc[bindingDB_data['Cluster'].isin(val_clusters)]
    test_dataset = bindingDB_data.loc[bindingDB_data['Cluster'].isin(test_clusters)]
    
    return train_dataset, val_dataset, test_dataset

def apply_clustering(cluster_type = 'k_means'):
  
  if cluster_type == 'k_means':
    bindingDB_dataset = 
  elif cluster_type == 'agglomerative':
    bindingDB_dataset = 
  else cluster_type == 'dbscan':
    bindingDB_dataset =
    
    return bindingDB_dataset

def main():
  
  # For now hard coding the values for all methods below, change to read it using config file  
  
  # Fetch the bindingDB dataset based on the name defined in config file
  bindingDB_dataset = DTI(name = 'BindingDB_Kd')
  
  # Apply clustering based on the 3 types of clustering
  # can define cluster_type -> k_means, agglomerative, dbscan
  bindingDB_dataset = apply_clustering(cluster_type = 'agglomerative')
  
  # Split the data based on the clusters formed by specifying the split in fraction
  train_dataset, val_dataset, test_dataset = get_split_by_clusters(bindingDB_dataset, num_of_clusters)
  
  
  
  
  
  
  
  
  
