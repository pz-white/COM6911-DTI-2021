from tdc.multi_pred import DTI
from cluster.agglomerative_clustering import Agglomerative
import numpy as np



def get_split_by_clusters(dataset, num_of_clusters, frac=[0.65, 0.05, 0.30]):

    clusters = np.arange(0, num_of_clusters)
    np.random.shuffle(clusters)
    train_clusters, test_clusters, val_clusters = np.split(clusters, [int(frac[0] * len(clusters)), int((frac[0]+frac[2]) * len(clusters))])
    
    train_dataset = bindingdb_data.loc[bindingdb_data['Cluster'].isin(train_clusters)]
    val_dataset = bindingdb_data.loc[bindingdb_data['Cluster'].isin(val_clusters)]
    test_dataset = bindingdb_data.loc[bindingdb_data['Cluster'].isin(test_clusters)]
    
    return train_dataset, val_dataset, test_dataset

def apply_clustering(cluster_type = 'k_means', bindingdb_dataset):
  
  if cluster_type == 'k_means':
    #bindingdb_dataset =
  elif cluster_type == 'agglomerative':
    bindingdb_dataset = Agglomerative(bindingdb_dataset, 'fp').cluster
  else cluster_type == 'dbscan':
    #bindingdb_dataset =
    
    return bindingdb_dataset

def main():
  
  # For now hard coding the values for all methods below, change to read it using config file  
  
  # Fetch the bindingDB dataset based on the name defined in config file
  bindingdb_dataset = DTI(name = 'BindingDB_Kd')
  num_of_clusters = 50

  # Apply cluster based on the 3 types of cluster
  # can define cluster_type -> k_means, agglomerative, dbscan
  bindingdb_dataset = apply_clustering(cluster_type = 'agglomerative', bindingdb_dataset)
  
  # Split the data based on the clusters formed by specifying the split in fraction
  train_dataset, val_dataset, test_dataset = get_split_by_clusters(bindingdb_dataset, num_of_clusters)
  
  
  
  
  
  
  
  
  
