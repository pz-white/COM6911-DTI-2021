from embedding import Embedding
from sklearn.cluster import SpectralClustering
import torch
import time
import numpy as np
import sklearn as sk


class Spectral:

    def __init__(self, bindingdb_dataset, num_of_clusters, embedding='morgan-fp'):
        self.data = bindingdb_dataset
        self.embedding = embedding
        self.num_of_clusters = num_of_clusters

    def cluster(self):

        drug_ids_data = self.data.get_data()[['Drug_ID', 'Drug']]
        unique_drug_ids_data = drug_ids_data.drop_duplicates().reset_index(drop=True)

        drugid_index_mapping = {drugid: drug_ids_data.index[drug_ids_data['Drug_ID'] == drugid].tolist()
                                for drugid in unique_drug_ids_data.Drug_ID}

        drug_ids_data = Embedding(unique_drug_ids_data, 'fp').perform_embedding()

        st = time.time()
        data = torch.tensor(list(drug_ids_data['Drug_vector']))
        
        # calculates a distance matrix, according to type METRIC euclidean
        data = sk.metrics.pairwise.pairwise_distances(data, metric='euclidean')
        # converts the distance matrix in above step to a similarity matrix
        beta = 1.0
        data = np.exp(-beta * data / data.std())

        clustering = SpectralClustering(n_clusters=self.num_of_clusters, n_init=1, n_neighbors=5, eigen_tol=0.000001, n_jobs=6,
                                         affinity='precomputed')
              
        print("Start fitting```")
        clustering.fit(data)
        et = time.time()
        print("Time required for clustering:", et - st)

        # Map the drug ids to cluster and save it in a dict
        drug_ids_cluster_map = dict(zip(drug_ids_data.Drug_ID, clustering.labels_))

        # Fetch entire data and add cluster column
        data = self.data.get_data()
        data['Cluster'] = ''

        # map the clusters to entire data
        for i, (k, v) in enumerate(drugid_index_mapping.items()):
            cluster_val = drug_ids_cluster_map.get(k)
            for ind in v:
                data['Cluster'][ind] = cluster_val

        return data
