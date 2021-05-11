from embedding import Embedding
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
import torch
import time


class Spectral:

    def __init__(self, bindingdb_dataset, num_of_clusters, n_components=150, embedding='morgan-fp'):
        self.data = bindingdb_dataset
        self.embedding = embedding
        self.num_of_clusters = num_of_clusters
        self.n_components = n_components

    def cluster(self):

        drug_ids_data = self.data.get_data()[['Drug_ID', 'Drug']]
        drug_ids_data = Embedding(drug_ids_data, self.embedding).perform_embedding()

        st = time.time()
        pca = PCA(self.n_components)

        temp_data = list(drug_ids_data['Drug_vector'])
        data = torch.tensor(pca.fit_transform(temp_data, y=None))

        clustering = SpectralClustering(n_clusters=self.num_of_clusters)
        clustering.fit(data)
        et = time.time()
        print("Time required for clustering:", et - st)

        # Fetch entire data and add cluster column
        data = self.data.get_data()
        data['Cluster'] = clustering.labels_

        return data