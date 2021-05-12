from embedding import Embedding
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import time
import torch

class dbscan:
    def __init__(self, bindingdb_dataset, eps=3.5, min_samples=30, n_components=150, embedding='morgan-fp'):
        self.data = bindingdb_dataset
        self.embedding = embedding
        self.eps = eps
        self.min_samples = min_samples
        self.n_components = n_components

    def cluster(self):

        drug_ids_data = self.data.get_data()[['Drug_ID', 'Drug']]
        drug_ids_data = Embedding(drug_ids_data, self.embedding).perform_embedding()

        st = time.time()
        pca = PCA(self.n_components)

        temp_data = list(drug_ids_data['Drug_vector'])
        data = torch.tensor(pca.fit_transform(temp_data, y=None))

        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        clustering.fit(data)
        et = time.time()
        print("Time required for clustering:", et-st)

        # Fetch entire data and add cluster column
        data = self.data.get_data()
        data['Cluster'] = clustering.labels_

        return data
