from embedding import Embedding
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import time
import torch

class DBSCAN:
    def __init__(self, bindingdb_dataset, eps=3.5, min_samples=30, n_components=150, embedding='morgan-fp'):
        self.data = bindingdb_dataset
        self.embedding = embedding
        self.eps = eps
        self.min_samples = min_samples
        self.n_components = n_components

    def cluster(self):

        drug_ids_data = self.data.get_data()[['Drug_ID', 'Drug']]
        unique_drug_ids_data = drug_ids_data.drop_duplicates().reset_index(drop=True)

        drugid_index_mapping = {drugid: drug_ids_data.index[drug_ids_data['Drug_ID'] == drugid].tolist()
                                for drugid in unique_drug_ids_data.Drug_ID}

        drug_ids_data = Embedding(unique_drug_ids_data, 'fp').perform_embedding()

        st = time.time()
        pca = PCA(self.n_components)

        temp_data = list(drug_ids_data['Drug_vector'])
        data = torch.tensor(pca.fit_transform(temp_data, y=None))

        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        clustering.fit(data)
        et = time.time()
        print("Time required for clustering:", et-st)

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
