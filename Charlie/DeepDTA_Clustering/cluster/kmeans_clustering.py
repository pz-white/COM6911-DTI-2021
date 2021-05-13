from embedding import Embedding
from sklearn.cluster import KMeans
import torch
import time


class kmeans:

    def __init__(self, bindingdb_dataset, num_of_clusters, embedding='morgan-fp'):
        self.data = bindingdb_dataset
        self.embedding = embedding
        self.num_of_clusters = num_of_clusters

    def cluster(self):
        drug_ids_data = self.data.get_data()[['Drug_ID', 'Drug']]
        #         unique_drug_ids_data = drug_ids_data.drop_duplicates().reset_index(drop=True)

        drugid_index_mapping = {drugid: drug_ids_data.index[drug_ids_data['Drug_ID'] == drugid].tolist()
                                for drugid in drug_ids_data.Drug_ID}

        drug_ids_data = Embedding(drug_ids_data, 'fp').perform_embedding()

        st = time.time()
        data = torch.tensor(list(drug_ids_data['Drug_vector']))

        clustering = KMeans(n_clusters=self.num_of_clusters)
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
