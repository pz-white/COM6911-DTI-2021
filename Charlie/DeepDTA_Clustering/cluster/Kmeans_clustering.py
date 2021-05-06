from embedding import Embedding
from sklearn.cluster import KMeans
import torch


class Kmeans:

    def __init__(self, bindingdb_dataset, num_of_clusters, embedding='morgan-fp'):
        self.data = bindingdb_dataset
        self.embedding = embedding
        self.num_of_clusters = num_of_clusters

    def cluster(self):

        drug_ids_data = self.data[['Drug_ID', 'Drug']]
        unique_drug_ids_data = drug_ids_data.drop_duplicates().reset_index(drop=True)

        drugid_index_mapping = {drugid: self.data.index[self.data['Drug_ID'] == drugid].tolist()
                                for drugid in unique_drug_ids_data.Drug_ID}

        if 'fp' == self.embedding:
            data = Embedding.fp_embedding(self, unique_drug_ids_data)
        elif 'mol2vec' == self.embedding:
            data = Embedding.mol2vec_embedding(self, unique_drug_ids_data)
        else:
            data = Embedding.morgan_fp_embedding(self, unique_drug_ids_data)

        data = torch.tensor(list(data['Drug_vector']))
        clustering = KMeans(n_clusters=self.num_of_clusters)
        clustering.fit(data)

        clusters = clustering.labels_
        data['Cluster'] = '' # may not need this.
        for i, (k, v) in enumerate(drugid_index_mapping.items()):
            for ind in v:
                data['Cluster'][ind] = clusters[i]

        return data
