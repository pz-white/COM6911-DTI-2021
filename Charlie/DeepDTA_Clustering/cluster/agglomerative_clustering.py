from embedding import Embedding
from sklearn.cluster import AgglomerativeClustering
import torch

class Agglomerative:

    def __init__(self, bindingDB_dataset, embedding='morgan-fp'):
        self.data = bindingDB_dataset
        self.embedding = embedding


    def cluster(self):

        drug_ids_data = self.data[['Drug_ID', 'Drug']]
        unique_drug_ids_data = drug_ids_data.drop_duplicates().reset_index(drop=True)

        drugid_index_mapping = {drugid: self.data.index[self.data['Drug_ID'] == drugid].tolist()
                                for drugid in unique_drug_ids_data.Drug_ID }


        if 'fp' == self.embedding:
            data = Embedding.fp_embedding(self, unique_drug_ids_data)
        elif 'mol2vec' == self.embedding:
            data = Embedding.mol2vec_embedding(self, unique_drug_ids_data)
        else:
            data = Embedding.morgan_fp_embedding(self, unique_drug_ids_data)

        data = torch.tensor(list(data['Drug_vector']))
        clustering = AgglomerativeClustering(linkage='average', n_clusters=50)
        clustering.fit(data)

        clusters = clustering.labels_
        data['Cluster'] = ''
        for i, (k, v) in enumerate(drugid_index_mapping.items()):
            for ind in v:
                data['Cluster'][ind] = clusters[i]

        return data