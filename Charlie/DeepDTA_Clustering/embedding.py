import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())) + '/mol2vec')
from rdkit import Chem
from rdkit.Chem import AllChem
from mol2vec.features import mol2alt_sentence, DfVec
from gensim.models import word2vec


class Embedding:

    def __init__(self, bindingdb_dataset, embedding_type='morgan-fp'):
        self.data = bindingdb_dataset
        self.embedding_type = embedding_type

    def fp_embedding(self):

        for ind, drug in enumerate(self.data['Drug']):
            mol = Chem.MolFromSmiles(drug)
            fp = Chem.RDKFingerprint(mol)
            fp_list = list(map(int, fp.ToBitString()))
            self.data['Drug_vector'][ind] = np.array(fp_list)
        return self.data

    def morgan_fp_embedding(self):

        for ind, drug in enumerate(self.data['Drug']):
            mol = Chem.MolFromSmiles(drug)
            morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2)
            morgan_fp_list = list(map(int, morgan_fp.ToBitString()))
            self.data['Drug_vector'][ind] = np.array(morgan_fp_list)
        return self.data

    def sentences2vec_updated(self, sentences, model, unseen=None):

        # Change made in this line as vocab is removed in gensim 4.0
        # keys = set(model.wv.vocab.keys())
        keys = set(model.wv.key_to_index.keys())

        vec = []
        if unseen:
            unseen_vec = model.wv.word_vec(unseen)

        for sentence in sentences:
            if unseen:
                vec.append(sum([model.wv.word_vec(y) if y in set(sentence) & keys
                                else unseen_vec for y in sentence]))
            else:
                vec.append(sum([model.wv.word_vec(y) for y in sentence
                                if y in set(sentence) & keys]))
        return np.array(vec)

    def mol2vec_embedding(self):

        # Load the pre-trained Mol2vec model
        model = word2vec.Word2Vec.load('model_300dim.pkl')

        for ind, drug in enumerate(self.data['Drug']):
            mol = Chem.MolFromSmiles(drug)
            sent2vec = self.sentences2vec_updated(mol2alt_sentence(mol, 2), model, unseen='UNK')
            self.data['Drug_vector'][ind] = DfVec(sent2vec)
        return self.data

    def perform_embedding(self):

        self.data['Drug_vector'] = ''

        if self.embedding_type == 'fp':
            bindingdb_dataset = self.fp_embedding()
        elif self.embedding_type == 'mol2vec':
            bindingdb_dataset = self.mol2vec_embedding()
        else:
            bindingdb_dataset = self.morgan_fp_embedding()

        return bindingdb_dataset
