import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd()))+'/mol2vec')
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem.Draw import IPythonConsole
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
from rdkit.Chem import AllChem
import tensorflow as tf
import torch



class Embedding:
  
 
  def fp_embedding(self, bindingDB_dataset):
    
    for ind, drug in enumerate(bindingDB_dataset['Drug']):
      mol = Chem.MolFromSmiles(drug)
      fp = Chem.RDKFingerprint(mol)
      intmap = map(int, fp.ToBitString())
      bindingDB_dataset['Drug_vector'][ind] = np.array(list(intmap))
      return bindingDB_dataset

                                                       
  def morgan_fp_embedding(self, bindingDB_dataset): 
    
    for ind, drug in enumerate(bindingDB_dataset['Drug']):
      mol = Chem.MolFromSmiles(drug)
      morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2) 
      intmap = map(int, morgan_fp.ToBitString())
      bindingDB_dataset['Drug_vector'][ind] = np.array(list(intmap))))
      return bindingDB_dataset
    
                            
  def sentences2vec_updated(sentences, model, unseen=None):
    
    #Change made in this line as vocab is removed in gensim 4.0
    #keys = set(model.wv.vocab.keys())
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
                            
  
  def mol2vec_embedding(self, bindingDB_dataset):
    
    #Load the pre-trained Mol2vec model 
    model = word2vec.Word2Vec.load('model_300dim.pkl')

    for ind, drug in enumerate(bindingDB_dataset['Drug']):
      mol = Chem.MolFromSmiles(drug)
      sent2vec = sentences2vecUpdated(mol2alt_sentence(mol, 2), model, unseen='UNK')
      bindingDB_dataset['Drug_vector'][ind] = DfVec(sent2vec)
      return bindingDB_dataset
    
  
  def perform_embedding(self, bindingDB_dataset, embedding_type):
    
    bindingDB_dataset['Drug_vector'] = ''
 
    if embedding_type == 'fp':
      bindingDB_dataset = self.fp_embedding(bindingDB_dataset)
    elif embedding_type == 'morgan_fp':
      bindingDB_dataset = self.morgan_fp_embedding(bindingDB_dataset)
    else embedding_type == 'mol2vec':
      bindingDB_dataset = self.mol2vec_embedding(bindingDB_dataset)
      
    return bindingDB_dataset
    
