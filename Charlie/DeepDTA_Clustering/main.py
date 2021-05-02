from tdc.multi_pred import DTI
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
import math

data = DTI(name = 'BindingDB_Kd')
# data = DTI(name = 'DAVIS')
# data = DTI(name = 'KIBA')

def drugTarget2vec(data):
    data = data.get_data()
    
    data_selected = data[['Drug_ID','Drug','Target_ID','Target','Y']] 
    data_selected['Drug2vector'] = ''
    data_selected['TargetId'] = ''
    
    # use rdkit calculate ECFPs
    for ind, drug in enumerate(data_selected['Drug']):
        mol = Chem.MolFromSmiles(drug)
        Morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2) 
        #Explicitbitvects, which record whether or not a bit exists, are usually faster than SparseBitVects, but take up more memory, similar to fixed-length bit strings.
        intmap = map(int, Morgan_fp.ToBitString())
        data_selected['Drug2vector'][ind] = np.array(list(intmap))
    
    # create a dict to record the map relation of Target and TargetID
    ID_to_Target = dict(enumerate(list(dict.fromkeys(data_selected['Target']))))
    Target_to_ID = {value:key for key,value in ID_to_Target.items()} 
    
    # build the target2ID columns into dataframe 
    for ind, target in enumerate(data_selected['Target']):
        data_selected['TargetId'][ind] = np.array(Target_to_ID[target])   
    
    return data_selected

# the new df with Drug2Vector and Target2Id
data_transformed = drugTarget2vec(data)

# select the new columns
data_selected = data_transformed[['Drug2vector','Drug','Target2Id','Target','Y']]

from torch.utils.data import DataLoader
drug_vector = data_selected['Drug2vector']
train_loader = DataLoader(dataset=drug_vector, shuffle=False, batch_size=len(data_selected))
sample = next(iter(train_loader))
sample = np.array(sample) # get the array like Drug2Vector
print(sample)
print(sample.shape) # show the shape of Drug2vector
