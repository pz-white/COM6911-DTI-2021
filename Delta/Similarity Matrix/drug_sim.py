# import modules
import numpy as np
from tdc.multi_pred import DTI
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

# load in data
data_Kd = DTI(name = 'BindingDB_Kd')
data_Kd.convert_to_log(form = 'binding')
split = data_Kd.get_split(method = 'random', seed = 42, frac = [0.6, 0.05, 0.35])

train = split['train']
test = split['test']
print('Data loaded')

ID_to_Drug = dict(enumerate(list(dict.fromkeys(train['Drug']))))
Drug_to_ID = dict((v,k) for k,v in ID_to_Drug.items())
print('Drug dictionaries completed')

num_drugs = len(Drug_to_ID.keys())

drug_fingerprints={}

for i in range(num_drugs):
    drug = ID_to_Drug[i]
    m = Chem.MolFromSmiles(drug)
    fp = AllChem.GetMorganFingerprint(m, 2)
    drug_fingerprints[i] = fp
    

drug_sim = np.zeros((num_drugs, num_drugs))
for i in range(num_drugs):
    if i % 1000 == 0:
        print('\n1000 drug similarities calculated')
    fp1 = drug_fingerprints[i]
    for j in range(num_drugs):
        fp2 = drug_fingerprints[j]
        sim_score = DataStructs.DiceSimilarity(fp1, fp2)
        drug_sim[i][j] = sim_score

print('Drug similarity matrix completed')

np.savetxt('drug_sim.txt', drug_sim, delimiter=',')
print('Drug similarity matrix saved')
