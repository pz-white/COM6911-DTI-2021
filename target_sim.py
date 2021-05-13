# import modules
from pyspark.sql import SparkSession
import numpy as np
from tdc.multi_pred import DTI
from Bio import Align

spark = SparkSession.builder \
    .master("local[30]") \
    .appName("targets") \
    .config("spark.local.dir", "/fastdata/acp20swm") \
    .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

data_Kd = DTI(name='BindingDB_Kd')
data_Kd.convert_to_log(form='binding')
print('Data loaded and converted to log form')

split = data_Kd.get_split(seed=42,  frac=[0.6, 0.05, 0.35])
train = split['train']
print('Data split into training and test')

ID_to_Target = dict(enumerate(list(dict.fromkeys(train['Target']))))
Target_to_ID = dict((v, k) for k, v in ID_to_Target.items())
print('Target dictionaries created')

targets = list(Target_to_ID.keys())

aligner = Align.PairwiseAligner()
aligner.mode = 'local'
print('Algorithm is ', aligner.algorithm)

target_sim = np.zeros((len(targets), len(targets)))
for i in range(len(targets)):
    protein1 = targets[i]
    score1 = aligner.score(protein1, protein1)

    if i % 100 == 0:
        print('\n100 target similarities calculated')

    for j in range(len(targets)):
        protein2 = targets[j]
        score = aligner.score(protein1, protein2)
        score2 = aligner.score(protein2, protein2)

        norm_score = score / (np.sqrt(score1) * np.sqrt(score2))
        target_sim[i][j] = round(norm_score, 4)

print('Target similarity matrix created')

np.savetxt('target_sim_Kd.txt', target_sim, delimiter=',')

print('Target matrix saved')
