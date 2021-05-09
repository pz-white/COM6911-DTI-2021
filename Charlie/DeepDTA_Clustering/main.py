import numpy as np
from tdc.multi_pred import DTI
import argparse
from config import get_cfg_defaults
from model import get_model
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.utils.data import DataLoader
from cluster.DBSCAN_clustering import DBSCAN
from cluster.Kmeans_clustering import Kmeans
from cluster.agglomerative_clustering import Agglomerative
from dta_datasets import DTADataset


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description='DeepDTA on BindingDB dataset')
    parser.add_argument('--cfg', required=True, help='path to config file', type=str)
    parser.add_argument('--gpus', default='0', help='gpu id(s) to use', type=str)
    parser.add_argument('--resume', default='', type=str)
    args = parser.parse_args()
    return args


def get_split_by_clusters(bindingdb_data, num_of_clusters, frac=[0.65, 0.05, 0.30]):

    clusters = np.arange(0, num_of_clusters)
    np.random.shuffle(clusters)
    train_clusters, test_clusters, val_clusters = np.split(clusters, [int(frac[0] * len(clusters)),
                                                                      int((frac[0] + frac[2]) * len(clusters))])

    train_dataset = bindingdb_data.loc[bindingdb_data['Cluster'].isin(train_clusters)]
    val_dataset = bindingdb_data.loc[bindingdb_data['Cluster'].isin(val_clusters)]
    test_dataset = bindingdb_data.loc[bindingdb_data['Cluster'].isin(test_clusters)]

    train_dataset = train_dataset.drop(['Cluster'], axis=1).reset_index()
    val_dataset = val_dataset.drop(['Cluster'], axis=1).reset_index()
    test_dataset = test_dataset.drop(['Cluster'], axis=1).reset_index()

    return train_dataset, val_dataset, test_dataset


def apply_clustering(bindingdb_dataset, num_of_clusters, cluster_type='k_means'):
    if cluster_type == 'agglomerative':
        bindingdb_dataset = Agglomerative(bindingdb_dataset, num_of_clusters, 'fp').cluster()
    elif cluster_type == 'dbscan':
        print("dbscan")
        #bindingdb_dataset = Dbscan(bindingdb_dataset, 'fp', num_of_clusters).cluster()
    else:
        #bindingdb_dataset = Kmeans(bindingdb_dataset, 'fp', num_of_clusters).cluster()
        print("kmeans")
    return bindingdb_dataset


def main():
    # For now hard coding the values for all methods below, change to read it using config file
    print("Start..")
    args = arg_parse()

    # ---- set configs, logger and device ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Fetch the bindingDB dataset based on the name defined in config file
    bindingdb_dataset = DTI(name=cfg.DATASET.NAME)

    # Apply cluster based on the 3 types of cluster
    # can define cluster_type -> k_means, agglomerative, dbscan
    bindingdb_dataset = apply_clustering(bindingdb_dataset, num_of_clusters=cfg.SOLVER.NUM_OF_CLUSTERS,
                                         cluster_type=cfg.MODEL.CLUSTER_TYPE)

    # Split the data based on the clusters formed by specifying the split in fraction
    train_dataset, val_dataset, test_dataset = get_split_by_clusters(bindingdb_dataset,
                                                                     num_of_clusters=cfg.SOLVER.NUM_OF_CLUSTERS)
    
    train_dataset = DTADataset(ds=train_dataset)
    val_dataset = DTADataset(ds=val_dataset)
    test_dataset = DTADataset(ds=test_dataset)

    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=cfg.SOLVER.TRAIN_BATCH_SIZE)
    val_loader = DataLoader(dataset=val_dataset, shuffle=True, batch_size=cfg.SOLVER.TEST_BATCH_SIZE)
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=cfg.SOLVER.TEST_BATCH_SIZE)

    # ---- set model ----
    model = get_model(cfg)

    # ---- training and evaluation ----
    gpus = 1 if device == "cuda" else 0
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode="min")
    trainer = pl.Trainer(max_epochs=cfg.SOLVER.MAX_EPOCHS, gpus=gpus, callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
    trainer.test(test_dataloaders=test_loader)

    print("Done!!!")


if __name__ == "__main__":
    main()
