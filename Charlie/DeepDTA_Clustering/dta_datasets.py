import torch
from torch.utils import data
from tdc import utils
#from tdc.base_dataset import DataLoader
from kale.prepdata.chem_transform import integer_label_protein, integer_label_smiles


class DTADataset(data.Dataset):

    def __init__(
        self,
        ds,
        mode="cnn_cnn",
        y_log=True,
        drug_transform=None,
        protein_transform=None,
    ):
        """

        :type dataset: object
        """
        self.data = ds
        self.mode = mode.lower()
        if y_log:
            utils.convert_to_log(self.data.Y)
        self.drug_transform = drug_transform
        self.protein_transform = protein_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        drug, protein, label = self.data["Drug"][idx], self.data["Target"][idx], self.data["Y"][idx]
        mode_drug, mode_protein = self.mode.split("_")
        if mode_drug == "cnn":
            drug = torch.LongTensor(integer_label_smiles(drug))
        if mode_protein == "cnn":
            protein = torch.LongTensor(integer_label_protein(protein))
        label = torch.Tensor([label])
        if self.drug_transform is not None:
            self.drug_transform(drug)
        if self.protein_transform is not None:
            self.protein_transform(protein)
        return drug, protein, label
