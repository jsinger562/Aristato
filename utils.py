from torch.utils.data import DataLoader, Dataset
import torch

class PhilosophyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]
        

    def __getitem__(self, index):
        return self.X[index], torch.Tensor([self.Y[index]])

        