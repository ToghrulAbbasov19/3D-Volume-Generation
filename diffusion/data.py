import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

class Dataset_3d(Dataset):
    def __init__(self, path_chair, path_plane, path_table):
        self.chair_data = np.load(path_chair)
        self.plane_data = np.load(path_plane)
        self.table_data = np.load(path_table)

        mn = min(len(self.chair_data), len(self.plane_data), len(self.table_data))
        print(mn)
        self.chair_data = self.chair_data[:mn]
        self.plane_data = self.plane_data[:mn]
        self.table_data = self.table_data[:mn]

        self.data = np.concatenate((self.chair_data, self.plane_data, self.table_data))
        self.labels = self.get_classes()
        # self.data_w_labels = self.get_data()


    def get_classes(self):
        self.chair_data_classes = np.array([1 for data in self.chair_data])
        self.plane_data_classes = np.array([2 for data in self.plane_data])
        self.table_data_classes = np.array([3 for data in self.table_data])

        return np.concatenate((self.chair_data_classes, self.plane_data_classes, self.table_data_classes))



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    


class Dataset_3d_single(Dataset):
    def __init__(self, path_chair, path_plane, path_table):
        self.data = np.load(path_chair)
        # self.plane_data = np.load(path_plane)
        # self.table_data = np.load(path_table)
        # self.data = np.concatenate((self.chair_data, self.plane_data, self.table_data))
        # self.labels = self.get_classes()
        # self.data_w_labels = self.get_data()


    # def get_classes(self):
    #     self.chair_data_classes = np.array([1 for data in self.chair_data])
    #     self.plane_data_classes = np.array([2 for data in self.plane_data])
    #     self.table_data_classes = np.array([3 for data in self.table_data])

    #     return np.concatenate((self.chair_data_classes, self.plane_data_classes, self.table_data_classes))



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

