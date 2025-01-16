from torch.utils.data import Dataset
import numpy as np
import os

class VoxelDataset(Dataset):

    def __init__(self, files, transform=None):
        
        # files = ['table_voxels_train.npy, 'airplane_voxels_train.npy']

        self.BASE_DATA_DIR = '/data/hdf5_data/'
        self.files = files
        self.transform = transform
        self.data = self.process(self.files)

    
    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        return self.data[idx]
    
    
    def process(self, files):
        """
            Takes the names of npy files that stores our data and combine
            them in a data array in shuffled version

            input : files -> names of npy files for each category
            output : data -> data combined in an array and shuffled
        """
        arrays = []

        for file in files:
            print('here')
            file_path = os.path.join(self.BASE_DATA_DIR, file)
            # category_file = np.load(file_path)
            # print(category_file.shape)
            file_name = file[:-17]
            print(file_name)

            arrays.append(np.load(file_path, mmap_mode='r'))
            # Concatenate along the specified axis

        combined_array = np.concatenate(arrays, axis=0)
        print(combined_array.shape)
        # data = np.array(data)
        np.random.shuffle(combined_array)
        return combined_array




