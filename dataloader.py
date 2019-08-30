import lmdb
import pickle

from torch.utils.data import Dataset
from numpy import genfromtxt

class FloorLoader(Dataset):
    def __init__(self, data_folder, data_file, 
                 augmentations=None, lmdb_folder='cubi_lmdb/'):
        self.augmentations = augmentations

        self.lmdb = lmdb.open(data_folder+lmdb_folder, readonly=True,
                                max_readers=8, lock=False,
                                readahead=True, meminit=False)

        self.data_folder = data_folder
        
        # Load txt file to list
        self.folders = genfromtxt(data_folder + data_file, dtype='str', delimiter="\n")

    def __len__(self):
        """__len__"""
        return len(self.folders)
    
    def __getitem__(self, index):
        sample = self.get_lmdb(index)
        
        if self.augmentations is not None:
            sample = self.augmentations(sample)
        
        return sample
    
    def get_lmdb(self, index):
        key = self.folders[index].encode('utf-8')
        with self.lmdb.begin(write=False) as f:
            data = f.get(key)
        sample = pickle.loads(data)
        return sample
