
import linecache
from torch.utils.data import Dataset
import numpy as np
    
class FileDataset(Dataset):
    def __init__(self, filename):
        super(FileDataset, self).__init__()
        self._filename = filename
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())
    
    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        line = line.strip().split("\t")
        xxx = line[0]
        batch = {
            "xxx": np.asarray(xxx),
        }
        return batch
    
    def __len__(self):
        return self._total_data