from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd

class ToyData(Dataset):
    """custom dataset class
    """
    def __init__(self, df: pd.DataFrame):
        self.X = torch.from_numpy(np.array(df[['Feature_0', 'Feature_1']])).type(torch.float)
        self.y = torch.from_numpy(np.array(df[['target']])).type(torch.float)

    def __getitem__(self, index: int):
        """return item i

        :param int index: element i
        :return _type_: return Features and target for eleemtn i
        """
        return self.X[index], self.y[index] 

    def __len__(self):
        """return the number of samples

        :return _type_: _description_
        """
        return len(self.X)
    
    def __repr__(self):
        """pretty print

        :return _type_: _description_
        """
        return f"X shape: {self.X.shape}, y shape: {self.y.shape}, X type: {type(self.X)}, y type: {type(self.y)}"