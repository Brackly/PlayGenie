from typing import Literal

import numpy
import torch
import pyarrow.dataset as ds
from logging import getLogger
logger = getLogger(__name__)
import pandas as pd
import numpy as np

class DataSet(torch.utils.data.Dataset):
    def __init__(self, folder_path, data_ratio=1):
        self.dataset:ds.dataset = ds.dataset(folder_path, format="parquet")
        self.data_ratio = data_ratio

    def __len__(self)->int:
        return int(len(self.dataset.files)*self.data_ratio)

    def __getitem__(self, idx):
        tensors = torch.tensor(numpy.array([x.tolist() for x in pd.read_parquet(self.dataset.files[idx])['embeddings'].values])) #.squeeze(dim=0)
        return tensors[:,:10,:],tensors[:,1:11,:]


def get_dataloader(mode:Literal['train', 'val', 'test']='train',data_ratio=1):

    dataset_path :str = f'./data/{mode}'
    return torch.utils.data.DataLoader(
        dataset=DataSet(folder_path=dataset_path,data_ratio=data_ratio),
        batch_size=1, # this has to be low because of the way we are reading and loading the files
        shuffle=mode=='train',
    )


