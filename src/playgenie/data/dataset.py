import torch
from safetensors import safe_open
from typing import Literal, Union, Tuple

from logging import getLogger
logger = getLogger(__name__)

class DataSet(torch.utils.data.Dataset):
    def __init__(self, path,mode:Literal['train','validation','test']='train')->None:
        self.data:Union[torch.Tensor,None] = None
        with safe_open(path, framework="pt") as f:
            self.data = f.get_tensor(mode)

    def __len__(self)->int:
        return self.data.shape[0]

    def __getitem__(self,idx:int)->Tuple[torch.Tensor,torch.Tensor]:
        return self.data[idx,0:10,:],self.data[idx,1:11,:]


def get_dataloader(path:str,mode:Literal['train','validation','test'],batch_size:int)->torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(dataset=DataSet(path=path, mode=mode),shuffle=False,batch_size=batch_size)





