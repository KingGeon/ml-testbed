import os
import sys
import requests
from typing import Any, Dict, Optional, Tuple, List
from urllib.error import URLError
sys.path.append("../../..")

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import pyrootutils
import learn2learn as l2l
from learn2learn.data.transforms import NWays,KShots,FusedNWaysKShots, LoadData, RemapLabels, ConsecutiveLabels

path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)

from src.data.datasets.new_cwru import CWRU

class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'data': self.data[idx],
            'target': self.targets[idx]
        }
        return sample

class CWRUDataModule(LightningDataModule):
    def __init__(self,
                 train_exps: List[float] =  ['12DriveEndFault'],
                 train_rpms: List[int] = ['1797','1772', '1750', '1730'], 
                 test_exps: List[float] =  ['12FanEndFault'],
                 test_rpms: List[int] = ['1797','1772', '1750', '1730'],
                 train_val_split: Tuple[float, float] = [0.6, 0.4],
                 ways: int = 2,
                 shots: int = 2,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 persistent_workers: bool = True):
                     
        super().__init__()
        self.save_hyperparameters(logger=False) # self.hparams activated
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        
    def prepare_data(self):
        self.train_data = CWRU(exps=self.hparams.train_exps,
                         rpms=self.hparams.train_rpms
                         )
        self.data_test = CWRU(exps=self.hparams.test_exps,
                         rpms=self.hparams.test_rpms
                         )
        

    def setup(self):
        if not self.data_train and not self.data_val:
            print(self.train_data.__len__())
            self.data_train, self.data_val = random_split(
                dataset=self.train_data,
                lengths=self.hparams.train_val_split,
                generator=torch.Generator().manual_seed(42)
            )
        self.data_train = l2l.data.MetaDataset(self.data_train)
        self.data_val  = l2l.data.MetaDataset(self.data_val)
        self.data_test = l2l.data.MetaDataset(self.data_test)
        self.data_train = l2l.data.MetaDataset(self.data_train)
        self.data_val = l2l.data.MetaDataset(self.data_val)
        self.data_test = l2l.data.MetaDataset(self.data_test)
    def make_tasks(self):
        train_transforms = [
        NWays(self.data_train, self.hparams.ways),
        KShots(self.data_train, 2*self.hparams.shots),
        LoadData(self.data_train),
        RemapLabels(self.data_train),
        ConsecutiveLabels(self.data_train),
    ]
        self.train_tasks = l2l.data.TaskDataset(self.data_train,
                                            task_transforms=train_transforms,
                                            num_tasks=600)

        valid_transforms = [
                NWays(self.data_val, self.hparams.ways),
                KShots(self.data_val, 2*self.hparams.shots),
                LoadData(self.data_val),
                ConsecutiveLabels(self.data_val),
                RemapLabels(self.data_val),
            ]
        self.valid_tasks = l2l.data.TaskDataset(self.data_val,
                                            task_transforms=valid_transforms,
                                            num_tasks=600)

        test_transforms = [
                NWays(self.data_test, self.hparams.ways),
                KShots(self.data_test, 2*self.hparams.shots),
                LoadData(self.data_test),
                RemapLabels(self.data_test),
                ConsecutiveLabels(self.data_test),
            ]
        self.test_tasks = l2l.data.TaskDataset(self.data_test,
                                            task_transforms=test_transforms,
                                            num_tasks=600)        
        return self.train_tasks, self.valid_tasks, self.test_tasks    
        
if __name__ == '__main__':
    _ = CWRUDataModule()

