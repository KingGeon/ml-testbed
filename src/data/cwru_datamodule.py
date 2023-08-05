import os
import sys
import requests
from typing import Any, Dict, Optional, Tuple, List
from urllib.error import URLError
sys.path.append("../../..")

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
import pyrootutils
path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)

from src.data.datasets.cwru import CWRU


class CWRUDataModule(LightningDataModule):
    def __init__(self,
                 fault_diameter: List[float] = None,
                 rpm: List[int] = None, 
                 sensor_position: str = None,
                 timeseries_len: str = 8192,
                 number_of_samples: int = 1000,
                 data_dir: str = "data/CWRU",
                 train_val_test_split: Tuple[float, float, float] = [0.4, 0.3, 0.3],
                 batch_size: int = 1024,
                 num_workers: int = 16,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 download = True):
                     
        super().__init__()
        self.save_hyperparameters(logger=False) # self.hparams activated
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        
    def prepare_data(self):
        self.data = CWRU(fault_diameter=self.hparams.fault_diameter,
                         rpm=self.hparams.rpm, 
                         sensor_position=self.hparams.sensor_position,
                         timeseries_len=self.hparams.timeseries_len,
                         number_of_samples=self.hparams.number_of_samples,
                         root = self.hparams.data_dir, 
                         download=True)

    def setup(self, stage):
        if not self.data_train and not self.data_val and not self.data_test:
            print(self.data.__len__())
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=self.data,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42)
            )
            
    def train_dataloader(self):
        return DataLoader(dataset=self.data_train,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          persistent_workers=self.hparams.persistent_workers,
                          shuffle=True)
        
    def val_dataloader(self):
        return DataLoader(dataset=self.data_val,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          persistent_workers=self.hparams.persistent_workers,
                          shuffle=False)
        
    def test_dataloader(self):
        return DataLoader(dataset=self.data_test,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          persistent_workers=self.hparams.persistent_workers,
                          shuffle=False)
        
        
if __name__ == '__main__':
    _ = CWRUDataModule()