import os
import sys
import requests
from typing import Any, Dict, Optional, Tuple, List
from urllib.error import URLError
sys.path.append("../../..")

import torch
from torch.utils.data import Dataset
import pyrootutils
import learn2learn as l2l
from learn2learn.data.transforms import NWays,KShots,FusedNWaysKShots, LoadData, RemapLabels, ConsecutiveLabels
from src.data.datasets.aihub_motor_vibraion import Motor_Vibration
path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)

class Motor_vibration_TaskModule():
    def __init__(self,
                 test_motor_power: List[float] = ["7.5kW","30kW"],
                sampling_frequency_before_upsample: str = 4000,
                sampling_frequency_after_upsample: str = 8192,
                fault_type_dict = {"정상": 0,
                    "베어링불량": 1,
                    "벨트느슨함": 2,
                    "축정렬불량": 3,
                    "회전체불평형": 4},
                upsample_method = "soxr_vhq", #["soxr_vhq", "soxr_hq","kaiser_fast","kaiser_best","sinc_best","sinc_fastest"]
                train: bool = True,
                csv_num_to_use: int = 500,
                data_dir: str = "/home/mongoose01/mongooseai/data/cms/open_source/AI_hub/기계시설물 고장 예지 센서/Training/vibration",
                train_val_split: Tuple[float, float, float] = [0.7, 0.3],
                ways: int = 4,
                shots: int = 4,
                pin_memory: bool = True,
                persistent_workers: bool = True,
                batch_size: int = 512):
                     
        super().__init__()
        self.save_hyperparameters(logger=False) # self.hparams activated
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        
    def setup(self):
        if not self.data_train and not self.data_val:
            self.data_train, self.data_val, self.data_test =  Motor_Vibration(test_motor_power = self.hparams.test_motor_power,
                         sampling_frequency_before_upsample = self.hparams.sampling_frequency_before_upsample, 
                         sampling_frequency_after_upsample = self.hparams.sampling_frequency_after_upsample, 
                         fault_type_dict = self.hparams.fault_type_dict,
                         upsample_method = self.hparams.upsample_method,
                         train = self.hparams.train,
                         csv_num_to_use=self.hparams.csv_num_to_use,
                         train_val_test_split = self.hparams.train_val_split,
                         root = self.hparams.data_dir).load_data()
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
                                            num_tasks=2000)

        valid_transforms = [
                NWays(self.data_val, self.hparams.ways),
                KShots(self.data_val, 2*self.hparams.shots),
                LoadData(self.data_val),
                ConsecutiveLabels(self.data_val),
                RemapLabels(self.data_val),
            ]
        self.valid_tasks = l2l.data.TaskDataset(self.data_val,
                                            task_transforms=valid_transforms,
                                            num_tasks=500)

        test_transforms = [
                NWays(self.data_test, self.hparams.ways),
                KShots(self.data_test, 2*self.hparams.shots),
                LoadData(self.data_test),
                RemapLabels(self.data_test),
                ConsecutiveLabels(self.data_test),
            ]
        self.test_tasks = l2l.data.TaskDataset(self.data_test,
                                            task_transforms=test_transforms,
                                            num_tasks=500)        
        return self.train_tasks, self.valid_tasks, self.test_tasks    
        
if __name__ == '__main__':
    _ = Motor_vibration_TaskModule()
