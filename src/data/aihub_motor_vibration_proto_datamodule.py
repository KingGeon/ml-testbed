import sys
from torch.utils.data import Dataset, random_split
from scipy.signal import spectrogram, butter, sosfilt
from scipy.stats import kurtosis
from typing import Any, Dict, Optional, Tuple, List
from urllib.error import URLError
sys.path.append("../../..")
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import pyrootutils
import gc
path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)
from src.data.datasets.aihub_motor_vibraion_proto import Motor_Vibration
from src.utils.meta_utils import FewShotBatchSampler_ProtoNet
import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels

class Motor_Vibration_Meta_DataModule(LightningDataModule):
    def __init__(self,
                test_motor_power: List[str] = ["5.5kW"],
                val_motor_power: List[str] = [],
                sampling_frequency_before_upsample: str = 4000,
                sampling_frequency_after_upsample: str = 8192,
                fault_type_dict = {"정상": 0,
                    "베어링불량": 1,
                    "벨트느슨함": 2,
                    "축정렬불량": 3,
                    "회전체불평형": 4},
                upsample_method = "soxr_vhq", #["soxr_vhq", "soxr_hq","kaiser_fast","kaiser_best","sinc_best","sinc_fastest"]
                train: bool = True,
                csv_num_to_use: int = 100,
                data_dir: str = "/home/mongoose01/mongooseai/data/cms/open_source/AI_hub/기계시설물 고장 예지 센서/Training/vibration",
                N_WAY = 4,
                K_SHOT = 4,
                num_workers: int = 4,
                pin_memory: bool = False,
                persistent_workers: bool = False):
                     
        super().__init__()
        self.save_hyperparameters(logger=False) # self.hparams activated
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.N_WAY = N_WAY
        self.K_SHOT = K_SHOT
        

    def setup(self,stage):
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train, self.data_val, self.data_test = Motor_Vibration(test_motor_power = self.hparams.test_motor_power, 
                                                             val_motor_power = self.hparams.val_motor_power,
                        sampling_frequency_before_upsample = self.hparams.sampling_frequency_before_upsample, 
                        sampling_frequency_after_upsample = self.hparams.sampling_frequency_after_upsample, 
                        fault_type_dict = self.hparams.fault_type_dict,
                        upsample_method = self.hparams.upsample_method,
                        train = self.hparams.train,
                        csv_num_to_use=self.hparams.csv_num_to_use,
                        root = self.hparams.data_dir).load_data()
            
    def train_dataloader(self):
        self.data_train = l2l.data.MetaDataset(self.data_train)
        train_transforms = [
            NWays(self.data_train, self.N_WAY),
            KShots(self.data_train, self.K_SHOT * 2),
            LoadData(self.data_train),
            RemapLabels(self.data_train),
        ]
        train_tasks = l2l.data.Taskset(
            self.data_train,
            task_transforms=train_transforms,
            num_tasks= 50,
        )
        return DataLoader(dataset = train_tasks,
                          num_workers = self.hparams.num_workers,
                          pin_memory = self.hparams.pin_memory,
                          persistent_workers=self.hparams.persistent_workers,shuffle = True)
        
    def val_dataloader(self):
        self.data_val = l2l.data.MetaDataset(self.data_val)
        valid_transforms = [
            NWays(self.data_val, self.N_WAY),
            KShots(self.data_val, self.K_SHOT * 2),
            LoadData(self.data_val),
            RemapLabels(self.data_val),
        ]
        valid_tasks = l2l.data.Taskset(
            self.data_val,
            task_transforms=valid_transforms,
            num_tasks=50,
        )
        return DataLoader(dataset = valid_tasks,
                          num_workers = self.hparams.num_workers,
                          pin_memory = self.hparams.pin_memory,
                          persistent_workers = self.hparams.persistent_workers)
    
    def test_dataloader(self):
        self.data_test = l2l.data.MetaDataset(self.data_test)
        test_transforms = [
            NWays(self.data_test, self.N_WAY),
            KShots(self.data_test, self.K_SHOT * 2),
            LoadData(self.data_test),
            RemapLabels(self.data_test),
        ]
        test_tasks = l2l.data.Taskset(
            self.data_test,
            task_transforms = test_transforms,
            num_tasks=50,
        )
        return DataLoader(dataset = test_tasks,
                          num_workers = self.hparams.num_workers,
                          pin_memory = self.hparams.pin_memory,
                          persistent_workers = self.hparams.persistent_workers)
    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        # Explicitly delete the datasets
        self.data_train = None
        self.data_val = None
        self.data_test = None

        # Call the garbage collector
        gc.collect()
    
     
if __name__ == '__main__':
    _ = Motor_Vibration_Meta_DataModule()

