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
path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)
from src.data.datasets.aihub_motor_vibraion_proto import Motor_Vibration
from src.utils.meta_utils import FewShotBatchSampler_ProtoNet
class Motor_Vibration_Meta_DataModule(LightningDataModule):
    def __init__(self,
                test_motor_power: List[str] = ["7.5kW","22kW","30kW"],
                val_motor_power: List[str] = ["2.2kW"],
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
                N_WAY = 4,
                K_SHOT = 4,
                num_workers: int = 16,
                pin_memory: bool = True,
                persistent_workers: bool = True):
                     
        super().__init__()
        self.save_hyperparameters(logger=False) # self.hparams activated
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.N_WAY = N_WAY
        self.K_SHOT = K_SHOT
        

    def setup(self,stage):
        if not self.data_train and not self.data_val:
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
        return DataLoader(dataset=self.data_train,
                          batch_sampler=FewShotBatchSampler_ProtoNet(self.data_train.get_targets(), include_query=True, 
                                                                     N_way=self.N_WAY, K_shot=self.K_SHOT, shuffle=True, shuffle_once=True),
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          persistent_workers=self.hparams.persistent_workers)
        
    def val_dataloader(self):
        return DataLoader(dataset=self.data_val,
                          batch_sampler=FewShotBatchSampler_ProtoNet(self.data_val.get_targets(), include_query=True, 
                                                                     N_way=self.N_WAY, K_shot=self.K_SHOT, shuffle=False, shuffle_once=True),
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          persistent_workers=self.hparams.persistent_workers)
    
    def test_dataloader(self):
        return DataLoader(dataset=self.data_test,
                          batch_sampler=FewShotBatchSampler_ProtoNet(self.data_test.get_targets(), include_query=True, 
                                                                     N_way=self.N_WAY, K_shot=self.K_SHOT, shuffle=False, shuffle_once=True),
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          persistent_workers=self.hparams.persistent_workers)
    
     
if __name__ == '__main__':
    _ = Motor_Vibration_Meta_DataModule()

