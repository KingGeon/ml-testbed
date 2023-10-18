import os
from lightning import LightningDataModule
from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy.io import loadmat
from sklearn.utils import shuffle
from typing import List
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import torch
import pandas as pd


DATAFOLDER_ROOT = "/home/mongoose01/mongooseai/data/cms/open_source/phm-ieee-2012-data-challenge-dataset/Learning_set"


class Train_IMS(Dataset):
    def __init__(
        self,
        data_root: int = DATAFOLDER_ROOT,
        train_val_bearing_list  =  ["Bearing1_1","Bearing1_2", "Bearing1_5", "Bearing1_6","Bearing1_7",
                      "Bearing2_1","Bearing2_2", "Bearing2_3","Bearing2_4","Bearing2_5","Bearing2_6","Bearing2_7",
                      "Bearing3_1","Bearing3_3",]
    ):
        super().__init__()
        self.root = data_root
        self.train_val_bearing_list = train_val_bearing_list
        
        numpy_df_list = []
        numpy_target_list = []
        temp_x_list = []
        temp_y_list = []
        li = []
        for bearing in self.train_val_bearing_list:
            li = []
            csv_list = sorted(os.listdir(os.path.join(data_root, bearing)))
            print(f"{bearing}, length: {len(csv_list)}, RUL: {len(csv_list) -1 }")
            for i, csv in enumerate(csv_list):  # enumerate를 사용하여 인덱스와 함께 반복
                if bearing == "Bearing1_4" : 
                    df = pd.read_csv(os.path.join(data_root, bearing, csv), header=None,sep=';')
                else : df = pd.read_csv(os.path.join(data_root, bearing, csv), header=None)
                df["time_after_start"] = i
                numpy_df = np.array(df.iloc[:, -3:]).reshape(2560, 3)
                li.append(numpy_df)
            y = np.arange(len(csv_list)-1, -1, -1)
            min_value = y.min()
            max_value = y.max()

            # Min-Max 스케일링 적용
            y_scaled = (y - min_value) / (max_value - min_value)
            numpy_target_list.append(y_scaled.reshape(-1,1))
            numpy_df_list.append(np.stack(li,axis=0))


        for j in range(len(numpy_df_list)):
            for i in np.arange(0,len(numpy_df_list[j])-39,1):
                df = numpy_df_list[j][i:i+40]
                temp_x_list.append(df)
        for j in range(len(numpy_target_list)):
            for i in np.arange(0,len(numpy_target_list[j])-39,1):
                df = numpy_target_list[j][i+39]
                temp_y_list.append(df)
        self.X_train_val = temp_x_list
        self.Y_train_val = temp_y_list
    
        
    def __len__(self):
        return self.X_train_val.shape[0]    
    
    def __getitem__(self, idx):
        x = torch.Tensor(self.X_train_val[idx])
        y = self.Y_train_val[idx]
        return x, y


class Test_IMS(Dataset):
    def __init__(
        self,
        data_root: int = DATAFOLDER_ROOT,
        test_bearing_list = ["Bearing1_3","Bearing1_4","Bearing3_2"]
    ):
        super().__init__()
        self.root = data_root
        self.test_bearing_list = test_bearing_list
        numpy_df_list = []
        numpy_target_list = []
        li = []

        for bearing in self.test_bearing_list:
            li = []
            csv_list = sorted(os.listdir(os.path.join(data_root, bearing)))
            print(f"{bearing}, length: {len(csv_list)}, RUL: {len(csv_list) -1 }")
            for i, csv in enumerate(csv_list):  # enumerate를 사용하여 인덱스와 함께 반복
                if bearing == "Bearing1_4" : 
                    df = pd.read_csv(os.path.join(data_root, bearing, csv), header=None,sep=';')
                else : df = pd.read_csv(os.path.join(data_root, bearing, csv), header=None)
                df["time_after_start"] = i
                numpy_df = np.array(df.iloc[:, -3:]).reshape(2560, 3)
                li.append(numpy_df)
                
            y = np.arange(len(csv_list)-1, -1, -1)
            min_value = y.min()
            max_value = y.max()

            # Min-Max 스케일링 적용
            y_scaled = (y - min_value) / (max_value - min_value)
            numpy_target_list.append(y_scaled.reshape(-1,1))
            numpy_df_list.append(np.stack(li,axis=0))

        temp_x_list = []
        temp_y_list = []
        for j in range(len(numpy_df_list)):
            for i in np.arange(0,len(numpy_df_list[j])-39,1):
                
                df = numpy_df_list[j][i:i+40]
                temp_x_list.append(df)
        for j in range(len(numpy_target_list)):
            for i in np.arange(0,len(numpy_target_list[j])-39,1):
                df = numpy_target_list[j][i+39]
                temp_y_list.append(df)
        self.X_test = temp_x_list
        self.Y_test = temp_y_list
    
        
    def __len__(self):
        return self.X_test.shape[0]    
    
    def __getitem__(self, idx):
        x = torch.Tensor(self.X_test[idx])
        y = self.Y_test[idx]
        return x, y
    
class IMSDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/home/mongoose01/mongooseai/data/cms/open_source/phm-ieee-2012-data-challenge-dataset/Learning_set",
        train_val_split: Tuple[int, int] = (0.6, 0.4),
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_val_bearing_list = ["Bearing1_1","Bearing1_2", "Bearing1_5", "Bearing1_6","Bearing1_7",
                      "Bearing2_1","Bearing2_2", "Bearing2_3","Bearing2_4","Bearing2_5","Bearing2_6","Bearing2_7",
                      "Bearing3_1","Bearing3_3",],
        test_bearing_list = ["Bearing1_3","Bearing1_4","Bearing3_2"]
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        self.train_data = Train_IMS(data_root = self.hparams.data_dir, train_val_bearing_list= self.hparams.train_val_bearing_list )
        self.test_data = Test_IMS(data_root= self.hparams.data_dir, test_bearing_list = self.hparams.test_bearing_list )

    def setup(self, stage):
        if not self.data_train and not self.data_val:
            print(self.train_data.__len__())
            self.data_train, self.data_val = random_split(
                dataset=self.train_data,
                lengths=self.hparams.train_val_split,
                generator=torch.Generator().manual_seed(42)
            )

        if not self.data_test:
            print(self.test_data.__len__())
            self.data_test = self.test_data

            
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
        

if __name__ == "__main__":
    _ = IMSDataModule()
