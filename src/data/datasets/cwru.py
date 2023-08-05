import os
import sys
import requests
import torch
from typing import List
from urllib.error import URLError

from torch.utils.data import Dataset
from scipy.io import loadmat
import numpy as np

# Normal
# 12k Drive End Bearinng

fault_diamter_list = ['7', '14', '21', '28']
rpm_list = ['1797', '1772', '1750', '1730']
sensor_position_list = ['DE', 'FE', 'BA']

DATAFOLDER_ROOT = './data/CWRU'


class CWRU(Dataset):
    def __init__(
        self,
        fault_diameter: List[float] = fault_diamter_list,
        rpm: List[int] = rpm_list, 
        sensor_position: str = sensor_position_list,
        timeseries_len: str = 8192,
        number_of_samples: int = 1000,
        download: bool = False,
        train: bool = True,
        root: int = DATAFOLDER_ROOT
    ):
        super().__init__()
        self.fault_diameter = fault_diameter
        self.rpm = rpm
        self.timeseries_len = timeseries_len
        self.number_of_samples = number_of_samples
        self.sensor_position = sensor_position
        self.download = download
        self.train = train
        self.root = root
        
        if download:
            self._download()
            
        # if fault_diameter not in fault_diamter_list:
        #     raise ValueError("Fault Severity Error - 0.007, 0,014, 0.021, 0.028 중 하나를 입력하세요.")
          
        # if rpm not in rpm_list:
        #     raise ValueError("RPM Error - 1797, 1772, 1750, 1730 중 하나를 입력하세요")
        
        # if sensor_position not in sensor_position_list:
        #     raise ValueError("Sensor Position Error - 'drive_end' 혹은 'fan_end' 중 하나를 입력하세요.")

        
        normal_dataset, n_len = self._load_normal_dataset()
        inner_race_dataset, i_len = self._load_fault_dataset('inner_race')
        ball_dataset, b_len = self._load_fault_dataset('inner_race')
        outer_race_dataset, o_len = self._load_fault_dataset('outer_race')
        
        normal_label = [0] * n_len
        inner_label = [1] * i_len
        ball_label = [2] * b_len
        outer_label = [3] * o_len
        
        self.data = np.concatenate([normal_dataset, inner_race_dataset, ball_dataset, outer_race_dataset])
        self.targets = np.concatenate([normal_label, inner_label, ball_label, outer_label])
        
        print(self.data.shape)
        print(self.targets.shape)
        
    def __len__(self):
        return self.targets.shape[0]    
    
    def __getitem__(self, idx):
        x = torch.Tensor(self.data[idx])
        y = torch.Tensor(self.targets[idx])
        return x, y
        
        
    def _load_mat_file(self, filename):
        file_dict = loadmat(filename)
        data = []
        FE_flag =True
        DE_flag = True
        for key in file_dict.keys():
            if ('time' in key):
                if FE_flag:
                    data.append(file_dict[key])
                    FE_flag = False

        return np.stack(data)
    
    
    def _load_fault_dataset(self, fault_class):
        data_list = []
        path = os.path.join(DATAFOLDER_ROOT, fault_class)
        file_list = os.listdir(path)
        for filename in file_list:
            splits = filename.split('_')
            if splits[0] in self.fault_diameter:
                if splits[1] in self.sensor_position:
                    if splits[2] in self.rpm:
                        data_stack = self._load_mat_file(os.path.join(path, filename))
                        for data in data_stack:
                            length = data.shape[0]
                            start_indexes = np.random.choice(np.arange(length - self.timeseries_len), self.number_of_samples)
                            for start_index in start_indexes:
                                data_list.append(data[start_index:start_index + self.timeseries_len, :]) # (length, 2channel)
        return np.stack(data_list, axis=0), len(data_list)
    
    
    def _load_normal_dataset(self):
        data_list = []
        path = os.path.join(DATAFOLDER_ROOT, 'normal')
        file_list = os.listdir(path)
        for filename in file_list:
            data_stack = self._load_mat_file(os.path.join(path, filename))
            for data in data_stack:
                length = data.shape[0]
                start_indexes = np.random.choice(np.arange(length - self.timeseries_len), self.number_of_samples)
                for start_index in start_indexes:
                    data_list.append(data[start_index:start_index + self.timeseries_len, :]) # (length, 2channel)
        return np.stack(data_list, axis=0), len(data_list)
            
            
    def _download(self):
        os.makedirs(self.root, exist_ok=True)
        self._download_normal_data()
        self._download_inner_race_fault_data()
        self._download_ball_fault_data()
        self._download_outer_race_fault_data()
        
    def _download_from_url(self, url, folder, filename):
        response = requests.get(url)
        if response.status_code == 200:
            with open(os.path.join(self.root, folder, filename), 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            raise URLError(f"{filename}.mat 파일을 찾을 수 없습니다. 데이터 저장소가 옮겨진 것으로 추정됩니다.")
        
    def _download_normal_data(self):
        os.makedirs(os.path.join(self.root, 'normal'), exist_ok=True)
        for i, rpm in enumerate(rpm_list):
            self._download_from_url(f'https://engineering.case.edu/sites/default/files/{97+i}.mat', 
                                    'normal', 
                                    f'{rpm}_{97 + i}.mat')
            
    def _download_inner_race_fault_data(self):
        os.makedirs(os.path.join(self.root, 'inner_race'), exist_ok=True)
        start_index_list = [105, 169, 209, 3001, 278, 274, 270]
        positions = ['DE', 'DE', 'DE', 'DE', 'FE', 'FE', 'FE']
        diameters = [7, 14, 21, 28, 7, 14, 21]

        for i, rpm in enumerate(rpm_list):
            for j, (start_index, position, diameter) in enumerate(zip(start_index_list, positions, diameters)):
                self._download_from_url(f'https://engineering.case.edu/sites/default/files/{start_index + i}.mat', 
                                        'inner_race', 
                                        f'{diameter}_{position}_{rpm}_{start_index + i}.mat')

                
    def _download_ball_fault_data(self):
        os.makedirs(os.path.join(self.root, 'ball'), exist_ok=True)
        start_index_list = [118, 197, 234, 282, 286, 290]
        positions = ['DE', 'DE', 'DE', 'FE', 'FE', 'FE']
        diameters = [7, 14, 21, 7, 14, 21]

        for i, rpm in enumerate(rpm_list):
            for j, (start_index, position, diameter) in enumerate(zip(start_index_list, positions, diameters)):
                self._download_from_url(f'https://engineering.case.edu/sites/default/files/{start_index + i}.mat', 
                                        'ball', 
                                        f'{diameter}_{position}_{rpm}_{start_index + i}.mat')


    def _download_outer_race_fault_data(self):
        os.makedirs(os.path.join(self.root, 'outer_race'), exist_ok=True)
        start_index_list = [130, 197, 234, 144, 246, 156, 258, 294, 298, 302]
        positions = ['DE', 'DE', 'DE', 'DE', 'DE', 'DE', 'DE', 'FE', 'FE', 'FE']
        diameters = [7, 14, 21, 7, 21, 7, 21, 7, 7, 7]
        locations = ['Center', 'Center', 'Center', 'Orthogonal', 'Orthogonal', 'Opposite', 'Opposite', 'Center', 'Orthogonal', 'Opposite']

        for i, rpm in enumerate(rpm_list):
            for j, (start_index, position, diameter, location) in enumerate(zip(start_index_list, positions, diameters, locations)):
                if start_index == 156:
                    correction_list = [0, 1, 1, 1]
                    self._download_from_url(f'https://engineering.case.edu/sites/default/files/{start_index + i + correction_list[i]}.mat', 
                                            'outer_race', 
                                            f'{diameter}_{position}_{rpm}_{location}_{start_index + i + correction_list[i]}.mat')
                    
                elif start_index == 302:
                    correction_list = [0, 2, 2, 2]
                    self._download_from_url(f'https://engineering.case.edu/sites/default/files/{start_index + i + correction_list[i]}.mat', 
                                            'outer_race', 
                                            f'{diameter}_{position}_{rpm}_{location}_{start_index + i + correction_list[i]}.mat')

                else:
                    self._download_from_url(f'https://engineering.case.edu/sites/default/files/{start_index + i}.mat', 
                                            'outer_race', 
                                            f'{diameter}_{position}_{rpm}_{location}_{start_index + i}.mat')

        
        data_indexes = [313, 310, 309, 311, 312, 315, 316, 317, 318]
        positions = ['FE'] * 9
        diameters = [14, 14, 14, 14, 14, 21, 21, 21, 21]
        locations = ['Center', 'Orthogonal', 'Orthogonal', 'Orthogonal', 'Orthogonal', 'Center', 'Orthogonal', 'Orthogonal', 'Orthogonal']
        
        for i, (data_index, position, diameter, location) in enumerate(zip(data_indexes, positions, diameters, locations)):
            self._download_from_url(f'https://engineering.case.edu/sites/default/files/{data_index}.mat', 
                                            'outer_race', 
                                            f'{diameter}_{position}_{rpm}_{location}_{data_index}.mat')
        
    
        
# diameter, rpm, 위치
# fault : 폴더별 분류
        
if __name__ == '__main__':
    cwru = CWRU()
    print("end")
        
        