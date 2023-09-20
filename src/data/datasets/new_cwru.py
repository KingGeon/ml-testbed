import os,re
import errno
import random
import urllib.request as urllib
import numpy as np
from scipy.io import loadmat
from sklearn.utils import shuffle
from typing import List
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

exps = ['12DriveEndFault']
rpms = ['1797','1772', '1750', '1730']
timeseries_length = 8192
DATAFOLDER_ROOT = '../src/data/datasets'

def min_max_scaling(data):
    min_val = np.min(data, axis = 0)
    max_val = np.max(data,axis = 0)
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data

def fliter_key(keys):
    fkeys = []
    for key in keys:
        matchObj = re.match( r'(.*)FE_time', key, re.M|re.I)
        if matchObj:
            fkeys.append(matchObj.group(1))
    if(len(fkeys)>1):
        print(keys)
    return fkeys[0]+'DE_time',fkeys[0]+'FE_time'


exps_idx = {
    '12DriveEndFault':0,
    '12FanEndFault':9,
    '48DriveEndFault':0
}

faults_idx = {
    'Normal': 0,
    '0.007-Ball': 1,
    '0.014-Ball': 1,
    '0.021-Ball': 1,
    '0.007-InnerRace': 2,
    '0.014-InnerRace': 2,
    '0.021-InnerRace': 2,
    '0.007-OuterRace6': 3,
    '0.014-OuterRace6': 3,
    '0.021-OuterRace6': 3,
#     '0.007-OuterRace3': 10,
#     '0.014-OuterRace3': 11,
#     '0.021-OuterRace3': 12,
#     '0.007-OuterRace12': 13,
#     '0.014-OuterRace12': 14,
#     '0.021-OuterRace12': 15,
}

def get_class(exp,fault):
    if fault == 'Normal':
        return 0
    return exps_idx[exp] + faults_idx[fault]
    


class CWRU(Dataset):
    def __init__(
        self,
        exps: List[float] = exps,
        rpms: List[int] = rpms, 
        length: str = timeseries_length,
        root: int = DATAFOLDER_ROOT
    ):
        super().__init__()
        self.exps = exps
        self.rpms = rpms
        self.timeseries_length = timeseries_length
        self.root = root
    
        for exp in exps:
            if exp not in ('12DriveEndFault', '12FanEndFault', '48DriveEndFault'):
                print("wrong experiment name: {}".format(exp))
                return
        for rpm in rpms:    
            if rpm not in ('1797', '1772', '1750', '1730'):
                print("wrong rpm value: {}".format(rpm))
                return
        # root directory of all data
        rdir = os.path.join(DATAFOLDER_ROOT)
        print(rdir,exp,rpm)
    
        fmeta = os.path.join(DATAFOLDER_ROOT, 'cwru_meta.txt')
        all_lines = open(fmeta).readlines()
        all_lines = open(fmeta).readlines()
        lines = []
        for line in all_lines:
            l = line.split()
            if (l[0] in exps or l[0] == 'NormalBaseline') and l[1] in rpms:
                if 'Normal' in l[2] or '0.007' in l[2] or '0.014' in l[2] or '0.021' in l[2]:
                    if faults_idx.get(l[2],-1)!=-1:
                        lines.append(l)
 
        self.length = length  # sequence length
        lines = sorted(lines, key=lambda line: get_class(line[0],line[2])) 
        self._load_and_slice_data(rdir, lines)
        # shuffle training and test arrays
        self._shuffle()
        self.all_labels = tuple(((line[0]+line[2]),get_class(line[0],line[2])) for line in lines)
        self.classes = sorted(list(set(self.all_labels)), key=lambda label: label[1]) 
        self.nclasses = len(self.classes)  # number of classes

    def __len__(self):
        return self.y_train.shape[0]    
    
    def __getitem__(self, idx):
        x = torch.Tensor(self.X_train[idx])
        y = self.y_train[idx]
        return x, y
    
    
    def _mkdir(self, path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                print("can't create directory '{}''".format(path))
                exit(1)
 
    def _download(self, fpath, link):
        print(link + " Downloading to: '{}'".format(fpath))
        urllib.urlretrieve(link, fpath)
        
    def _load_and_slice_data(self, rdir, infos):
        self.X_train = np.zeros((0, self.length, 2))
        self.y_train = []
        train_cuts = list(range(0,120000,320))[:330]
        for idx, info in enumerate(infos):
 
            # directory of this file
            fdir = os.path.join(rdir, info[0], info[1])
            self._mkdir(fdir)
            fpath = os.path.join(fdir, info[2] + '.mat')
            print(idx,fpath)
            if not os.path.exists(fpath):
                self._download(fpath, info[3].rstrip('\n'))
 
            mat_dict = loadmat(fpath)
            key1,key2 = fliter_key(mat_dict.keys())
            time_series = np.hstack((mat_dict[key1], mat_dict[key2]))
            idx_last = -(time_series.shape[0] % self.length)
            
            print(time_series.shape)
            
            clips = np.zeros((0, 2))
            for cut in shuffle(train_cuts):
                clips = np.vstack((clips, min_max_scaling(time_series[cut:cut+self.length])))
            clips = clips.reshape(-1, self.length,2)
            self.X_train = np.vstack((self.X_train, clips))
            
            
            self.y_train += [get_class(info[0],info[2])] * 330
            
        self.X_train.reshape(-1, self.length,2)  

 
    def _shuffle(self):
        # shuffle training samples
        index = list(range(self.X_train.shape[0]))
        random.Random(0).shuffle(index)
        self.X_train = self.X_train[index]
        self.y_train = np.array(tuple(self.y_train[i] for i in index))
 

 
if __name__ == '__main__':
    data = CWRU()
    print("end")
