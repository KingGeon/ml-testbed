from torch import nn
import torch
from typing import List
import numpy as np
import torch.fft as fft


class CONV_LSTM_Classifier(nn.Module):
    def __init__(self):
        super(CONV_LSTM_Classifier, self).__init__()


        self.silu = nn.SiLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=16, stride = 5, padding = 1)
        self.conv1_out = (8192 + 2 * 1 - 16) // 5 + 1
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16, stride=5,padding = 1)
        self.conv2_out = (self.conv1_out + 2 * 1  - 16) // 5 + 1
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=8, stride=1,padding = 1)
        self.conv3_out = (self.conv2_out//2 + 2 * 1  - 8) // 1 + 1
        self.batchnorm3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=8, stride=1,padding = 1)
        self.conv4_out = (self.conv3_out + 2 * 1  - 8) // 1 + 1
        self.batchnorm4 = nn.BatchNorm1d(32)
        self.lstm = nn.LSTM(input_size=self.conv4_out//2, hidden_size=256, batch_first=True)
        self.dense1 = nn.Linear(256, 64)
        self.dense2 = nn.Linear(64, 16)
        self.dense3 = nn.Linear(16, 4)

    def forward(self, x):
        
        z = self.conv1(x.transpose(1,2))
        z = self.batchnorm1(z)
        z = self.silu(z)
        z = self.conv2(z)
        z = self.batchnorm2(z)
        z = self.silu(z)
        z = self.maxpool(z)
        z = self.conv3(z)
        z = self.batchnorm3(z)
        z = self.silu(z)
        z = self.conv4(z)
        z = self.batchnorm4(z)
        z = self.silu(z)
        z = self.maxpool(z)
        z, _ = self.lstm(z)
        z = self.dense1(z[:, -1, :])  # Using the last output of lstm
        z = self.dropout(z)
        z = self.dense2(z)
        z = self.dropout(z)
        outputs = self.dense3(z)
        return outputs
    def predict(self, x):
        self.eval()
        with torch.no_grad():
        
            z = self.conv1(x.transpose(1,2))
            z = self.batchnorm1(z)
            z = self.silu(z)
            z = self.conv2(z)
            z = self.batchnorm2(z)
            z = self.silu(z)
            z = self.maxpool(z)
            z = self.conv3(z)
            z = self.batchnorm3(z)
            z = self.silu(z)
            z = self.conv4(z)
            z = self.batchnorm4(z)
            z = self.silu(z)
            z = self.maxpool(z)
            z, _ = self.lstm(z)
            z = self.dense1(z[:, -1, :])  # Using the last output of lstm
            z = self.dropout(z)
            z = self.dense2(z)
            z = self.dropout(z)
            outputs = self.dense3(z)
        return outputs
            

        
        


if __name__ == '__main__':
    _ = CONV_LSTM_Classifier()