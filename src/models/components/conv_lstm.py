from torch import nn
import torch
from typing import List
import numpy as np
import torch.fft as fft

class FFTReal(nn.Module):
    def forward(self, inputs):
        inputs = inputs.type(torch.complex64)
        fft_result = fft.fft(inputs)
        return torch.real(fft_result)

class FFTImag(nn.Module):
    def forward(self, inputs):
        inputs = inputs.type(torch.complex64)
        fft_result = fft.fft(inputs)
        return torch.imag(fft_result)
    
class CONV_LSTM(nn.Module):
    def __init__(self,
                 input_dim: int = 8192,
                 in_channel: int = 3,
                 conv1_channel: int = 32,
                 conv2_channel: int = 64,
                 conv3_channel: int = 64,
                 conv4_channel: int = 32,
                 conv1_kernel_size: int = 80,
                 conv2_kernel_size: int = 80,
                 conv3_kernel_size: int = 20,
                 conv4_kernel_size: int = 20,
                 pool_kernel_size: int = 2,
                 drop_out: float = 0.5,
                 hidden_dim_list: List[int] = [256, 64, 16],
                 output_size: int = 4):
        super().__init__()
        self.fft_real = FFTReal()
        self.fft_imag = FFTImag()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=conv1_channel, kernel_size=conv1_kernel_size, stride = 2, padding = 1),
            nn.BatchNorm1d(conv1_channel),
            nn.SiLU()
        )
        self.conv1_out = (input_dim + 2 * 1 - conv1_kernel_size) // 2 + 1
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=conv1_channel, out_channels=conv2_channel, kernel_size=conv2_kernel_size, stride=2,padding = 1),
            nn.BatchNorm1d(conv2_channel),
            nn.SiLU(),
            nn.MaxPool1d(pool_kernel_size)
        )
        
        self.conv2_out = (self.conv1_out + 2 * 1 - conv2_kernel_size) // 2 + 1

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_channel, out_channels=conv3_channel, kernel_size=conv3_kernel_size, stride=1,padding = 1),
            nn.BatchNorm1d(conv3_channel),
            nn.SiLU()
        )
        self.dropout = nn.Dropout(drop_out)
        self.conv3_out = (self.conv2_out//2 + 2 * 1 - conv3_kernel_size) // 1 + 1

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=conv3_channel, out_channels=conv4_channel, kernel_size=conv4_kernel_size, stride=1,padding = 1),
            nn.BatchNorm1d(conv4_channel),
            nn.SiLU(),
            nn.MaxPool1d(pool_kernel_size)
        )
        self.conv4_out = (self.conv3_out + 2 * 1 - conv4_kernel_size) // 1 + 1
        self.dropout = nn.Dropout(drop_out)
        self.lstm = nn.LSTM(input_size=self.conv4_out//2, hidden_size=hidden_dim_list[0], batch_first=True)
        
        self.lin1 = nn.Linear(hidden_dim_list[0], hidden_dim_list[1])
        self.dropout = nn.Dropout(drop_out)
        self.lin2 = nn.Linear(hidden_dim_list[1], hidden_dim_list[2])
        self.dropout = nn.Dropout(drop_out)
        self.lin3 = nn.Linear(hidden_dim_list[2], output_size)
        
    def forward(self, x):
        i = self.fft_imag(x)
        r = self.fft_real(x)
        dynamic_features = torch.cat((i, r, x), dim=2)
        dynamic_features = dynamic_features.transpose(1,2)
        z = self.conv1(dynamic_features)
        z = self.conv2(z)
        z = self.dropout(z)
        z = self.conv3(z)
        z = self.conv4(z)
        z = self.dropout(z)
        z, _ = self.lstm(z)
        z = self.lin1(z[:, -1, :])
        z = self.dropout(z)
        z = self.lin2(z)
        z = self.dropout(z)
        z = self.lin3(z)
        return z
            

        
        


if __name__ == '__main__':
    _ = CONV_LSTM()