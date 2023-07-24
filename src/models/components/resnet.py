from typing import Any, List

from torch import nn 
import torch


class ResBlock(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 downsample: bool):
        
        super().__init__()
        self.downsample = downsample
        stride_for_input = 2 if downsample else 1
        
        self.residual_connection=nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, stride_for_input, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.BatchNorm2d(hidden_dim)
        )
        
        self.downsample_net=nn.Conv2d(input_dim, hidden_dim, 1, stride_for_input, 0)
        self.activation = nn.SiLU()
        
    def forward(self, x):
        z = self.residual_connection(x)
        if self.downsample:
            x = self.downsample_net(x)
        x = torch.add(x, z)
        return self.activation(x)
    
    
class PreActResBlock(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 downsample: bool):
        super().__init__()
        self.downsample = downsample
        stride_for_input = 2 if downsample else 1
        
        self.residual_connection=nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.SiLU(),
            nn.Conv2d(input_dim, hidden_dim, 3, stride_for_input, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
        )
        
        self.downsample_net=nn.Conv2d(input_dim, hidden_dim, 1, stride_for_input, 0)
        
    def forward(self, x):
        z = self.residual_connection(x)
        if self.downsample:
            x = self.downsample_net(x)
        return torch.add(x, z)
    

resnet_blocks_by_name = {
    "ResBlock": ResBlock,
    "PreActResBlock": PreActResBlock
}
    
    
class ResNet(nn.Module):
    def __init__(self,
                 hidden_dim_list: List[int],
                 downsample_list: List[int],
                 input_channel=3,
                 output_size=10,
                 block_name='ResBlock'):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channel, hidden_dim_list[0], 3, 1, 1)
        )
        
        if block_name=='ResBlock':
            self.encoder.add_module('input_batchnorm', nn.BatchNorm2d(hidden_dim_list[0]))
            self.encoder.add_module('input_activation', nn.SiLU())
            
        for i in range(len(hidden_dim_list)-1):
            self.encoder.add_module(f'resblock{i+1}', ResBlock(hidden_dim_list[i], 
                                                                hidden_dim_list[i+1], 
                                                                downsample_list[i]))
        
        self.classifier = nn.Linear(hidden_dim_list[-1], output_size)
        
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim=(-1, -2))
        x = self.classifier(x)
        return x