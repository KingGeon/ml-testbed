from typing import List
from torch import nn

class CNNBlock(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int):
        
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, 1, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )
        
    def forward(self, x):
        return self.model(x)

class SimpleCNN(nn.Module):
    def __init__(self,
                 hidden_dim_list: List[int] = [3, 64, 128, 128, 256], 
                 output_size: int = 10):
        super().__init__()
        
        self.encoder = nn.Sequential()
        for i in range(len(hidden_dim_list)-1):
            self.encoder.add_module(f'cnn{i+1}',CNNBlock(input_dim=hidden_dim_list[i], 
                                                       hidden_dim=hidden_dim_list[i+1]))
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
        x = x.mean(dim = (-1, -2))
        x = self.classifier(x)
        return x