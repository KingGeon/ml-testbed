from torch import nn
import torch
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
class Health_State_Analysis(nn.Module):
    def __init__(self, length_of_signal=2560):
        super(Health_State_Analysis, self).__init__()
        self.length_of_signal = length_of_signal
        self.outlier_threshold = 3
        self.num_bins = 20
        self.fundamental_freq = torch.tensor(11, dtype=torch.float)
        self.sampling_rate = 25600

    def hjorth_calculator(self, data):
        batch_size, seq_length, _ = data.shape

        activity = torch.var(data, dim=1)  # Variance (Activity)

        # Calculate first derivative
        diff1 = torch.diff(data, dim=1)

        mobility = torch.sqrt(torch.var(diff1, dim=1) / torch.var(data, dim=1))  # Mobility

        # Calculate second derivative
        diff2 = torch.diff(diff1, dim=1)

        complexity = torch.sqrt(torch.var(diff2, dim=1) / torch.var(diff1, dim=1))  # Complexity

        hjorth_parameters = torch.stack((activity, mobility, complexity), dim=1)
        return activity,mobility,complexity
    def compute_skewness(self, data):
        # Calculate mean and standard deviation along the 2560 dimension
        mean = torch.mean(data, dim=1, keepdim=True)
        std = torch.std(data, dim=1, keepdim=True)
        
        # Compute the third central moment (raw skewness term)
        third_moment = torch.mean(torch.pow((data - mean), 3), dim=1, keepdim=True)
        
        # Compute the normalized skewness
        skewness = third_moment / torch.pow(std, 3)
        
        return skewness.squeeze(-1)
    def compute_kurtosis(self, data):
        # Calculate mean and standard deviation along the 2560 dimension
        mean = torch.mean(data, dim=1, keepdim=True)
        std = torch.std(data, dim=1, keepdim=True)
        
        # Compute the fourth central moment (raw kurtosis term)
        fourth_moment = torch.mean(torch.pow((data - mean), 4), dim=1, keepdim=True)
        # Compute the normalized kurtosis
        kurtosis = fourth_moment / torch.pow(std, 4)
        
        return kurtosis.squeeze(-1)
    
    def rainflow_counting(self, data):
      batch_size, seq_length, _ = data.shape

      data_diff = data[:, :, 1:] - data[:, :, :-1]

      up_crossings = torch.where(data_diff[:, :, :-1] > 0)
      down_crossings = torch.where(data_diff[:, :, :-1] < 0)

      count = []
      for i in range(batch_size):
          up_indices = up_crossings[0][up_crossings[1] == i]
          down_indices = down_crossings[0][down_crossings[1] == i]

          if len(up_indices) > 0 and len(down_indices) > 0:
              if up_indices[0] > down_indices[0]:
                  down_indices = down_indices[1:]
              if down_indices[-1] < up_indices[-1]:
                  up_indices = up_indices[:-1]

          cycle_counts = [len(up_indices)]
          count.append(cycle_counts)

      return torch.tensor(count, dtype=torch.float32).to("cuda")

    def zero_crossing_rate(self,signal):
        sign_changes = torch.sum(torch.diff(torch.sign(signal)) != 0)
        zcr = sign_changes.float() / (2 * len(signal))
        return zcr

    def calculate_zcr(self, data):
        batch_size, seq_length, _ = data.shape
        data_flatten = data.view(batch_size, -1)  # 2D로 변환
        zcr_values = (torch.diff(torch.sign(data_flatten)) != 0).sum(dim=1, dtype=torch.float) / (2 * seq_length)
        zcr_values = zcr_values.view(batch_size, -1)  # (batch_size, 1)로 변환
        return zcr_values

    def calculate_velocity(self, data):
        time_diff = 1/2560
        velocity_integral = torch.cumsum(data, dim=1) * time_diff

        # 가속도의 절댓값의 합을 velocity_change 특성으로 사용
        velocity_feature = torch.sum(torch.abs(velocity_integral), dim=1)/2560

        return velocity_feature

    def calculate_displacement(self, data):
        time_diff = 1/2560
        velocity_integral = torch.cumsum(data, dim=1) * time_diff
        displacement_integral = torch.cumsum(velocity_integral, dim=1) * time_diff

        displacement_feature = torch.sum(torch.abs(displacement_integral), dim=1)/2560

        return displacement_feature
        
    def calculate_acceleration(self, data):
        time_diff = 1/2560
        acceleration_feature = torch.sum(torch.abs(data),dim = 1)/2560

        return acceleration_feature




    def calculate_thd(self, signal, fundamental_freq, sampling_rate):
        # FFT를 사용하여 주파수 스펙트럼 계산
        spectrum = torch.fft.fft(signal)

        # 기본 주파수와 관련된 스펙트럼 인덱스 계산
        fundamental_index = (fundamental_freq * signal.shape[-2] / sampling_rate).round().int()
        fundamental_power = torch.abs(spectrum[..., fundamental_index, 0])**2



        # 고조파 스펙트럼의 전력 계산
        fundamental_power = torch.abs(spectrum[..., fundamental_index, 0])**2
        harmonic_power = torch.sum(torch.abs(spectrum[..., fundamental_index * 2::fundamental_index, 0])**2, dim=-1)

        # THD 계산 (백분율로 표시)
        thd = (torch.sqrt(harmonic_power) / (fundamental_power  + 1e-10))*100

        return thd.unsqueeze(-1)

    def calculate_entropy(self, signal):
        batch_size, seq_length, _ = signal.shape

        # Calculate bin edges
        min_val = signal.min(dim=1).values
        max_val = signal.max(dim=1).values
        bin_width = (max_val - min_val) / self.num_bins

        # Calculate histogram indices
        hist_indices = ((signal - min_val.unsqueeze(2)) / bin_width.unsqueeze(2)).long()
        hist_indices = torch.clamp(hist_indices, 0, self.num_bins - 1)

        # Flatten hist_indices for each batch
        hist_indices_flat = hist_indices.view(batch_size, -1)

        # Count occurrences of each bin index
        hist_counts = torch.stack([torch.bincount(hist_idx, minlength=self.num_bins) for hist_idx in hist_indices_flat])

        # Normalize histograms
        histograms = hist_counts / torch.sum(hist_counts, dim=1, keepdim=True)

        # Calculate entropy
        entropy = -torch.sum(histograms * torch.log2(histograms + 1e-10), dim=1)

        return entropy.unsqueeze(-1)




    def forward(self, inputs):
        mean = torch.mean(inputs, dim=1)
        maximum = torch.max(inputs, dim=1).values
        minimum = torch.min(inputs, dim=1).values
        rms = torch.sqrt(torch.mean(inputs**2, dim=1))
        var = torch.var(inputs, dim=1)
        #acceleration, velocity, displacement = self.calculate_acceleration(inputs), self.calculate_velocity(inputs),self.calculate_displacement(inputs)
        skewness = self.compute_skewness(inputs)  # 상한값을 1e8로 제한
        #print("skewness",skewness)
        kurtosis = self.compute_kurtosis(inputs)
        #kurtosis = torch.clamp(kurtosis, max=1e3)  # 상한값을 1e8로 제한
        #print("kurtosist",kurtosis)
        shape_factor = rms * self.length_of_signal / torch.sum(torch.abs(inputs), dim=1)
        #print("shape_factor",shape_factor)
        crest_factor = torch.max(torch.abs(inputs), dim=1).values / rms
        #print("crest_factor",crest_factor)
        impulse_factor = torch.max(torch.abs(inputs), dim=1).values * self.length_of_signal / torch.sum(torch.abs(inputs), dim=1)
        #print("impulse_factor",impulse_factor)
        #entropy = self.calculate_entropy(inputs)
        #print("entropy",entropy)
        outliers = torch.sum((torch.abs(inputs.squeeze() - inputs.mean(dim=1)) > self.outlier_threshold * inputs.std(dim=1)).int(),dim = 1).unsqueeze(dim = 1)
        #print("outliers",outliers)
        #thd = self.calculate_thd(inputs,self.fundamental_freq,self.sampling_rate) 빼는게 좋아보임
        #print("thd",thd)
        #rainflow_count = self.rainflow_counting(inputs)
        zcr = self.calculate_zcr(inputs)
        activity,mobility,complexity = self.hjorth_calculator(inputs)
        p2p = maximum - minimum
        stacked_stats = torch.stack([mean,maximum, minimum,p2p,var, rms,skewness,kurtosis,shape_factor,impulse_factor,outliers,zcr,activity,mobility,complexity], dim=1)
        return stacked_stats.view(inputs.shape[0],-1)
class FFT_Health_State_Analysis(nn.Module):
    def __init__(self, length_of_signal=2560):
        super(FFT_Health_State_Analysis, self).__init__()
        self.length_of_signal = length_of_signal
        self.outlier_threshold = 3
        self.num_bins = 20
        self.fundamental_freq = torch.tensor(11, dtype=torch.float)
        self.sampling_rate = 25600

    def compute_skewness(self, data):
        # Calculate mean and standard deviation along the 2560 dimension
        mean = torch.mean(data, dim=1, keepdim=True)
        std = torch.std(data, dim=1, keepdim=True)
        
        # Compute the third central moment (raw skewness term)
        third_moment = torch.mean(torch.pow((data - mean), 3), dim=1, keepdim=True)
        
        # Compute the normalized skewness
        skewness = third_moment / torch.pow(std, 3)
        
        return skewness.squeeze(-1)
    def compute_kurtosis(self, data):
        # Calculate mean and standard deviation along the 2560 dimension
        mean = torch.mean(data, dim=1, keepdim=True)
        std = torch.std(data, dim=1, keepdim=True)
        
        # Compute the fourth central moment (raw kurtosis term)
        fourth_moment = torch.mean(torch.pow((data - mean), 4), dim=1, keepdim=True)
        # Compute the normalized kurtosis
        kurtosis = fourth_moment / torch.pow(std, 4)
        
        return kurtosis.squeeze(-1)
    


    def forward(self, inputs):
        
        top_10_mean_freq = torch.mean(torch.topk(inputs, k=10, axis = 1).indices.float(),axis = 1)  # Convert to float before calculating mean
        max_freq = torch.Tensor(torch.topk(inputs, k=1, axis = 1).indices.float()).squeeze(-1)
        top10_rms = torch.sqrt(torch.mean(torch.topk(inputs, k=10, axis = 1).values**2, dim=1))
    
        stacked_stats = torch.stack([top_10_mean_freq,max_freq,top10_rms], dim=1)
        return stacked_stats.view(inputs.shape[0],-1)

class CONV_LSTM(nn.Module):
    def __init__(self):
        super(CONV_LSTM, self).__init__()
        self.fft_real = FFTReal()
        self.fft_imag = FFTImag()
        self.hs = Health_State_Analysis()
        self.fft_hs = FFT_Health_State_Analysis()
        self.silu = nn.SiLU()
        self.softmax = nn.Softmax(dim=-1)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        self.layer_nrom1 = nn.LayerNorm(64)
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=40, stride = 3, padding = 1)
        self.conv1_out = (2560 + 2 * 1 - 40) // 3 + 1
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8, stride=2,padding = 1)
        self.conv2_out = (self.conv1_out + 2 * 1  - 8) // 2 + 1
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=1,padding = 1)
        self.conv3_out = (self.conv2_out//2 + 2 * 1  - 4) // 1 + 1
        self.batchnorm3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=4, stride=1,padding = 1)
        self.conv4_out = (self.conv3_out + 2 * 1  - 4) // 1 + 1
        self.batchnorm4 = nn.BatchNorm1d(32)
        self.lstm = nn.LSTM(input_size=self.conv4_out//2, hidden_size=32, batch_first=True, bidirectional = True)
        self.dense1 = nn.Linear(64 + 15 , 64)
        self.dense2 = nn.Linear(64, 16)
        self.dense3 = nn.Linear(16, 1)

    def forward(self, x):
        r = self.fft_real(x[:,:,0].unsqueeze(-1))
        i = self.fft_imag(x[:,:,0].unsqueeze(-1))
        hs = self.hs(x[:,:,0].unsqueeze(-1))
        fft_hs = self.fft_hs(r)
        t = x[:,0,2].unsqueeze(-1)
        dynamic_features = torch.cat((i, r, x[:,:,0].unsqueeze(-1),x[:,:,2].unsqueeze(-1)), dim=2)
        dynamic_features = dynamic_features.transpose(1,2)
        z = self.conv1(dynamic_features)
        z = self.batchnorm1(z)
        z = self.silu(z)
        z = self.dropout(z)
        z = self.conv2(z)
        z = self.batchnorm2(z)
        z = self.silu(z)
        z = self.dropout(z)
        z = self.maxpool(z)
        z = self.conv3(z)
        z = self.batchnorm3(z)
        z = self.silu(z)
        z = self.dropout(z)
        z = self.conv4(z)
        z = self.batchnorm4(z)
        z = self.silu(z)
        z = self.maxpool(z)
        z, _ = self.lstm(z)
        z_mean = torch.mean(z, dim=1) # Calculate mean along the second dimension
        z = torch.cat((z_mean, hs), dim=-1)  # Concatenate the mean tensor with hs

        z = self.dense1(z)  # Using the last output of lstm
        z = self.layer_nrom1(z)
        z = self.dropout(z)
        z = self.silu(z)
        z = self.dense2(z)
        z = self.dropout(z)
        z = self.silu(z)
        outputs = self.dense3(z)
        return outputs