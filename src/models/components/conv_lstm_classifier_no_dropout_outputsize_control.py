import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

class FFTReal(nn.Module):
    def forward(self, inputs, dim):
        inputs = inputs.type(torch.complex64)
        fft_result = fft.fft(inputs, dim = dim)
        return torch.real(fft_result)

class Discriminator(nn.Module):
    def __init__(self, input_size, dense1_out, dense2_out):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, dense1_out)
        self.fc2 = nn.Linear(dense1_out, dense2_out)
        self.fc3 = nn.Linear(dense2_out, 1)  # Output layer for binary classification

        self.silu = nn.SiLU(0.2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.silu(self.fc1(x))
        x = self.silu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Sigmoid for binary classification
        return x
    
class TripletLoss(nn.Module):
    def __init__(self, margin=2):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.triplet_loss(anchor, positive, negative)
    

class Health_State_Analysis(nn.Module):
    def __init__(self, length_of_signal=8192):
        super(Health_State_Analysis, self).__init__()
        self.length_of_signal = length_of_signal
        self.outlier_threshold = 3


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
        time_diff = 1/self.length_of_signal
        velocity_integral = torch.cumsum(data, dim=1) * time_diff
        # 가속도의 절댓값의 합을 velocity_change 특성으로 사용
        velocity_feature = torch.sum(torch.abs(velocity_integral), dim=1)/self.length_of_signal
        return velocity_feature

    def calculate_displacement(self, data):
        time_diff = 1/self.length_of_signal
        velocity_integral = torch.cumsum(data, dim=1) * time_diff
        displacement_integral = torch.cumsum(velocity_integral, dim=1) * time_diff
        displacement_feature = torch.sum(torch.abs(displacement_integral), dim=1)/self.length_of_signal
        return displacement_feature
        
    def calculate_acceleration(self, data):
        time_diff = 1/self.length_of_signal
        acceleration_feature = torch.sum(torch.abs(data),dim = 1)/self.length_of_signal
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
        skewness = self.compute_skewness(inputs)  
        kurtosis = self.compute_kurtosis(inputs)
        shape_factor = rms * self.length_of_signal / torch.sum(torch.abs(inputs), dim=1)
        crest_factor = torch.max(torch.abs(inputs), dim=1).values / rms
        impulse_factor = torch.max(torch.abs(inputs), dim=1).values * self.length_of_signal / torch.sum(torch.abs(inputs), dim=1)
        outliers = torch.sum((torch.abs(inputs.squeeze() - inputs.mean(dim=1)) > self.outlier_threshold * inputs.std(dim=1)).int(),dim = 1).unsqueeze(dim = 1)
        zcr = self.calculate_zcr(inputs)
        activity,mobility,complexity = self.hjorth_calculator(inputs)
        p2p = maximum - minimum
        stacked_stats = torch.stack([mean, maximum, minimum, p2p,var, rms, skewness, kurtosis,
                                     crest_factor, shape_factor, impulse_factor, outliers,
                                     zcr, activity, mobility, complexity], dim=1)
        return stacked_stats.view(inputs.shape[0],-1)
    
class FFT_Health_State_Analysis(nn.Module):
    def __init__(self, topk):
        super(FFT_Health_State_Analysis, self).__init__()
        self.topk = topk

    def forward(self, inputs):
        inputs = inputs[:8192//2]
        top3_mean_freq = torch.mean(torch.topk(inputs, k=3, axis = 1).indices.float(),axis = 1)  
        top3_rms = torch.sqrt(torch.mean(torch.topk(inputs, k=3, axis = 1).values**2, dim=1))
        max_freq = torch.Tensor(torch.topk(inputs, k=1, axis = 1).indices.float()).squeeze(-1) 
        max_rms = torch.sqrt(torch.mean(torch.topk(inputs, k=1, axis = 1).values**2, dim=1))
        top_k_freqs = torch.topk(inputs, k=self.topk, axis = 1).indices.float() 
        stacked_stats = torch.stack([top3_mean_freq,top3_rms,max_freq,max_rms], dim=1)
        stacked_stats = torch.cat([stacked_stats, top_k_freqs], dim=1)
        return stacked_stats.view(inputs.shape[0],-1)

class EngineOrderFFT(nn.Module):
    def __init__(self):
        super(EngineOrderFFT, self).__init__()

    def engine_order_fft(self, signal, rpm, sf = 8192, res = 40, ts= 1):
        # signal shape: (batch, signal_length, channel_size)
        batch_size, signal_length, channel_size = signal.shape
        # Truncate the signal
        truncated_signal = signal[:, :int(sf*ts), :]
        # Calculate the padding length for each item in the batch
        pad_lengths = (res * 60 / rpm - ts) * sf
        max_pad_length = torch.max(pad_lengths).long()  # Find the maximum padding length
        # Apply padding to the whole batch
        padded_signal = F.pad(truncated_signal, (0, 0, 0, max_pad_length))
        # Remove extra padding for each item in the batch
        padded_fft_result = [torch.abs(torch.fft.fft(padded_signal[i, :signal_length + int(pad_lengths[i]), :].unsqueeze(0), dim = 1))[:,:sf,:] for i in range(batch_size)]
        # Stack the padded signals and perform FFT
        stacked_eofft = torch.stack(padded_fft_result, dim=0).view(batch_size,signal_length,channel_size)
        return torch.abs(stacked_eofft)

    def forward(self, inputs, rpm):
        return self.engine_order_fft(inputs, rpm)

class RpmEstimator(nn.Module):
    def __init__(self):
        super(RpmEstimator, self).__init__()

    def original_fft(self,signal, sf):
        # signal: (batch, signal_length, channel_size)
        y = torch.fft.fft(signal, dim=1) / signal.size(1)
        y = torch.abs(y[:, :signal.size(1)//2, :])
        k = torch.arange(signal.size(1)).unsqueeze(0).expand(signal.size(0), -1).unsqueeze(2)
        f0 = k * sf / signal.size(1)
        f0 = f0[:, :signal.size(1)//2, :]
        return f0, y

    def is_peak_at_frequency(self,freq, spectrum, threshold):
        # freq: (batch_size, )
        # spectrum: (batch, signal_length//2, channel_size)
        # threshold: (batch, channel_size)
        mask = torch.zeros(spectrum.size(0), dtype=torch.bool)
        for i, f in enumerate(freq.long()):
            mask[i] = spectrum[i, f, :] > threshold[i, :]
        return mask

    def estimate_rpm(self,batch_signal, sf=8192, f_min=27.6, f_max=29.1, f_r=1, M=60, c=2):
        # batch_signal: (batch, signal_length, channel_size)
        f, magnitude_spectrum = self.original_fft(batch_signal, sf)
        candidates = torch.arange(f_min, f_max, f_r/M).unsqueeze(0).expand(batch_signal.size(0), -1)
        probabilities = torch.ones_like(candidates)
        threshold = torch.mean(magnitude_spectrum, dim=1) * 1.5
        for i, fc in enumerate(candidates.T):
            for k in range(1, M+1):
                harmonic_freq = k * fc
                mask = ~self.is_peak_at_frequency(harmonic_freq, magnitude_spectrum, threshold)
                probabilities[mask, i] /= c
        estimated_speeds = candidates[torch.arange(batch_signal.size(0)), torch.argmax(probabilities, dim=1)]
        return estimated_speeds * 60
    
    def forward(self, signal):
        return self.estimate_rpm(signal)
    
class MixStyle(nn.Module):
    def __init__(self):
        super(MixStyle, self).__init__()

    def forward(self, x):
        eps = 1e-6
        device = x.device  # Get the device of the input tensor
        B = x.size(0)
        alpha = torch.tensor([0.1]).to(device)  # Ensure alpha is on the same device
        mu = x.mean(dim=[2], keepdim=True)  # compute instance mean
        var = x.var(dim=[2], keepdim=True)  # compute instance variance
        sig = (var + eps).sqrt()  # compute instance standard deviation
        mu, sig = mu.detach(), sig.detach()  # block gradients
        x_normed = (x - mu) / sig  # normalize input
        lmda = torch.distributions.beta.Beta(alpha, alpha).sample((B, 1, 1)).squeeze(-1).to(device)  # Ensure lambda is on the same device
        perm = torch.randperm(B).to(device)  # Ensure permutation is on the same device
        mu2, sig2 = mu[perm], sig[perm]  # shuffling
        mu_mix = mu * lmda + mu2 * (1 - lmda)  # generate mixed mean
        sig_mix = sig * lmda + sig2 * (1 - lmda)  # generate mixed standard deviation
        return x_normed * sig_mix + mu_mix  # denormalize input using the mixed statistics


class CONV_LSTM_Classifier(nn.Module):
    def __init__(
        self,
        in_length: int = 8192,
        output_size: int = 5,
        out_channel1_size = 16,
        out_channel2_size = 64,
        out_channel3_size = 16,
        out_channel4_size = 8,
        kernel1_size = 64,
        kernel2_size = 32,
        kernel3_size = 16,
        kernel4_size = 4,
        lstm_hidden_size : int = 64,
        dense1_out_size : int = 128,
        dense2_out_size : int = 64,
        use_raw: bool = False,
        use_bandpass: bool = True,
        use_bandpass_fft: bool = True,
        use_envelope_spectrum: bool = True,
        use_envelope: bool = True,
        use_eofft: bool = True,
        use_fft_stat: bool = True,
        use_stat: bool = True,
        topk_freq: int = 20,
        discriminator_dense1_outsize = 128,
        discriminator_dense2_outsize = 32,
    ):
        super(CONV_LSTM_Classifier, self).__init__()
        self.output_size = output_size
        self.mix_style = MixStyle()
        self.dropout  = nn.Dropout(0.3)
        self.in_length = in_length
        self.fft_real = FFTReal()
        self.fft_hs = FFT_Health_State_Analysis(topk = topk_freq)
        self.hs = Health_State_Analysis()
        self.layer_norm1 = nn.LayerNorm(dense1_out_size)
        self.layer_norm2 = nn.LayerNorm(dense2_out_size)
        self.silu = nn.SiLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.use_eofft = use_eofft
        self.use_envelope = use_envelope
        self.use_envelope_spectrum = use_envelope_spectrum
        self.use_fft_stat = use_fft_stat
        self.use_stat = use_stat
        self.use_raw = use_raw
        self.use_bandpass = use_bandpass
        self.use_bandpass_fft = use_bandpass_fft

        self.in_channels =  1 + 9 * int(use_bandpass_fft) +  9 * int(use_bandpass) + 1 * int(use_raw) + int(use_envelope) + int(use_envelope_spectrum)
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=out_channel1_size, kernel_size=kernel1_size, stride=4, padding=1)
        self.batchnorm1 = nn.BatchNorm1d(out_channel1_size)
        self.conv2 = nn.Conv1d(in_channels=out_channel1_size, out_channels=out_channel2_size, kernel_size=kernel2_size, stride=2, padding=1)
        self.batchnorm2 = nn.BatchNorm1d(out_channel2_size)
        self.conv3 = nn.Conv1d(in_channels=out_channel2_size, out_channels=out_channel3_size, kernel_size=kernel3_size, stride=2, padding=1)
        self.batchnorm3 = nn.BatchNorm1d(out_channel3_size)
        self.conv4 = nn.Conv1d(in_channels=out_channel3_size, out_channels=out_channel4_size, kernel_size=kernel4_size, stride=1, padding=1)
        self.batchnorm4 = nn.BatchNorm1d(out_channel4_size)
        # Calculate the output size after convolutions
        self.conv4_out = self.calculate_conv_output_size()
        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.conv4_out, hidden_size=lstm_hidden_size, batch_first=True, bidirectional=True)
        # Since LSTM is bidirectional, we concatenate the outputs, hence the hidden size is doubled
        self.dense1 = nn.Linear(lstm_hidden_size * 2 + 16 * int(self.use_stat) + 2 * (topk_freq + 4) * int(self.use_fft_stat), dense1_out_size)  # Hidden size is doubled because LSTM is bidirectional
        # Dense layers
        self.dense2 = nn.Linear(dense1_out_size, dense2_out_size)
        self.dense3 = nn.Linear(dense2_out_size, output_size)

        self.triplet_loss = TripletLoss()
        self.discriminator = Discriminator(dense1_out_size,discriminator_dense1_outsize,discriminator_dense2_outsize)
              
    def fault_classfier(self,z):
        
        z = self.dense3(self.dense2(z))
        return z
    
    def prepare_triplet(self, batch):
    # batch_data: 입력 데이터, labels: 데이터 레이블
        batch_data, labels = batch
        batch_data = batch_data.squeeze(0)
        labels = labels.squeeze(0)
        n = batch_data.size(0)  # 배치 크기

        # Triplet 샘플을 저장할 리스트 초기화
        anchors = []
        positives = []
        negatives = []

        for i in range(n):
            anchor = batch_data[i]
            anchor_label = labels[i]

            # Positive 샘플 선택 (동일 클래스)
            positive_indices = (labels == anchor_label).nonzero(as_tuple=True)[0]
            positive_index = positive_indices[torch.randint(len(positive_indices), (1,))].item()
            positive = batch_data[positive_index]

            # Negative 샘플 선택 (다른 클래스)
            negative_indices = (labels != anchor_label).nonzero(as_tuple=True)[0]
            negative_index = negative_indices[torch.randint(len(negative_indices), (1,))].item()
            negative = batch_data[negative_index]

            anchors.append(anchor.unsqueeze(0))
            positives.append(positive.unsqueeze(0))
            negatives.append(negative.unsqueeze(0))

        # 배치 형태로 변환
        anchors = torch.cat(anchors, dim=0)
        positives = torch.cat(positives, dim=0)
        negatives = torch.cat(negatives, dim=0)

        return anchors, positives, negatives
    def fake_generator(self, x, alpha):
        # 원본 데이터 x에 잡음 추가 및 스케일 조정
        alpha = alpha.to(x.device)
        noise = torch.randn_like(x) * alpha  # 배치 데이터에 대한 잡음 추가

        # 0.95에서 1.05 사이의 랜덤한 상수 c 생성
        beta = 0.8 + torch.rand(1).to(x.device) * 0.4
        scaled_x = x * beta + torch.abs(noise)   # x에 c를 곱하고 랜덤한 노이즈를 추가
        return scaled_x
    
    def calculate_adversarial_loss_and_acc(self, real_data, fake_data):
        # Adversarial Loss calculation
        real_output = self.discriminator(real_data)
        fake_output = self.discriminator(fake_data)

        adversarial_loss = F.binary_cross_entropy(real_output, torch.ones_like(real_output)) + \
                        F.binary_cross_entropy(fake_output, torch.zeros_like(fake_output))

        threshold = 0.5
        # Since outputs are probabilities, directly compare with the threshold
        real_predictions = (real_output > threshold).float()
        fake_predictions = (fake_output > threshold).float()

        # Calculate accuracies
        real_accuracy = torch.mean((real_predictions == 1).float())
        fake_accuracy = torch.mean((fake_predictions == 0).float())

        return adversarial_loss, real_accuracy, fake_accuracy

    
    def calculate_conv_output_size(self):
        # Dummy input for calculating the size of LSTM input
        # This assumes the input size to the first conv layer is (batch_size, channels, sequence_length)
        dummy_input = torch.zeros(1, self.in_channels, self.in_length)
        dummy_output = self.maxpool(self.batchnorm4(self.conv4(self.silu(self.batchnorm3(self.conv3(self.silu(self.maxpool(self.batchnorm2(self.conv2(self.silu(self.batchnorm1(self.conv1(dummy_input)))))))))))))
        return dummy_output.shape[-1]
    
    def forward(self, x):
        eofft = x[:,:,-1].unsqueeze(-1)
        envelope_spectrum = x[:,:,-2].unsqueeze(-1)
        envelope = x[:,:,-3].unsqueeze(-1)
        
        y = x[:,:,1:-3]
        x = x[:,:,0].unsqueeze(-1)
        fft = self.fft_real(x, 1)
        fft_bandpassed =self.fft_real(y, 1)
        if self.use_bandpass_fft:
            if self.use_bandpass:
                if self.use_raw:
                    if self.use_envelope :
                        if self.use_envelope_spectrum:
                            if self.use_eofft:
                                dynamic_features = torch.cat((x, y, eofft, fft_bandpassed, envelope, envelope_spectrum), dim=2)
                            else:
                                dynamic_features = torch.cat((x, y, fft, fft_bandpassed, envelope, envelope_spectrum), dim=2)
                        else:
                            if self.use_eofft:
                                dynamic_features = torch.cat((x, y, eofft, fft_bandpassed, envelope), dim=2)
                            else:
                                dynamic_features = torch.cat((x, y, fft, fft_bandpassed, envelope), dim=2)
                    else:
                        if self.use_envelope_spectrum:
                            if self.use_eofft:
                                dynamic_features = torch.cat((x, y, eofft,fft_bandpassed, envelope_spectrum), dim=2)
                            else:
                                dynamic_features = torch.cat((x, y, fft,fft_bandpassed, envelope_spectrum), dim=2)
                        else:
                            if self.use_eofft:
                                dynamic_features = torch.cat((x, y, eofft, fft_bandpassed), dim=2)
                            else:
                                dynamic_features = torch.cat((x, y, fft, fft_bandpassed), dim=2)
                else:
                    if self.use_envelope :
                        if self.use_envelope_spectrum:
                            if self.use_eofft:
                                dynamic_features = torch.cat((y, eofft,fft_bandpassed, envelope, envelope_spectrum), dim=2)
                            else:
                                dynamic_features = torch.cat((y, fft,fft_bandpassed, envelope, envelope_spectrum), dim=2)
                        else:
                            if self.use_eofft:
                                dynamic_features = torch.cat((y, eofft,fft_bandpassed, envelope), dim=2)
                            else:
                                dynamic_features = torch.cat((y, fft,fft_bandpassed, envelope), dim=2)
                    else:
                        if self.use_envelope_spectrum:
                                        if self.use_eofft:
                                            dynamic_features = torch.cat((y, eofft,fft_bandpassed, envelope_spectrum), dim=2)
                                        else:
                                            dynamic_features = torch.cat((y, fft,fft_bandpassed, envelope_spectrum), dim=2)
                        else:
                            if self.use_eofft:
                                dynamic_features = torch.cat((y, eofft,fft_bandpassed), dim=2)
                            else:
                                dynamic_features = torch.cat((y, fft,fft_bandpassed), dim=2)
            else:
                if self.use_raw:
                    if self.use_envelope :
                        if self.use_envelope_spectrum:
                            if self.use_eofft:
                                dynamic_features = torch.cat((x, eofft,fft_bandpassed, envelope, envelope_spectrum), dim=2)
                            else:
                                dynamic_features = torch.cat((x, fft,fft_bandpassed, envelope, envelope_spectrum), dim=2)
                        else:
                            if self.use_eofft:
                                dynamic_features = torch.cat((x, eofft,fft_bandpassed, envelope), dim=2)
                            else:
                                dynamic_features = torch.cat((x, fft,fft_bandpassed, envelope), dim=2)
                    else:
                        if self.use_envelope_spectrum:
                            if self.use_eofft:
                                dynamic_features = torch.cat((x, eofft,fft_bandpassed, envelope_spectrum), dim=2)
                            else:
                                dynamic_features = torch.cat((x, fft,fft_bandpassed, envelope_spectrum), dim=2)
                        else:
                            if self.use_eofft:
                                dynamic_features = torch.cat((x, eofft,fft_bandpassed), dim=2)
                            else:
                                dynamic_features = torch.cat((x, fft,fft_bandpassed), dim=2)

                else:
                    if self.use_envelope :
                        if self.use_envelope_spectrum:
                            if self.use_eofft:
                                dynamic_features = torch.cat((eofft,fft_bandpassed, envelope, envelope_spectrum), dim=2)
                            else:
                                dynamic_features = torch.cat((fft,fft_bandpassed, envelope, envelope_spectrum), dim=2)
                        else:
                            if self.use_eofft:
                                dynamic_features = torch.cat((eofft,fft_bandpassed, envelope), dim=2)
                            else:
                                dynamic_features = torch.cat((fft,fft_bandpassed, envelope), dim=2)
                    else:
                        if self.use_envelope_spectrum:
                            if self.use_eofft:
                                dynamic_features = torch.cat((eofft,fft_bandpassed, envelope_spectrum), dim=2)
                            else:
                                dynamic_features = torch.cat((fft,fft_bandpassed, envelope_spectrum), dim=2)
                        else:
                            if self.use_eofft:
                                dynamic_features = torch.cat((eofft,fft_bandpassed), dim=2)
                            else:
                                dynamic_features = torch.cat((fft,fft_bandpassed), dim=2)

        else:
            if self.use_bandpass:
                if self.use_raw:
                    if self.use_envelope :
                        if self.use_envelope_spectrum:
                            if self.use_eofft:
                                dynamic_features = torch.cat((x, y, eofft, envelope, envelope_spectrum), dim=2)
                            else:
                                dynamic_features = torch.cat((x, y, fft, envelope, envelope_spectrum), dim=2)
                        else:
                            if self.use_eofft:
                                dynamic_features = torch.cat((x, y, eofft, envelope), dim=2)
                            else:
                                dynamic_features = torch.cat((x, y, fft, envelope), dim=2)
                    else:
                        if self.use_envelope_spectrum:
                            if self.use_eofft:
                                dynamic_features = torch.cat((x, y, eofft, envelope_spectrum), dim=2)
                            else:
                                dynamic_features = torch.cat((x, y, fft, envelope_spectrum), dim=2)
                        else:
                            if self.use_eofft:
                                dynamic_features = torch.cat((x, y, eofft), dim=2)
                            else:
                                dynamic_features = torch.cat((x, y, fft), dim=2)
                else:
                    if self.use_envelope :
                        if self.use_envelope_spectrum:
                            if self.use_eofft:
                                dynamic_features = torch.cat((y, eofft, envelope, envelope_spectrum), dim=2)
                            else:
                                dynamic_features = torch.cat((y, fft, envelope, envelope_spectrum), dim=2)
                        else:
                            if self.use_eofft:
                                dynamic_features = torch.cat((y, eofft, envelope), dim=2)
                            else:
                                dynamic_features = torch.cat((y, fft, envelope), dim=2)
                    else:
                        if self.use_envelope_spectrum:
                            if self.use_eofft:
                                dynamic_features = torch.cat((y, eofft, envelope_spectrum), dim=2)
                            else:
                                dynamic_features = torch.cat((y, fft, envelope_spectrum), dim=2)
                        else:
                            if self.use_eofft:
                                dynamic_features = torch.cat((y, eofft), dim=2)
                            else:
                                dynamic_features = torch.cat((y, fft), dim=2)
            else:
                if self.use_raw:
                    if self.use_envelope :
                        if self.use_envelope_spectrum:
                            if self.use_eofft:
                                dynamic_features = torch.cat((x, eofft, envelope, envelope_spectrum), dim=2)
                            else:
                                dynamic_features = torch.cat((x, fft, envelope, envelope_spectrum), dim=2)
                        else:
                            if self.use_eofft:
                                dynamic_features = torch.cat((x, eofft, envelope), dim=2)
                            else:
                                dynamic_features = torch.cat((x, fft, envelope), dim=2)
                    else:
                        if self.use_envelope_spectrum:
                            if self.use_eofft:
                                dynamic_features = torch.cat((x, eofft , envelope_spectrum), dim=2)
                            else:
                                dynamic_features = torch.cat((x, fft, envelope_spectrum), dim=2)
                        else:
                            if self.use_eofft:
                                dynamic_features = torch.cat((x, eofft), dim=2)
                            else:
                                dynamic_features = torch.cat((x, fft), dim=2)
                else:
                    if self.use_envelope :
                        if self.use_envelope_spectrum:
                            if self.use_eofft:
                                dynamic_features = torch.cat((eofft, envelope, envelope_spectrum), dim=2)
                            else:
                                dynamic_features = torch.cat((fft, envelope, envelope_spectrum), dim=2)
                        else:
                            if self.use_eofft:
                                dynamic_features = torch.cat((eofft, envelope), dim=2)
                            else:
                                dynamic_features = torch.cat((fft, envelope), dim=2)

                    else:
                        if self.use_envelope_spectrum:
                            if self.use_eofft:
                                dynamic_features = torch.cat((eofft, envelope_spectrum), dim=2)
                            else:
                                dynamic_features = torch.cat((fft, envelope_spectrum), dim=2)
                        else:
                            if self.use_eofft:
                                dynamic_features = eofft
                            else:
                                dynamic_features = fft


        dynamic_features = dynamic_features.transpose(1,2)
        z = self.silu(self.batchnorm1(self.conv1(dynamic_features)))
        z = self.silu(self.batchnorm2(self.conv2(z)))
        z = self.maxpool(z)
        z = self.silu(self.batchnorm3(self.conv3(z)))
        z = self.silu(self.batchnorm4(self.conv4(z)))
        z = self.maxpool(z)
        z, _ = self.lstm(z)
        z = z[:, -1, :]  # Take the last time step
        if self.use_fft_stat:
            if self.use_eofft:
                eofft_hs = self.fft_hs(eofft)
                fft_hs = self.fft_hs(fft)
            z = torch.cat((z, fft_hs, eofft_hs), dim=-1) 
        if self.use_stat:
            hs = self.hs(x)
            z = torch.cat((z,hs), dim=-1) 
        z = self.layer_norm1(self.dense1(z))

        return z


    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)
        
            
if __name__ == "__main__":
    _ = CONV_LSTM_Classifier()