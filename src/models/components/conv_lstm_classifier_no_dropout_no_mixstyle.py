import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

class FFTReal(nn.Module):
    def forward(self, inputs, dim):
        inputs = inputs.type(torch.complex64)
        fft_result = fft.fft(inputs, dim = dim)
        return torch.real(fft_result)


class FFT_Health_State_Analysis(nn.Module):
    def __init__(self):
        super(FFT_Health_State_Analysis, self).__init__()

    def forward(self, inputs):
        
        top_k_mean_freq = torch.mean(torch.topk(inputs, k=10, axis = 1).indices.float(),axis = 1)  
        top_k_rms = torch.sqrt(torch.mean(torch.topk(inputs, k=10, axis = 1).values**2, dim=1))
        
        max_freq = torch.Tensor(torch.topk(inputs, k=1, axis = 1).indices.float()).squeeze(-1) 
        max_rms = torch.sqrt(torch.mean(torch.topk(inputs, k=1, axis = 1).values**2, dim=1))

        top_k_freqs = torch.topk(inputs, k=50, axis = 1).indices.float() 
        stacked_stats = torch.stack([top_k_mean_freq,top_k_rms,max_freq,max_rms], dim=1)
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
        # inputs shape: (batch, signal_length, channel_size)
        # rpm shape: (batch,)
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
    def forward(self, x, channel_dim):
        eps = 0.0001
        B = x.size(0)
        alpha = torch.tensor([0.1])
        mu = x.mean(dim=[2], keepdim=True) # compute instance mean
        var = x.var(dim=[2], keepdim=True) # compute instance variance
        sig = (var + eps).sqrt() # compute instance standard deviation
        mu, sig = mu.detach(), sig.detach() # block gradients
        x_normed = (x - mu) / sig # normalize input
        lmda = torch.distributions.beta.Beta(alpha, alpha).sample((B, 1, 1)) # sample instance-wise convex weights
        perm = torch.randperm(B) # generate shuffling indices
        mu2, sig2 = mu[perm], sig[perm] # shuffling
        mu_mix = mu * lmda + mu2 * (1 - lmda) # generate mixed mean
        sig_mix = sig * lmda + sig2 * (1 - lmda) # generate mixed standard deviation
        return x_normed * sig_mix + mu_mix # denormalize input using the mixed statistics

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
        use_raw_bandpass_filterd: bool = True,
        use_fft_bandpass_filterd: bool = True,
        use_eofft: bool = True,
        use_fft_stat: bool = True,
    ):
        super(CONV_LSTM_Classifier, self).__init__()
        self.in_channels = 2 + int(use_raw_bandpass_filterd) + int(use_fft_bandpass_filterd)
        self.in_length = in_length
        self.fft_real = FFTReal()
        self.fft_hs = FFT_Health_State_Analysis()
        self.layer_norm1 = nn.LayerNorm(64)
        self.silu = nn.SiLU()
        self.dropout  = nn.Dropout(0.5)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.use_raw_bandpass_filterd = use_raw_bandpass_filterd
        self.use_fft_bandpass_filterd = use_fft_bandpass_filterd
        self.use_fft_stat = use_fft_stat
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
        self.dense1 = nn.Linear(lstm_hidden_size * 2 + 54*int(self.use_fft_stat), 64)  # Hidden size is doubled because LSTM is bidirectional
        # Dense layers
        self.dense2 = nn.Linear(64, 32)
        self.dense3 = nn.Linear(32, output_size)
        if use_eofft:
            if use_raw_bandpass_filterd:
                if use_fft_bandpass_filterd:
                    self.forward = self._forward_with_both_filters_eofft
                else:
                    self.forward = self._forward_with_raw_filter_only_eofft
            else:
                if use_fft_bandpass_filterd:
                    self.forward = self._forward_with_raw_fft_filters_eofft
                else:
                    self.forward = self._forward_without_filters_eofft
        else:
            if use_raw_bandpass_filterd:
                if use_fft_bandpass_filterd:
                    self.forward = self._forward_with_both_filters_fft
                else:
                    self.forward = self._forward_with_raw_filter_only_fft
            else:
                if use_fft_bandpass_filterd:
                    self.forward = self._forward_with_raw_fft_filters_fft
                else:
                    self.forward = self._forward_without_filters_fft
            


    def calculate_conv_output_size(self):
        # Dummy input for calculating the size of LSTM input
        # This assumes the input size to the first conv layer is (batch_size, channels, sequence_length)
        dummy_input = torch.zeros(1, self.in_channels, self.in_length)
        dummy_output = self.maxpool(self.batchnorm4(self.conv4(self.silu(self.batchnorm3(self.conv3(self.silu(self.maxpool(self.batchnorm2(self.conv2(self.silu(self.batchnorm1(self.conv1(dummy_input)))))))))))))
        return dummy_output.shape[-1]

    def _forward_with_both_filters_eofft(self, x):
        # FFT and Real components
        eofft = x[:,:,2].unsqueeze(-1)
        y = x[:,:,1].unsqueeze(-1)
        x = x[:,:,0].unsqueeze(-1)
        r_filtered = self.fft_real(y, 1)
        dynamic_features = torch.cat((eofft,r_filtered, x, y), dim=2)
        dynamic_features = dynamic_features.transpose(1,2)
        # Apply layers
        z = self.silu(self.batchnorm1(self.conv1(dynamic_features)))
        z = self.silu(self.batchnorm2(self.dropout(self.conv2(z))))
        z = self.maxpool(z)
        z = self.silu(self.batchnorm3(self.conv3(z)))
        z = self.silu(self.batchnorm4(self.dropout(self.conv4(z))))
        z = self.maxpool(z)
        z, _ = self.lstm(z)
        z = z[:, -1, :]  # Take the last time step
        if self.use_fft_stat:
            fft_hs = self.fft_hs(eofft)
            z = torch.cat((z,fft_hs), dim=-1) 
        z = self.layer_norm1(self.dropout(self.dense1(z)))
        z = self.dense3(self.dropout(self.dense2(z)))
        return z
    
    def _forward_with_both_filters_fft(self, x):
        # FFT and Real components
        eofft = x[:,:,2].unsqueeze(-1)
        y = x[:,:,1].unsqueeze(-1)
        x = x[:,:,0].unsqueeze(-1)
        fft = self.fft_real(x, 1)
        r_filtered = self.fft_real(y, 1)
        dynamic_features = torch.cat((fft, r_filtered, x, y), dim=2)
        dynamic_features = dynamic_features.transpose(1,2)
        # Apply layers
        z = self.silu(self.batchnorm1(self.conv1(dynamic_features)))
        z = self.silu(self.batchnorm2(self.dropout(self.conv2(z))))
        z = self.maxpool(z)
        z = self.silu(self.batchnorm3(self.conv3(z)))
        z = self.silu(self.batchnorm4(self.dropout(self.conv4(z))))
        z = self.maxpool(z)
        z, _ = self.lstm(z)
        z = z[:, -1, :]  # Take the last time step
        if self.use_fft_stat:
            fft_hs = self.fft_hs(fft)
            z = torch.cat((z,fft_hs), dim=-1) 
        z = self.layer_norm1(self.dropout(self.dense1(z)))
        z = self.dense3(self.dropout(self.dense2(z)))
        return z

    def _forward_with_raw_filter_only_eofft(self, x):
        # FFT and Real components
        print(x.shape)
        eofft = x[:,:,2].unsqueeze(-1)
        y = x[:,:,1].unsqueeze(-1)
        x = x[:,:,0].unsqueeze(-1)
        dynamic_features = torch.cat((eofft, x, y), dim=2)
        dynamic_features = dynamic_features.transpose(1,2)
        # Apply layers
        z = self.silu(self.batchnorm1(self.conv1(dynamic_features)))
        z = self.silu(self.batchnorm2(self.dropout(self.conv2(z))))
        z = self.maxpool(z)
        z = self.silu(self.batchnorm3(self.conv3(z)))
        z = self.silu(self.batchnorm4(self.dropout(self.conv4(z))))
        z = self.maxpool(z)
        z, _ = self.lstm(z)
        z = z[:, -1, :]  # Take the last time step
        if self.use_fft_stat:
            fft_hs = self.fft_hs(eofft)
            z = torch.cat((z,fft_hs), dim=-1) 
        z = self.layer_norm1(self.dropout(self.dense1(z)))
        z = self.dense3(self.dropout(self.dense2(z)))
        return z
    
    def _forward_with_raw_filter_only_fft(self, x):
        # FFT and Real components
        print(x.shape)
        eofft = x[:,:,2].unsqueeze(-1)
        y = x[:,:,1].unsqueeze(-1)
        x = x[:,:,0].unsqueeze(-1)
        fft = self.fft_real(x, 1)
        dynamic_features = torch.cat((fft, x, y), dim=2)
        dynamic_features = dynamic_features.transpose(1,2)
        # Apply layers
        z = self.silu(self.batchnorm1(self.conv1(dynamic_features)))
        z = self.silu(self.batchnorm2(self.dropout(self.conv2(z))))
        z = self.maxpool(z)
        z = self.silu(self.batchnorm3(self.conv3(z)))
        z = self.silu(self.batchnorm4(self.dropout(self.conv4(z))))
        z = self.maxpool(z)
        z, _ = self.lstm(z)
        z = z[:, -1, :]  # Take the last time step
        if self.use_fft_stat:
            fft_hs = self.fft_hs(fft)
            z = torch.cat((z,fft_hs), dim=-1) 
        z = self.layer_norm1(self.dropout(self.dense1(z)))
        z = self.dense3(self.dropout(self.dense2(z)))
        return z
    
    def _forward_with_raw_fft_filters_eofft(self, x):
        eofft = x[:,:,2].unsqueeze(-1)
        y = x[:,:,1].unsqueeze(-1)
        x = x[:,:,0].unsqueeze(-1)
        fft = self.fft_real(x, 1)
        r_filtered = self.fft_real(y, 1)
        dynamic_features = torch.cat((eofft,r_filtered, x), dim=2)
        dynamic_features = dynamic_features.transpose(1,2)
        # Apply layers
        z = self.silu(self.batchnorm1(self.conv1(dynamic_features)))
        z = self.silu(self.batchnorm2(self.dropout(self.conv2(z))))
        z = self.maxpool(z)
        z = self.silu(self.batchnorm3(self.conv3(z)))
        z = self.silu(self.batchnorm4(self.dropout(self.conv4(z))))
        z = self.maxpool(z)
        z, _ = self.lstm(z)
        z = z[:, -1, :]  # Take the last time step
        if self.use_fft_stat:
            fft_hs = self.fft_hs(eofft)
            z = torch.cat((z,fft_hs), dim=-1) 
        z = self.layer_norm1(self.dropout(self.dense1(z)))
        z = self.dense3(self.dropout(self.dense2(z)))
        return z
    
    def _forward_with_raw_fft_filters_fft(self, x):
        eofft = x[:,:,2].unsqueeze(-1)
        y = x[:,:,1].unsqueeze(-1)
        x = x[:,:,0].unsqueeze(-1)
        fft = self.fft_real(x, 1)
        r_filtered = self.fft_real(y, 1)
        dynamic_features = torch.cat((fft,r_filtered, x), dim=2)
        dynamic_features = dynamic_features.transpose(1,2)
        # Apply layers
        z = self.silu(self.batchnorm1(self.conv1(dynamic_features)))
        z = self.silu(self.batchnorm2(self.dropout(self.conv2(z))))
        z = self.maxpool(z)
        z = self.silu(self.batchnorm3(self.conv3(z)))
        z = self.silu(self.batchnorm4(self.dropout(self.conv4(z))))
        z = self.maxpool(z)
        z, _ = self.lstm(z)
        z = z[:, -1, :]  # Take the last time step
        if self.use_fft_stat:
            fft_hs = self.fft_hs(eofft)
            z = torch.cat((z,fft_hs), dim=-1) 
        if self.use_fft_stat:
            fft_hs = self.fft_hs(fft)
            z = torch.cat((z,fft_hs), dim=-1) 
        z = self.layer_norm1(self.dropout(self.dense1(z)))
        z = self.dense3(self.dropout(self.dense2(z)))
        return z
    
    def _forward_without_filters_eofft(self, x):
        # FFT and Real components
        eofft = x[:,:,2].unsqueeze(-1)
        x = x[:,:,0].unsqueeze(-1)
        dynamic_features = torch.cat((eofft, x), dim=2)
        dynamic_features = dynamic_features.transpose(1,2)
        # Apply layers
        z = self.silu(self.batchnorm1(self.conv1(dynamic_features)))
        z = self.silu(self.batchnorm2(self.dropout(self.conv2(z))))
        z = self.maxpool(z)
        z = self.silu(self.batchnorm3(self.conv3(z)))
        z = self.silu(self.batchnorm4(self.dropout(self.conv4(z))))
        z = self.maxpool(z)
        z, _ = self.lstm(z)
        z = z[:, -1, :]  # Take the last time step
        if self.use_fft_stat:
            fft_hs = self.fft_hs(eofft)
            z = torch.cat((z,fft_hs), dim=-1) 
        z = self.layer_norm1(self.dropout(self.dense1(z)))
        z = self.dense3(self.dropout(self.dense2(z)))
        return z
    def _forward_without_filters_fft(self, x):
        # FFT and Real components
        eofft = x[:,:,2].unsqueeze(-1)
        x = x[:,:,0].unsqueeze(-1)
        fft = self.fft_real(x, 1)
        dynamic_features = torch.cat((fft, x), dim=2)
        dynamic_features = dynamic_features.transpose(1,2)
        # Apply layers
        z = self.silu(self.batchnorm1(self.conv1(dynamic_features)))
        z = self.silu(self.batchnorm2(self.dropout(self.conv2(z))))
        z = self.maxpool(z)
        z = self.silu(self.batchnorm3(self.conv3(z)))
        z = self.silu(self.batchnorm4(self.dropout(self.conv4(z))))
        z = self.maxpool(z)
        z, _ = self.lstm(z)
        z = z[:, -1, :]  # Take the last time step
        if self.use_fft_stat:
            fft_hs = self.fft_hs(fft)
            z = torch.cat((z,fft_hs), dim=-1) 
        z = self.layer_norm1(self.dropout(self.dense1(z)))
        z = self.dense3(self.dropout(self.dense2(z)))
        return z

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)
        
            
if __name__ == "__main__":
    _ = CONV_LSTM_Classifier()