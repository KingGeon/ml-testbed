import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

class FFTReal(nn.Module):
    def forward(self, inputs, dim):
        inputs = inputs.type(torch.complex64)
        fft_result = fft.fft(inputs, dim = dim)
        return torch.real(fft_result)
    
class EngineOrderFFT(nn.Module):
    def __init__(self):
        super(EngineOrderFFT, self).__init__()

    def engine_order_fft(self, signal, rpm, sf=8192, res=20, ts=0.1):
        # signal shape: (batch, signal_length, channel_size)
        batch_size, signal_length, _ = signal.shape

        # Truncate the signal
        truncated_signal = signal[:, :int(sf*ts), :]

        # Calculate the padding length for each item in the batch
        pad_lengths = (res * 60 / rpm - ts) * sf
        max_pad_length = torch.max(pad_lengths).long()  # Find the maximum padding length

        # Apply padding to the whole batch
        padded_signal = F.pad(truncated_signal, (0, max_pad_length))

        # Remove extra padding for each item in the batch
        padded_signals = [padded_signal[i, :, :signal_length + int(pad_lengths[i])] for i in range(batch_size)]

        # Stack the padded signals and perform FFT
        stacked_signals = torch.stack(padded_signals, dim=0)
        fft_result = torch.fft.fft(stacked_signals, dim=1)

        return torch.abs(fft_result)[:, :sf, :]

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
        use_raw_bandpass_filterd = True,
        use_fft_bandpass_filterd = True,
    ):
        super(CONV_LSTM_Classifier, self).__init__()

        self.in_channels = 2 + int(use_raw_bandpass_filterd) + int(use_fft_bandpass_filterd)
        self.in_length = in_length
        self.fft_real = FFTReal()
        self.layer_norm1 = nn.LayerNorm(64)
        self.silu = nn.SiLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.use_raw_bandpass_filterd = use_raw_bandpass_filterd
        self.use_fft_bandpass_filterd = use_fft_bandpass_filterd
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
        self.dense1 = nn.Linear(lstm_hidden_size * 2, 64)  # Hidden size is doubled because LSTM is bidirectional
        # Dense layers
        self.dense2 = nn.Linear(64, 32)
        self.dense3 = nn.Linear(32, output_size)

        if use_raw_bandpass_filterd:
            if use_fft_bandpass_filterd:
                self.forward = self._forward_with_both_filters
            else:
                self.forward = self._forward_with_raw_filter_only
        else:
            self.forward = self._forward_without_filters

    def calculate_conv_output_size(self):
        # Dummy input for calculating the size of LSTM input
        # This assumes the input size to the first conv layer is (batch_size, channels, sequence_length)
        dummy_input = torch.zeros(1, self.in_channels, self.in_length)
        dummy_output = self.maxpool(self.batchnorm4(self.conv4(self.silu(self.batchnorm3(self.conv3(self.silu(self.maxpool(self.batchnorm2(self.conv2(self.silu(self.batchnorm1(self.conv1(dummy_input)))))))))))))
        return dummy_output.shape[-1]

    def _forward_with_both_filters(self, x):
        # FFT and Real components
        y = x[:,:,1].unsqueeze(-1)
        x = x[:,:,0].unsqueeze(-1)
        r_original = self.fft_real(x, 1)
        r_filtered = self.fft_real(y, 1)
        dynamic_features = torch.cat((r_original,r_filtered, x, y), dim=2)
        dynamic_features = dynamic_features.transpose(1,2)
        # Apply layers
        z = self.silu(self.batchnorm1(self.conv1(dynamic_features)))
        z = self.silu(self.batchnorm2(self.conv2(z)))
        z = self.maxpool(z)
        z = self.silu(self.batchnorm3(self.conv3(z)))
        z = self.silu(self.batchnorm4(self.conv4(z)))
        z = self.maxpool(z)
        z, _ = self.lstm(z)
        z = z[:, -1, :]  # Take the last time step
        z = self.layer_norm1(self.dense1(z))
        z = self.dense3(self.dense2(z))
        return z

    def _forward_with_raw_filter_only(self, x):
        # FFT and Real components
        y = x[:,:,1].unsqueeze(-1)
        x = x[:,:,0].unsqueeze(-1)
        r_original = self.fft_real(x)
        dynamic_features = torch.cat((r_original, x, y), dim=2)
        dynamic_features = dynamic_features.transpose(1,2)
        # Apply layers
        z = self.silu(self.batchnorm1(self.conv1(dynamic_features)))
        z = self.silu(self.batchnorm2(self.conv2(z)))
        z = self.maxpool(z)
        z = self.silu(self.batchnorm3(self.conv3(z)))
        z = self.silu(self.batchnorm4(self.conv4(z)))
        z = self.maxpool(z)
        z, _ = self.lstm(z)
        z = z[:, -1, :]  # Take the last time step
        z = self.layer_norm1(self.dense1(z))
        z = self.dense3(self.dense2(z))
        return z
    
    def _forward_without_filters(self, x):
        # FFT and Real components
        y = x[:,:,1].unsqueeze(-1)
        x = x[:,:,0].unsqueeze(-1)
        r_original = self.fft_real(x)
        dynamic_features = torch.cat((r_original, x), dim=2)
        dynamic_features = dynamic_features.transpose(1,2)
        # Apply layers
        z = self.silu(self.batchnorm1(self.conv1(dynamic_features)))
        z = self.silu(self.batchnorm2(self.conv2(z)))
        z = self.maxpool(z)
        z = self.silu(self.batchnorm3(self.conv3(z)))
        z = self.silu(self.batchnorm4(self.conv4(z)))
        z = self.maxpool(z)
        z, _ = self.lstm(z)
        z = z[:, -1, :]  # Take the last time step
        z = self.layer_norm1(self.dense1(z))
        z = self.dense3(self.dense2(z))
        return z

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)
        
            
if __name__ == "__main__":
    _ = CONV_LSTM_Classifier()