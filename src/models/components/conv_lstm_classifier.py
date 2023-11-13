import torch
import torch.fft as fft
import torch.nn as nn

class FFTReal(nn.Module):
    def forward(self, inputs, dim):
        inputs = inputs.type(torch.complex64)
        fft_result = fft.fft(inputs, dim = dim)
        return torch.real(fft_result)

    
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
        self.dropout  = nn.Dropout(0.5)
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
        z = self.silu(self.batchnorm1(self.dropout(self.conv1(dynamic_features))))
        z = self.silu(self.batchnorm2(self.conv2(z)))
        z = self.maxpool(z)
        z = self.silu(self.batchnorm3(self.dropout(self.conv3(z))))
        z = self.silu(self.batchnorm4(self.conv4(z)))
        z = self.maxpool(z)
        z, _ = self.lstm(z)
        z = z[:, -1, :]  # Take the last time step
        z = self.layer_norm1(self.dropout(self.dense1(z)))
        z = self.dense3(self.dropout(self.dense2(z)))
        return z

    def _forward_with_raw_filter_only(self, x):
        # FFT and Real components
        y = x[:,:,1].unsqueeze(-1)
        x = x[:,:,0].unsqueeze(-1)
        r_original = self.fft_real(x, 1)
        dynamic_features = torch.cat((r_original, x, y), dim=2)
        dynamic_features = dynamic_features.transpose(1,2)
        # Apply layers
        z = self.silu(self.batchnorm1(self.dropout(self.conv1(dynamic_features))))
        z = self.silu(self.batchnorm2(self.conv2(z)))
        z = self.maxpool(z)
        z = self.silu(self.batchnorm3(self.dropout(self.conv3(z))))
        z = self.silu(self.batchnorm4(self.conv4(z)))
        z = self.maxpool(z)
        z, _ = self.lstm(z)
        z = z[:, -1, :]  # Take the last time step
        z = self.layer_norm1(self.dropout(self.dense1(z)))
        z = self.dense3(self.dropout(self.dense2(z)))
        return z
    
    def _forward_without_filters(self, x):
        # FFT and Real components
        y = x[:,:,1].unsqueeze(-1)
        x = x[:,:,0].unsqueeze(-1)
        r_original = self.fft_real(x, 1)
        dynamic_features = torch.cat((r_original, x), dim=2)
        dynamic_features = dynamic_features.transpose(1,2)
        # Apply layers
        z = self.silu(self.batchnorm1(self.dropout(self.conv1(dynamic_features))))
        z = self.silu(self.batchnorm2(self.conv2(z)))
        z = self.maxpool(z)
        z = self.silu(self.batchnorm3(self.dropout(self.conv3(z))))
        z = self.silu(self.batchnorm4(self.conv4(z)))
        z = self.maxpool(z)
        z, _ = self.lstm(z)
        z = z[:, -1, :]  # Take the last time step
        z = self.layer_norm1(self.dropout(self.dense1(z)))
        z = self.dense3(self.dropout(self.dense2(z)))
        return z

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)
        
            
if __name__ == "__main__":
    _ = CONV_LSTM_Classifier()