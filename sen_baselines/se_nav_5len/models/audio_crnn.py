import numpy as np
import torch
import torch.nn as nn

from sen_baselines.se_nav.models.seldnet import MultiHeadAttentionLayer, ConvBlock


class AudioCRNN(nn.Module):
    def __init__(self, observation_space, output_size, audiogoal_sensor) -> None:
        super().__init__()
        # dimension of input data is [batch, frames, mel_bins, channels]
        # dimension of audio sensor is [frames, mel_bins, channels]
        self._n_input_audio = observation_space.spaces[audiogoal_sensor].shape[2]
        self._audiogoal_sensor = audiogoal_sensor
        self.f_pool_size = [4, 4, 2]
        self.t_pool_size = [5, 1, 1]

        self.cnn = nn.Sequential()
        for conv_cnt in range(len(self.f_pool_size)):
            self.cnn.extend([
                ConvBlock(
                    in_channels=64 if conv_cnt else self._n_input_audio,
                    out_channels=64,
                ),
                nn.MaxPool2d((
                    self.t_pool_size[conv_cnt],
                    self.f_pool_size[conv_cnt]
                )),
                nn.Dropout2d(
                    p=0.05,
                )
            ])

        self.rnn = nn.GRU(
            input_size=64 * int(np.floor(
                observation_space.spaces[audiogoal_sensor].shape[1] / np.prod(self.f_pool_size)
            )),
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.05,
            bidirectional=True,
        )

        self.attn = MultiHeadAttentionLayer(
            hidden_size=128,
            n_heads=4,
            dropout=0.05,
        )
        # the dimension after GRU layer is [batch, frames, 128]
        
        self.linear1 = nn.Linear(128, 128, bias=True)
        self.linear2 = nn.Linear(
            128 * int(np.floor(
                observation_space.spaces[audiogoal_sensor].shape[0] / np.prod(self.t_pool_size)
            )), 
            output_size, 
            bias=True
        )
        # the dimension after linear2 layer is [batch, output_size]

    def forward(self, observations):
        input = []
        audio_observations = observations[self._audiogoal_sensor]
        # [batch, frames, mel_bins, channels]
        audio_observations = audio_observations.permute(0, 3, 1, 2)
        # [batch, channels, frames, mel_bins]
        input.append(audio_observations)
        
        x = torch.cat(input, dim=1)
        x = self.cnn(x)

        x = x.transpose(1, 2).contiguous()
        # [batch_size, feature_map, time_steps, mel_bins]
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        # [batch_size, time_steps, feature_map * mel_bins]
        (x, _) = self.rnn(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]
        
        x = self.attn.forward(x, x, x)
        x = torch.tanh(x)

        x = self.linear1(x)
        x = torch.flatten(x, 1)
        x = self.linear2(x)

        return x


# def conv_output_dim(in_dim, kernel_size_c, stride_c, padding_c, kernel_size_p, stride_p):
#     out_dim = in_dim
#     for i in range(len(kernel_size_c)):
#         out_dim = [
#             (out_dim[0] + 2 * padding_c[i] - kernel_size_c[i]) // stride_c[i] + 1,
#             (out_dim[1] + 2 * padding_c[i] - kernel_size_c[i]) // stride_c[i] + 1]
#         out_dim = [
#             (out_dim[0] - kernel_size_p[i]) // stride_p[i] + 1,
#             (out_dim[1] - kernel_size_p[i]) // stride_p[i] + 1
#         ]
#     return out_dim
        