import numpy as np
import torch
# torch.backends.cudnn.benchmark = False
import torch.nn as nn

from ss_baselines.common.utils import Flatten
from sen_baselines.av_nav.models.visual_cnn import conv_output_dim, layer_init

class AudioCNN(nn.Module):
    def __init__(self, observation_space, output_size, audiogoal_sensor) -> None:
        super().__init__()
        # 这里按照四通道处理
        self._number_input_audio = observation_space.spaces[audiogoal_sensor].shape[2]
        self._audiogoal_sensor = audiogoal_sensor

        cnn_dimensions = np.array(
            observation_space.spaces[audiogoal_sensor].shape[:2], dtype=np.int32
        )

        if cnn_dimensions[0] < 30 or cnn_dimensions[1] < 30:
            self._cnn_layers_kernel_size = [(5, 5), (3, 3), (3, 3)]
            self._cnn_layers_stride = [(2, 2), (2, 2), (1, 1)]
        else:
            self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]
            self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]

        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dimensions = conv_output_dim(
                dimension=cnn_dimensions,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )
        # print("+++++++++++CNN DIMENSIONS++++++++++++ ")
        # print(cnn_dimensions)
            
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=self._number_input_audio,
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[0],
                stride=self._cnn_layers_stride[0],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[1],
                stride=self._cnn_layers_stride[1],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[2],
                stride=self._cnn_layers_stride[2],
            ),
            Flatten(),
            nn.Linear(64 * cnn_dimensions[0] * cnn_dimensions[1], output_size),
            nn.ReLU(True),
        )
    
        layer_init(self.cnn)

    def forward(self, observations):
        cnn_input = []
        audio_observations = observations[self._audiogoal_sensor]
        # print("AUDIO_CNN audio_observations shape: ", audio_observations.shape)
        # torch.Size([10, 65, 76, 4])
        # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
        audio_observations = audio_observations.permute(0, 3, 1, 2)
        # print("AUDIO_CNN audio_observation permuted shape: ", audio_observations.shape)
        # torch.Size([10, 4, 65, 76])
        cnn_input.append(audio_observations)
        
        cnn_input = torch.cat(cnn_input, dim=1)

        # print("******************")
        # print("Audio CNN forward")
        # print("Shape of Observation ", cnn_input.shape)
        return self.cnn(cnn_input)