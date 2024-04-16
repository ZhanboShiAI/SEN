import numpy as np
import torch
import torch.nn as nn

# from gym import spaces

# 用于将高维张量展平为二维张量, 大小是 batch_size*(自动计算大小)
from ss_baselines.common.utils import Flatten

# 计算经过CNN卷积之后的输出维度的大小
# 卷积的计算过程如下:
# 1. 计算输入图像经过填充之后的大小: padded_height = input_height + 2*padding[0] = 256
# 2. 计算卷积核在每个位置上感受野的大小. 感受野是指卷积核能够看到的输入图像区域的大小. 对于大小为(k_h, k_w)的卷积核:
#    receptive_field_height = (k_h - 1) * dilation[0] + 1 = 8
#    计算感受野时, 考虑了卷积核的膨胀率. 这里没有膨胀
# 3. 计算特征图的大小: out_put_height = (padded_height - receptive_field_height) / stride[0] + 1
#    即第一次卷积之后的输出大小是31*31
def conv_output_dim(dimension, padding, dilation, kernel_size, stride):
    # dimension=2表示输入的图像的 宽x高 两个维度. 实际中使用的各种参数都是对称的, 宽和高的计算结果相同
    assert len(dimension) == 2, "VISUAL CNN INPUT DIMENSION ERROR"
    output_dimension = []
    for i in range(len(dimension)):
        output_dimension.append(
            int(
                np.floor(
                    ((dimension[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i]) + 1
                )
            )
        )
    return tuple(output_dimension)

def layer_init(cnn):
    for layer in cnn:
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
            if layer.bias is not None:
                nn.init.constant_(layer.bias, val=0)

# 具有3个卷积层和一个全连接层的CNN
# 接受observations, 输出对应的RGB或Depth的结果
class VisualCNN(nn.Module):
    # observation_space是habitat.VectorEnv或ThreadedVectorEnv类下的observation_spaces中的一个
    # observation_spaces 是 List[spaces.Dict] 
    # 所以这里传入的其实是一个 spaces.Dict, 经过输出为 <class 'gym.spaces.dict.Dict'>
    # 为了取消编译器报错, 这里不再指定输入的数据类型, 即 observation_space : spaces.Dict
    def __init__(self, observation_space, output_size, extra_rgb) -> None:
        super().__init__()
        
        # 这里应该是编译器的问题, 使用VectorEnv创建env并传入observation_spaces[0], 可以传入
        # observation_space.spaces["rgb"] 这里的rgb相当于是传感器的uuid
        # 这个函数的三个维度分别是 [H, W, C_in], 表示高, 宽和通道数. 
        # 其顺序与136行 rgb_observations.permute(0, 3, 1, 2)的结果相符
        if "rgb" in observation_space.spaces and not extra_rgb:
            self._number_input_rgb = observation_space.spaces["rgb"].shape[2]
        else:
            self._number_input_rgb = 0
            
        if "depth" in observation_space.spaces:
            # print("++++++++++++++VISUAL CNN++++++++++++++")
            # print("++++++++++OBSERVATION SPACE++++++++++")
            # print(type(observation_space))
            # print(type(observation_space.spaces))
            # print(observation_space.spaces["depth"].shape)
            # <class 'gym.spaces.dict.Dict'>
            # <class 'collections.OrderedDict'>
            # (128, 128, 1)
            self._number_input_depth = observation_space.spaces["depth"].shape[2]
        else:
            self._number_input_depth = 0
            
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]
        self._cnn_layers_stride = [(4, 4), (2, 2), (2, 2)]
        
        cnn_dimensions = [0, 0]
        if self._number_input_rgb > 0:
            cnn_dimensions = np.array(
                observation_space.spaces["rgb"].shape[:2], dtype=np.float32
            )
        elif self._number_input_depth > 0:
            cnn_dimensions = np.array(
                observation_space.spaces["depth"].shape[:2], dtype=np.float32
            )
            
        if self.is_blind:
            self.cnn = nn.Sequential()
        else:
            for kernel_size, stride in zip(
                self._cnn_layers_kernel_size, self._cnn_layers_stride
            ):
                # 这里不能直接分成cnn_input_dimensions和cnn_output_dimensions
                # 因为conv_output_dim这个函数是一个类似递归的函数, 下一层的输入是上一层的输出
                # 这里就维持这种写法即可
                cnn_dimensions = conv_output_dim(
                    dimension=cnn_dimensions,
                    padding=np.array([0, 0], dtype=np.float32),
                    dilation=np.array([1, 1], dtype=np.float32),
                    kernel_size=np.array(kernel_size, dtype=np.float32),
                    stride=np.array(stride, dtype=np.float32)
                )
            
            # torch中许多网络结构的默认输入形状是 [B, C_in, H, W], 其中B是batch size, 
            # C_in是输入channel的数量, H和W分别是高和宽 (多用于图像数据)
            # 所以下面的网络结构中隐含batch size 这个维度, 且默认为第一个维度
            self.cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=self._number_input_depth + self._number_input_rgb,
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
                # batch size是隐含的第一个维度, 这里就是将多维数据展平为二维
                Flatten(),
                # 线性层的参数是输入和输出的数量, batch size 同样是隐含的第一个维度
                nn.Linear(64 * cnn_dimensions[0] * cnn_dimensions[1], output_size),
                nn.ReLU(True),
            )
        layer_init(self.cnn)
        
    @property
    def is_blind(self):
        return self._number_input_depth + self._number_input_rgb == 0
    
    # observations的类型是<class 'dict'>
    # <<个forward函数应该在env.step()中调用的>>
    def forward(self, observations : dict):
        cnn_input = []
        if self._number_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # 将张量转化为torch网络结构的默认格式, 即 [B, C_in, H, W]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0
            cnn_input.append(rgb_observations)
            
        if self._number_input_depth > 0:
            depth_observations = observations["depth"]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            cnn_input.append(depth_observations)
            
        # 将两个输入在C_in这个维度上连接起来, 其输出维度如98行中Conv2d层的输入维度相同
        # in_channels=self._number_input_depth + self._number_input_rgb,
        cnn_input = torch.cat(cnn_input, dim=1)
        
        # print("******************")
        # print("Visual CNN forward")
        # print("Shape of Observation ", cnn_input.shape)
        return self.cnn(cnn_input)
