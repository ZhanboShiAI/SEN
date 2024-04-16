import torch
import torch.nn as nn

'''
猜测:
训练方法是先执行step, 根据每一步的计算执行小批量的更新
一定step之后进行大批量的序列学习, 使用之前计算的特征序列进行更新
yaml文件中PPO下的num_steps表示decide the length of history that ppo encodes
与上述猜测吻合
'''

# 强化学习的状态编码器
class RNNStateEncoder(nn.Module):
    def __init__(
        self, 
        input_size: int,
        hidden_size: int,
        num_layers:int = 1,
        rnn_type: str = "GRU", 
    ):
        super().__init__()
        self._num_recurrent_layers = num_layers
        self._rnn_type = rnn_type
        
        # 获得名字属性, 相当于 nn.rnn_type
        # 使用默认参数GRU, 这里相当于是使用 nn.GRU() 对rnn网络进行创建
        # 类似于 self.cnn = nn.Sequential(), 这里本质是 self.rnn = nn.GRU或者self.rnn = nn.LSTM
        self.rnn = getattr(nn, rnn_type)(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers)
        
        self.layer_init()
        
    def layer_init(self):
        if isinstance(self.rnn, (nn.GRU, nn.LSTM)):
            for name, param in self.rnn.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0)
        else:
            print("**************************")
            print("RNNStateEncoder Init Error")
            
    @property
    def num_recurrent_layers(self):
        return self._num_recurrent_layers * (2 if "LSTM" in self._rnn_type else 1)
    
    def _pack_hidden(self, hidden_states):
        # 一层LSTM有2层, 将两层的参数串联起来
        # LSTM有两层, 细胞状态和隐藏状态. 隐藏状态是网络每个时刻的输出, 细胞状态是网络的内部记忆
        if "LSTM" in self._rnn_type:
            hidden_states = torch.cat(
                [hidden_states[0], hidden_states[1]], dim=0
            )
        return hidden_states
    
    def _unpack_hidden(self, hidden_states):
        if "LSTM" in self._rnn_type:
            hidden_states = (
                hidden_states[0 : self.num_recurrent_layers],
                hidden_states[self.num_recurrent_layers :],
            )
        return hidden_states
    
    # hidden_states的维度是[1, 2, 512], masks的维度是 [1, 2, 1]
    # masks被扩展到 [1, 2, 512], 其中的0元被复制, 这表示有0元的那个process的不会有输出
    def _mask_hiddne(self, hidden_states, masks: torch.Tensor):
        # 当使用LSTM时, 解包后的hidden_states是一个包含两个元素的元组, 见unpack_hiddne()
        if isinstance(hidden_states, tuple):
            hidden_states = tuple(v * masks for v in hidden_states)
        else:
            hidden_states = masks * hidden_states
        return hidden_states
    
    # x: [2, 1026], hidden_states: [1, 2, 512]
    def single_forward(self, x: torch.Tensor, hidden_states, masks: torch.Tensor):
        hidden_states = self._unpack_hidden(hidden_states)
        # unsqueeze(0)在tensor的最前面添加一个维度, 形成新的维度(1, n), n表示tensor原来的维度
        # 这个方法常用于向tensor中添加一个维度表示batch, 因为torch的网络结构默认第一个维度表示batch size
        # 显然single_forward中没有batch的信息, 所以要向x和hidden_states中添加一个维度
        x, hidden_states = self.rnn(
            x.unsqueeze(0),
            self._mask_hiddne(hidden_states, masks.unsqueeze(0))
        )
        
        x = x.squeeze(0)
        hidden_states = self._pack_hidden(hidden_states)
        
        return x, hidden_states
    
    # 长度为T的sequence forward. 其中 x 是一个(T * N, -1)的tensor, 是从(T, N, -1)展平得到的. 
    # 其中T表示时间步, N表示batch size. RNN序列学习通常用于处理具有多个时间步的输入数据. 
    # 每个时间步都包含一个观测值. 这些时间步长可以是文本中的字, 或者语音信号中的帧. 
    # 在VectorEnv中, 每一个环境的step和reset方法上都是同步的. 可以理解为batch size N就是VectorEnv中同时运行的环境的数量
    # 经过输出测试, batch size 是yaml文件中配置的并行环境的数量, 即 num_processes
    # -1表示自动计算剩余的维度. 使用-1作为tensor形状的参数时, 表示自动计算, 使整个张量的总元素保持不变
    # masks是展平为(T * N)的tensor. 
    # 这个方法用于处理具有间断0值的masks的情况, 并在每个连续的非0掩码块上进行RNN操作. 用于提升速度
    def seq_forward(self, x: torch.Tensor, hidden_states, masks: torch.Tensor):
        '''
        x: (T * N, -1), T表示序列长度, N表示batch size, 与yaml中num_processes相同, 为VectorEnv中并行的环境数量
        实际为 (150 * 2, 1026), 这里的150是yaml中PPO中的num_steps, 用于表示ppo编码的历史长度
        
        hidden_states: (1, N, -1), N表示batch size, 实际为 (1, 2, 512)
        
        masks: (T * N), 那么应该为(150 * 2)
        
        '''
        n = hidden_states.size(1)   # n=2
        # print("******************")
        # print("Number of Batch ", n)
        t = int(x.size(0) / n)      # t=150
        
        # unflatten
        x = x.view(t, n, x.size(1)) # x [150, 2, 1026]
        masks = masks.view(t, n)    # masks [150, 2]
        
        # 在masks张量中寻找具有0值的时间步. 假设t=0时包含0值, 在下一步中进行矫正
        # masks = torch.tensor([
        #     [1.0, 1.0],
        #     [0.0, 1.0],
        #     [1.0, 0.0],
        #     [1.0, 1.0]
        # ])
        # 经过masks[1:]后为torch.tensor([
        #     [0.0, 1.0],
        #     [1.0, 0.0],
        #     [1.0, 1.0]
        # ])
        # 经过(masks[1:] == 0.0)为tensor([
        #   [ True, False],
        #   [False,  True],
        #   [False, False]])
        # 经过.any(dim=-1)为tensor([ True,  True,  False]), 
        # .any(dim=-1)表示在最后一维上是否有任意一个元素为True, 如果是, 则返回True
        # 也就是对于每一个时间步, 沿batch size N, 查询在这个时间步上每个环境中是否有0masks
        # 对于 [T, N]的masks, 这表示在哪一个环境上掩码含有0元, 即序列具有填充元素. 
        # 经过.nonzero()为 tensor([0], [1])
        # 经过.squeeze()为 tensor([0, 1])
        # 最后得到的是一个包含索引值的张量, 这些索引值对应masks中至少有一个位置为0的时间步
        # 注意, 这里的索引值是去掉了第0个时间步之后的索引值, 所以需要在下面的步骤中将索引值+1
        has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()
        
        # +1 to correct the masks[1:]
        # 如果has_zeros的维度是0, 表示其是一个标量张量, 使用item提取其值并+1, 并放入列表中
        if has_zeros.dim() == 0:
            has_zeros = [has_zeros.item() + 1]
        # 如果维度不是0, 表示其包含多个索引. 在这种情况下, 将每一个元素+1, 转化为numpy数组, 再转化为列表
        else:
            has_zeros = (has_zeros + 1).numpy().tolist()
            
        # add t=0 and t=T to the list
        # 在列表的开头添加一个值 0, 在列表的结尾添加一个值 t
        # 这样, has_zeros就是一个包含正确的索引值, 以及序列开头和结尾的完整列表
        has_zeros = [0] + has_zeros + [t]
        
        # print("******************")
        # print(has_zeros)
        
        hidden_states = self._unpack_hidden(hidden_states)
        outputs = []
        for i in range(len(has_zeros) - 1):
            # 先处理不包含0的部分, 这里就是依次找出不包含0元素的几行
            start_index = has_zeros[i]
            end_index = has_zeros[i + 1]
            
            # print("******************")
            # print(x[start_index:end_index])
            # print("******************")
            # print(masks[start_index].view(1, -1, 1))
            
            # x的维度是 [T, N, -1], 实际是 [150, 2, 1026], 切分之后x的维度是 [t, 2, 1026]
            # masks的维度是 [150, 2], 这里取start_index处的张量, 其形状为[2]
            # 再将其转化为形状为 [1, 2, 1]的张量
            '''这样切会导致每一段x都会有至少一个process的数据不会被用于输出, 可能会造成收敛速度变慢'''
            # rnn_scores, hidden_states = self.rnn(
            #     x[start_index:end_index],
            #     self._mask_hiddne(
            #         hidden_states, masks[int(start_index)].view(1, -1, 1)
            #     ),
            # )
            # outputs.append(rnn_scores)
            if end_index - start_index <= 1:
                rnn_scores, hidden_states = self.rnn(
                    x[start_index:end_index],
                    self._mask_hiddne(
                        hidden_states, masks[start_index].view(1, -1, 1)
                    ),
                )
                outputs.append(rnn_scores)
            # 如果两个masks中包含0元素的索引之间的距离较长, 先将包含0元的masks取出进行计算, 
            # 再计算后续不包含0元的masks
            else:
                rnn_scores, hidden_states = self.rnn(
                    x[start_index : start_index + 1],
                    self._mask_hiddne(
                        hidden_states, masks[start_index].view(1, -1, 1)
                    )
                )
                outputs.append(rnn_scores)
                rnn_scores, hidden_states = self.rnn(
                    x[start_index + 1 : end_index],
                    self._mask_hiddne(
                        hidden_states, masks[start_index + 1].view(1, -1, 1)
                    )
                )
                outputs.append(rnn_scores)
        
        # 重新连接输出, 形状为 (T, N, -1)
        x = torch.cat(outputs, dim=0)
        x = x.view(t * n, -1)   #flatten
        
        hidden_states = self._pack_hidden(hidden_states)
        return x, hidden_states
    
    # hidden_states的形状(T, N, -1), x的形状(T*N, -1)
    # 如果 N=T*N, 表示输入的single的, 否则作为序列进行处理
    # 这里batch的数量就是并行的环境数量. RNNStateEncoder应该是在VectorEnv的同步的step中调用的更新
    def forward(self, x, hidden_states, masks):
        # print("***********************")
        # print("RNNStateEncoder forward")
        # print("Shape of RNNStateEncoder ", x.shape)
        # torch.Size([2, 1026])
        # torch.Size([300, 1026])
        # 在每一个step下, x的维度是 [2, 1026], 表示在single情况下, batch size是2, 输入的大小是1026
        # 在sequence的情况下, x的维度是 [300, 1026], 表示在序列的情况下, 序列长度是150
        # print("Shape of HiddenStates ", hidden_states.shape)
        # torch.Size([1, 2, 512])
        # 无论在single还是seq情况下一样, 第一个维度应该是占位的?
        # 第二个维度是并行的环境数量, 第三个维度是隐层的节点数量
        # print("Shape of Masks ", masks.shape)
        
        if x.size(0) == hidden_states.size(1):
            return self.single_forward(x, hidden_states, masks)
        else:
            return self.seq_forward(x, hidden_states, masks)