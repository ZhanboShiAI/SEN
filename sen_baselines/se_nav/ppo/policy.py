import abc

import torch
import torch.nn as nn

from ss_baselines.common.utils import CategoricalNet
from sen_baselines.se_nav.models.audio_crnn import AudioCRNN
from ss_baselines.av_nav.models.visual_cnn import VisualCNN
from ss_baselines.av_nav.ppo.policy import Net
from ss_baselines.av_nav.models.rnn_state_encoder import RNNStateEncoder

DUAL_GOAL_DELIMITER = ','


class SenBaselineNet(Net):
    def __init__(self, observation_space, hidden_size, goal_sensor_uuid, extra_rgb=False) -> None:
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self._audiogoal = False
        self._pointgoal = False
        self._number_pointgoal = 0

        if DUAL_GOAL_DELIMITER in self.goal_sensor_uuid:
            goal_1_uuid, goal_2_uuid = self.goal_sensor_uuid.split(DUAL_GOAL_DELIMITER)
            self._audiogoal = self._pointgoal = True
            self._number_pointgoal = observation_space.spaces[goal_1_uuid].shape[0]
        else:
            if 'pointgoal_with_gps_compass' == self.goal_sensor_uuid:
                self._pointgoal = True
                self._number_pointgoal = observation_space.spaces[self.goal_sensor_uuid].shape[0]
            else:
                self._audiogoal = True

        self.visual_encoder = VisualCNN(observation_space, hidden_size, extra_rgb)
        audiogoal_sensor = ''
        if self._audiogoal:
            if 'audiogoal' in self.goal_sensor_uuid:
                audiogoal_sensor = 'audiogoal'
            elif 'spectrogram' in self.goal_sensor_uuid:
                audiogoal_sensor = 'spectrogram'
            self.audio_encoder = AudioCRNN(observation_space, hidden_size, audiogoal_sensor)

        rnn_input_size = (0 if self.is_blind else self._hidden_size) + \
                         (self._number_pointgoal if self._pointgoal else 0) + (self._hidden_size if self._audiogoal else 0)
        self.state_encoder = RNNStateEncoder(rnn_input_size, self._hidden_size)

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers
    
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = []
        
        if self._pointgoal:
            x.append(observations[self.goal_sensor_uuid.split(DUAL_GOAL_DELIMITER)[0]])
        
        if self._audiogoal:
            x.append(self.audio_encoder(observations))

        if not self.is_blind:
            x.append(self.visual_encoder(observations))

        x1 = torch.cat(x, dim=1)
        x2, rnn_hidden_states1 = self.state_encoder(x1, rnn_hidden_states, masks)

        if torch.isnan(x2).any().item():
            for key in observations:
                print(key, torch.isnan(observations[key]).any().item())
            print('rnn_old', torch.isnan(rnn_hidden_states).any().item())
            print('rnn_new', torch.isnan(rnn_hidden_states1).any().item())
            print('mask', torch.isnan(masks).any().item())
            assert True
        
        return x2, rnn_hidden_states1
    

class CriticHead(nn.Module):
    def __init__(self, input_size) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)
    

class Policy(nn.Module):
    def __init__(self, net: Net, dim_actions) -> None:
        super().__init__()

        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(self.net.output_size, self.dim_actions)
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError
    
    def act(self, observations, rnn_hidden_states, prev_actions, masks, deterministic=False):
        features, rnn_hidden_states = self.net(observations, rnn_hidden_states, prev_actions, masks)

        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()
        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(observations, rnn_hidden_states, prev_actions, masks)
        return self.critic(features)
    
    def evaluate_actions(self, observations, rnn_hidden_states, prev_actions, masks, action):
        features, rnn_hidden_states = self.net(observations, rnn_hidden_states, prev_actions, masks)
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        dist_entropy = distribution.entropy().mean()
        
        return value, action_log_probs, dist_entropy, rnn_hidden_states
    

class SenBaselinePolicy(Policy):
    def __init__(self, observation_space, action_space, goal_sensor_uuid, hidden_size=512, extra_rgb=False) -> None:
        net = SenBaselineNet(observation_space, hidden_size, goal_sensor_uuid, extra_rgb)
        super().__init__(net, action_space.n)
    
    def forward(self, *x):
        raise NotImplementedError