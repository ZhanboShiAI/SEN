import attr
from typing import Any, Optional, List
from gym import spaces
import numpy as np
import librosa
from skimage.measure import block_reduce

import os

from habitat.config import Config
from habitat.core.dataset import Episode, Dataset

from habitat.core.registry import registry
from habitat.core.simulator import (
    Sensor,
    SensorTypes,
    Simulator,
    AgentState,
)
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
)


@registry.register_sensor(name="SoundEventAudioGoalSensor")
class SoundEventAudioGoalSensor(Sensor):
    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "audiogoal"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (4, self._sim.config.AUDIO.RIR_SAMPLING_RATE)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_current_audiogoal_observation()
    

@registry.register_sensor(name="SoundEventSpectrogramSensor")
class SoundEventSpectrogramSensor(Sensor):
    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "spectrogram"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        spectrogram = self.compute_spectrogram(np.ones((4, self._sim.config.AUDIO.RIR_SAMPLING_RATE)))

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=spectrogram.shape,
            dtype=np.float32,
        )

    @staticmethod
    def compute_spectrogram(audio_data, sr=48000, hop_len_s=0.02):
        def _next_greater_power_of_2(x):
            return 2 ** (x-1).bit_length()

        def compute_stft(signal):
            hop_length = int(hop_len_s * sr)
            win_length = 2 * hop_length
            n_fft = _next_greater_power_of_2(win_length)
            
            # n_fft = 512
            # hop_length = 160
            # win_length = 400
            stft = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
            stft = block_reduce(stft, block_size=(4, 4), func=np.mean)
            return stft

        channel1_magnitude = np.log1p(compute_stft(audio_data[0]))
        channel2_magnitude = np.log1p(compute_stft(audio_data[1]))
        channel3_magnitude = np.log1p(compute_stft(audio_data[2]))
        channel4_magnitude = np.log1p(compute_stft(audio_data[3]))
        spectrogram = np.stack([
            channel1_magnitude, channel2_magnitude, channel3_magnitude, channel4_magnitude], axis=-1
        )

        return spectrogram

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        spectrogram = self._sim.get_current_spectrogram_observation(self.compute_spectrogram)

        return spectrogram
    

def optional_int(value):
    return int(value) if value is not None else None


@attr.s(auto_attribs=True, kw_only=True)
class SoundEventNavEpisode(NavigationEpisode):
    object_category: str
    sound_id: str
    duration: int = attr.ib(converter=int)
    offset: int = attr.ib(converter=int)
    interval_mean: int = attr.ib(converter=int)
    interval_upper_limit: int = attr.ib(converter=int)
    interval_lower_limit: int = attr.ib(converter=int)

    # TODO: Add distractor sound and noise sound
    # distractor_sound_id: Optional[str] = attr.ib(default=None)
    # distractor_duration: Optional[int] = attr.ib(default=None, converter=optional_int)
    # distractor_offset: Optional[int] = attr.ib(default=None, converter=optional_int)
    # distractor_position: Optional[List[float]] = attr.ib(default=None)
    # distractor_interval_mean: Optional[int] = attr.ib(default=None, converter=optional_int)
    # distractor_interval_upper_limit: Optional[int] = attr.ib(default=None, converter=optional_int)
    # distractor_interval_lower_limit: Optional[int] = attr.ib(default=None, converter=optional_int)

    # noise_sound_id: Optional[str] = attr.ib(default=None)

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals
        """
        return f"{os.path.basename(self.scene_id)}_{self.object_category}"


@attr.s(auto_attribs=True)
class ObjectViewLocation:
    agent_state: AgentState
    iou: Optional[float]

    
@attr.s(auto_attribs=True, kw_only=True)
class SoundEventGoal(NavigationGoal):
    object_id: str = attr.ib(default=None)
    object_name: Optional[str] = None
    object_category: Optional[str] = None
    room_id: Optional[str] = None
    room_name: Optional[str] = None
    view_points: Optional[List[ObjectViewLocation]] = attr.ib(factory=list)



