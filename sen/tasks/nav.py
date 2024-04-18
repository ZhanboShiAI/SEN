import attr
from typing import Any, Optional, List, Union
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
        # spectrogram = self.compute_spectrogram(np.ones((4, self._sim.config.AUDIO.RIR_SAMPLING_RATE)))
        # spectrogram = np.stack([spectrogram.real, spectrogram.imag], axis=-1)
        if self._sim.config.AUDIO.TYPE == "ambisonic":
            spectrogram = self.compute_spectrogram_ambisonic(np.ones((4, self._sim.config.AUDIO.RIR_SAMPLING_RATE)))
        elif self._sim.config.AUDIO.TYPE == "binaural":
            spectrogram = self.compute_spectrogram_binaural(np.ones((2, self._sim.config.AUDIO.RIR_SAMPLING_RATE)))
        elif self._sim.config.AUDIO.TYPE == "mel_foa_iv":
            spectrogram = self.compute_mel_and_foa_intensity_spectrogram(np.ones((4, self._sim.config.AUDIO.RIR_SAMPLING_RATE)))
        elif self._sim.config.AUDIO.TYPE == "mel_foa_iv_5len":
            spectrogram = self.compute_mel_and_foa_intensity_spectrogram(np.ones((4, self._sim.config.AUDIO.RIR_SAMPLING_RATE * 5)))
        else:
            raise NotImplementedError(f"Audio type {self._sim.config.AUDIO.TYPE} not supported")
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=spectrogram.shape,
            dtype=np.float32,
        )

    @staticmethod
    def compute_spectrogram_ambisonic(audio_data, sr=16000, hop_len_s=0.02):
        def _next_greater_power_of_2(x):
            return 2 ** (x-1).bit_length()
        
        def compute_stft(signal):
            # hop_length = int(hop_len_s * sr)
            # win_length = 2 * hop_length
            # n_fft = _next_greater_power_of_2(win_length)
            hop_length = 320
            win_length = 640
            n_fft = 1024
            stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            # print("stft shape: ", stft.shape)
            stft_stack = np.stack([np.real(stft), np.imag(stft)], axis=-1)
            # print("stft_stack shape: ", stft_stack.shape)
            return stft_stack
        
        spectrogram = np.stack([compute_stft(channel) for channel in audio_data], axis=-1)
        return spectrogram
    
    @staticmethod
    def compute_spectrogram_binaural(audio_data, sr=16000, hop_len_s=0.02):
        def compute_stft(signal):
            n_fft = 512
            hop_length = 160
            win_length = 400
            stft = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
            stft = block_reduce(stft, block_size=(4, 4), func=np.mean)
            return stft
        
        channel1_magnitude = np.log1p(compute_stft(audio_data[0]))
        channel2_magnitude = np.log1p(compute_stft(audio_data[1]))
        spectrogram = np.stack([
            channel1_magnitude, channel2_magnitude], axis=-1
        )
        return spectrogram
    
    @staticmethod
    def compute_mel_and_foa_intensity_spectrogram(audio_data, sr=16000, hop_len_s=0.02):
        def compute_stft(signal):
            hop_length = 320
            win_length = 640
            n_fft = 1024
            stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            return stft
        
        def compute_mel_spectrogram(signal, mel_wts):
            mel_feature = np.zeros((signal.shape[0], 64, signal.shape[-1]))
            for channel in range(signal.shape[-1]):
                mag_spectra = np.abs(signal[:, :, channel])**2
                mel_spectra = np.dot(mag_spectra, mel_wts)
                log_mel_spectra = librosa.power_to_db(mel_spectra)
                mel_feature[:, :, channel] = log_mel_spectra
            # dimension of mel_feature is (frames, mel_bins, channels(4))
            # mel_feature = mel_feature.transpose((0, 2, 1)).reshape((signal.shape[0], -1))
            return mel_feature
        
        def compute_foa_intensity_spectrogram(signal, mel_wts):
            W = signal[:, :, 0]
            I = np.real(np.conj(W)[:, :, np.newaxis] * signal[:, :, 1:])
            E = 1e-8 + (np.abs(W)**2 + ((np.abs(signal[:, :, 1:])**2).sum(-1))/3.0)

            I_norm = I / E[:, :, np.newaxis]
            I_norm_mel = np.transpose(np.dot(np.transpose(I_norm, (0, 2, 1)), mel_wts), (0, 2, 1))
            foa_iv = np.nan_to_num(I_norm_mel, nan=1e-8)
            # dimension of foa_iv is (frames, mel_bins, channels-1(3))
            # foa_iv = I_norm_mel.transpose((0, 2, 1)).reshape((signal.shape[0], 64*3))
            
            return foa_iv
        
        spectrogram = np.stack([
            compute_stft(channel) for channel in audio_data
        ], axis=-1).transpose((1, 0, 2))
        # the dimension of spectrogram is (frames, freq_bins, channels)

        mel_wts = librosa.filters.mel(sr=16000, n_fft=1024, n_mels=64).T
        mel_spectrogram = compute_mel_spectrogram(spectrogram, mel_wts)
        foa_iv = compute_foa_intensity_spectrogram(spectrogram, mel_wts)
        feature = np.concatenate((mel_spectrogram, foa_iv), axis=-1)
        # dimension of feature is (frames, mel_bins, channel(4+3))
        return feature
    
    # def compute_spectrogram(audio_data, sr=16000, hop_len_s=0.02):
    #     def _next_greater_power_of_2(x):
    #         return 2 ** (x-1).bit_length()
        
    #     # def compute_stft(signal):
    #     #     n_fft = 512
    #     #     hop_length = 160
    #     #     win_length = 400
    #     #     stft = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
    #     #     print("stft shape: ", stft.shape)
    #     #     stft = block_reduce(stft, block_size=(4, 4), func=np.mean)
    #     #     return stft
        
    #     def compute_stft(signal):
    #         hop_length = int(hop_len_s * sr)
    #         win_length = 2 * hop_length
    #         n_fft = _next_greater_power_of_2(win_length)
    #         stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    #         # print("stft shape: ", stft.shape)
    #         stft_stack = np.stack([np.real(stft), np.imag(stft)], axis=-1)
    #         # print("stft_stack shape: ", stft_stack.shape)
    #         return stft_stack

    #     # channel1_magnitude = np.log1p(compute_stft(audio_data[0]))
    #     # channel2_magnitude = np.log1p(compute_stft(audio_data[1]))
    #     # channel3_magnitude = np.log1p(compute_stft(audio_data[2]))
    #     # channel4_magnitude = np.log1p(compute_stft(audio_data[3]))
    #     # spectrogram = np.stack([
    #     #     channel1_magnitude, channel2_magnitude, channel3_magnitude, channel4_magnitude], axis=-1
    #     # )
    #     spectrogram = np.stack([compute_stft(channel) for channel in audio_data], axis=-1)
    #     # print("spectrogram shape: ", spectrogram.shape)

    #     return spectrogram

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        if self._sim.config.AUDIO.TYPE == "ambisonic":
            spectrogram = self._sim.get_current_spectrogram_observation(self.compute_spectrogram_ambisonic)
        elif self._sim.config.AUDIO.TYPE == "binaural":
            spectrogram = self._sim.get_current_spectrogram_observation(self.compute_spectrogram_binaural)
        elif self._sim.config.AUDIO.TYPE in ["mel_foa_iv", "mel_foa_iv_5len"]:
            spectrogram = self._sim.get_current_spectrogram_observation(self.compute_mel_and_foa_intensity_spectrogram)
        else:
            raise NotImplementedError(f"Audio type {self._sim.config.AUDIO.TYPE} not supported")

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



@registry.register_sensor(name="SoundEventCategory")
class SenCategory(Sensor):
    def __init__(self, sim: Union[Simulator, Config], config: Config, *args: Any, **kwargs: Any) -> None:
        self._sim = sim
        self._category_mapping = {
            "bathtub": 0, 
            "bed": 1, 
            "cabinet": 2, 
            "chair": 3, 
            "chest_of_drawers": 4, 
            "clothes": 5,
            "counter": 6, 
            "cushion": 7, 
            "fireplace": 8, 
            "gym_equipment": 9, 
            "picture": 10, 
            "plant": 11, 
            "seating": 12, 
            "shower": 13, 
            "sink": 14, 
            "sofa": 15, 
            "stool": 16, 
            "table": 17, 
            "toilet": 18, 
            "towel": 19, 
            "tv_monitor": 20
        }
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "category"
    
    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.COLOR
    
    def _get_observation_space(self, *args: Any, **kwargs: Any) -> spaces.Space:
        return spaces.Box(
            low=0,
            high=1,
            shape=(len(self._category_mapping.keys()),),
            dtype=bool,
        )
    
    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any) -> Any:
        index = self._category_mapping[episode.object_category]
        onehot = np.zeros(len(self._category_mapping))
        onehot[index] = 1
        return onehot