import contextlib
import os
import random
import time
from collections import defaultdict, deque
import logging
import json

import numpy as np
from ss_baselines.common.utils import observations_to_image, plot_top_down_map, resize_observation
import torch
import torch.distributed as distrib
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict
from gym import spaces

from habitat import Config, logger
from ss_baselines.common.baseline_registry import baseline_registry
from ss_baselines.common.env_utils import construct_envs
from ss_baselines.common.tensorboard_utils import TensorboardWriter
from ss_baselines.common.environments import get_env_class
from ss_baselines.common.rollout_storage import RolloutStorage
from ss_baselines.common.utils import batch_obs, linear_decay, generate_video
from ss_baselines.savi.ddppo.algo.ddp_utils import (
    EXIT,
    REQUEUE,
    add_signal_handlers,
    init_distrib_slurm,
    load_interrupted_state,
    requeue_job,
    save_interrupted_state,
)
from sen_baselines.av_nav.ddppo.ddppo import DDPPO
from sen_baselines.av_nav.ddppo.ppo_trainer import PPOTrainer
from sen_baselines.av_nav.ppo.policy import AudioNavBaselinePolicy

from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

from tqdm import tqdm


@baseline_registry.register_trainer(name="AVNavDDPPOTrainer")
class DDPPOTrainer(PPOTrainer):
    SHORT_ROLLOUT_THRESHOLD: float = 0.25

    def __init__(self, config=None) -> None:
        interrupted_state = load_interrupted_state()
        if interrupted_state is not None:
            config = interrupted_state["config"]

        super().__init__(config)

    def _setup_actor_critic_agent(self, ppo_cfg: Config, observation_space=None) -> None:
        r"""Sets up actor critic and agent for DD-PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        # Setup heuristic stop criterion if applicable
        action_space = self.envs.action_spaces[0]
        self.action_space = action_space
        # print("DDPPO Trainer setup_actor_critic_agnet action_space.n: ", action_space.n)

        has_distractor_sound = self.config.TASK_CONFIG.SIMULATOR.AUDIO.HAS_DISTRACTOR_SOUND
        self.actor_critic = AudioNavBaselinePolicy(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.action_space,
            hidden_size=ppo_cfg.hidden_size,
            goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
            extra_rgb=self.config.EXTRA_RGB,
        )

        self.actor_critic.to(self.device)

        if self.config.RL.DDPPO.reset_critic:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        self.agent = DDPPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )

    def train(self) -> None:
        r"""Main method for DD-PPO.

        Returns:
            None
        """
        self.local_rank, tcp_store = init_distrib_slurm(
            self.config.RL.DDPPO.distrib_backend
        )
        add_signal_handlers()

        # Stores the number of workers that have finished their rollout
        num_rollouts_done_store = distrib.PrefixStore(
            "rollout_tracker", tcp_store
        )
        num_rollouts_done_store.set("num_done", "0")

        self.world_rank = distrib.get_rank()
        print(f'World rank: {self.world_rank}')
        self.world_size = distrib.get_world_size()

        self.config.defrost()
        self.config.TORCH_GPU_ID = self.local_rank
        self.config.SIMULATOR_GPU_ID = self.local_rank
        # Multiply by the number of simulators to make sure they also get unique seeds
        self.config.TASK_CONFIG.SEED += (
            self.world_rank * self.config.NUM_PROCESSES
        )
        self.config.freeze()

        random.seed(self.config.TASK_CONFIG.SEED)
        np.random.seed(self.config.TASK_CONFIG.SEED)
        torch.manual_seed(self.config.TASK_CONFIG.SEED)

        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )

        ppo_cfg = self.config.RL.PPO
        if (
            not os.path.isdir(self.config.CHECKPOINT_FOLDER)
            and self.world_rank == 0
        ):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_actor_critic_agent(ppo_cfg)
        self.agent.init_distributed(find_unused_params=True)

        if self.world_rank == 0:
            logger.info(
                "agent number of trainable parameters: {}".format(
                    sum(
                        param.numel()
                        for param in self.agent.parameters()
                        if param.requires_grad
                    )
                )
            )
            logger.info(f"config: {self.config}")

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        obs_space = self.envs.observation_spaces[0]

        rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            obs_space,
            self.action_space,
            ppo_cfg.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
        )
        rollouts.to(self.device)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1, device=self.device),
            reward=torch.zeros(self.envs.num_envs, 1, device=self.device),
        )
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0
        start_update = 0
        prev_time = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        # Try to resume at previous checkpoint (independent of interrupted states)
        count_steps_start, count_checkpoints, start_update = self.try_to_resume_checkpoint()
        count_steps = count_steps_start

        interrupted_state = load_interrupted_state()
        if interrupted_state is not None:
            self.agent.load_state_dict(interrupted_state["state_dict"])
            self.agent.optimizer.load_state_dict(
                interrupted_state["optim_state"]
            )
            lr_scheduler.load_state_dict(interrupted_state["lr_sched_state"])

            requeue_stats = interrupted_state["requeue_stats"]
            env_time = requeue_stats["env_time"]
            pth_time = requeue_stats["pth_time"]
            count_steps = requeue_stats["count_steps"]
            count_checkpoints = requeue_stats["count_checkpoints"]
            start_update = requeue_stats["start_update"]
            prev_time = requeue_stats["prev_time"]

        with (
            TensorboardWriter(
                self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
            )
            if self.world_rank == 0
            else contextlib.suppress()
        ) as writer:
            for update in range(start_update, self.config.NUM_UPDATES):
                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                if EXIT.is_set():
                    self.envs.close()

                    if REQUEUE.is_set() and self.world_rank == 0:
                        requeue_stats = dict(
                            env_time=env_time,
                            pth_time=pth_time,
                            count_steps=count_steps,
                            count_checkpoints=count_checkpoints,
                            start_update=update,
                            prev_time=(time.time() - t_start) + prev_time,
                        )
                        state_dict = dict(
                                state_dict=self.agent.state_dict(),
                                optim_state=self.agent.optimizer.state_dict(),
                                lr_sched_state=lr_scheduler.state_dict(),
                                config=self.config,
                                requeue_stats=requeue_stats,
                            )
                        save_interrupted_state(state_dict)

                    requeue_job()
                    return

                count_steps_delta = 0
                self.agent.eval()
                for step in range(ppo_cfg.num_steps):

                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                    ) = self._collect_rollout_step(
                        rollouts, current_episode_reward, running_episode_stats
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps_delta += delta_steps

                    # This is where the preemption of workers happens.  If a
                    # worker detects it will be a straggler, it preempts itself!
                    if (
                        step
                        >= ppo_cfg.num_steps * self.SHORT_ROLLOUT_THRESHOLD
                    ) and int(num_rollouts_done_store.get("num_done")) > (
                        self.config.RL.DDPPO.sync_frac * self.world_size
                    ):
                        break

                num_rollouts_done_store.add("num_done", 1)

                self.agent.train()

                (
                    delta_pth_time,
                    value_loss,
                    action_loss,
                    dist_entropy,
                ) = self._update_agent(ppo_cfg, rollouts)
                pth_time += delta_pth_time

                stats_ordering = list(sorted(running_episode_stats.keys()))
                stats = torch.stack(
                    [running_episode_stats[k] for k in stats_ordering], 0
                )
                distrib.all_reduce(stats)

                for i, k in enumerate(stats_ordering):
                    window_episode_stats[k].append(stats[i].clone())

                stats = torch.tensor(
                    [value_loss, action_loss, dist_entropy, count_steps_delta],
                    device=self.device,
                )
                distrib.all_reduce(stats)
                count_steps += stats[3].item()

                if self.world_rank == 0:
                    num_rollouts_done_store.set("num_done", "0")

                    losses = [
                        stats[0].item() / self.world_size,
                        stats[1].item() / self.world_size,
                        stats[2].item() / self.world_size,
                    ]
                    deltas = {
                        k: (
                            (v[-1] - v[0]).sum().item()
                            if len(v) > 1
                            else v[0].sum().item()
                        )
                        for k, v in window_episode_stats.items()
                    }
                    deltas["count"] = max(deltas["count"], 1.0)

                    writer.add_scalar(
                        "Metrics/reward", deltas["reward"] / deltas["count"], count_steps
                    )

                    # Check to see if there are any metrics
                    # that haven't been logged yet
                    metrics = {
                        k: v / deltas["count"]
                        for k, v in deltas.items()
                        if k not in {"reward", "count"}
                    }
                    if len(metrics) > 0:
                        for metric, value in metrics.items():
                            writer.add_scalar(f"Metrics/{metric}", value, count_steps)

                    writer.add_scalar("Policy/value_loss", losses[0], count_steps)
                    writer.add_scalar("Policy/policy_loss", losses[1], count_steps)
                    writer.add_scalar("Policy/entropy_loss", losses[2], count_steps)
                    writer.add_scalar('Policy/learning_rate', lr_scheduler.get_lr()[0], count_steps)

                    # log stats
                    if update > 0 and update % self.config.LOG_INTERVAL == 0:
                        logger.info(
                            "update: {}\tfps: {:.3f}\t".format(
                                update,
                                (count_steps - count_steps_start)
                                / ((time.time() - t_start) + prev_time),
                            )
                        )

                        logger.info(
                            "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                            "frames: {}".format(
                                update, env_time, pth_time, count_steps
                            )
                        )
                        logger.info(
                            "Average window size: {}  {}".format(
                                len(window_episode_stats["count"]),
                                "  ".join(
                                    "{}: {:.3f}".format(k, v / deltas["count"])
                                    for k, v in deltas.items()
                                    if k != "count"
                                ),
                            )
                        )

                    # checkpoint model
                    if update % self.config.CHECKPOINT_INTERVAL == 0:
                        self.save_checkpoint(
                            f"ckpt.{count_checkpoints}.pth",
                            dict(step=count_steps),
                        )
                        count_checkpoints += 1

            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int,
    ) -> Dict:
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()        
        ppo_cfg = config.RL.PPO
        
        config.defrost()
        config.TASK_CONFIG.DATASET.SPlIT = config.EVAL.SPLIT
        if self.config.DISPLAY_RESOLUTION != config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH:
            model_resolution = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH
            config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = \
                config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT = \
                self.config.DISPLAY_RESOLUTION
        else:
            model_resolution = self.config.DISPLAY_RESOLUTION
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()
        elif "top_down_map" in self.config.VISUALIZATION_OPTION:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        if self.config.DISPLAY_RESOLUTION != model_resolution:
            observation_space = self.envs.observation_spaces[0]
            observation_space.spaces['depth'] = spaces.Box(low=0, high=1, shape=(model_resolution,
                                                           model_resolution, 1), dtype=np.uint8)
            observation_space.spaces['rgb'] = spaces.Box(low=0, high=1, shape=(model_resolution,
                                                         model_resolution, 3), dtype=np.uint8)
        else:
            observation_space = self.envs.observation_spaces[0]
        
        self._setup_actor_critic_agent(ppo_cfg, observation_space)

        # if config.FOLLOW_SHORTEST_PATH:
        #     follower = ShortestPathFollower(
        #         self.envs.workers[0]._env.habitat_env.sim, 0.5, False
        #     )

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        self.metric_uuids = []
        # 获得metric的name, 比如"spl", "success"
        for metric_name in self.config.TASK_CONFIG.TASK.MEASUREMENTS:
            metric_cfg = getattr(self.config.TASK_CONFIG.TASK, metric_name)
            measure_type = baseline_registry.get_measure(metric_cfg.TYPE)
            assert measure_type is not None, "invalid measurement type {}".format(metric_cfg.TYPE)
            self.metric_uuids.append(measure_type(sim=None, task=None, config=None)._get_uuid())

        observations = self.envs.reset()
        if self.config.DISPLAY_RESOLUTION != model_resolution:
            resize_observation(observations, model_resolution)
        batch = batch_obs(observations, device=self.device)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )
        stats_episodes = dict()

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]
        audios = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        t = tqdm(total=self.config.TEST_EPISODE_COUNT)
        while (
            len(stats_episodes) < self.config.TEST_EPISODE_COUNT
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()
            
            # print("DDPPO Trainer batch depth shape: ", batch["depth"].shape)
            # torch.Size([10, 128, 128, 1])
            # print("DDPPO Trainer hidden_states shape: ", test_recurrent_hidden_states.shape)
            # torch.Size([1, 10, 512])
            # print("DDPPO Trainer prev_actions shape: ", prev_actions.shape)
            # torch.Size([10])
            # print("DDPPO Trainer not_done_masks shape: ", not_done_masks.shape)
            # torch.Size([10, 1])
            
            
            with torch.no_grad():
                _, actions, _, test_recurrent_hidden_states = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )
                # print("DDPPO Trainer actions shape: ", actions.shape)
                # torch.Size([10, 1])
                prev_actions.copy_(actions)

            # if config.FOLLOW_SHORTEST_PATH:
            #     try:
            #         action = [follower.get_next_action(
            #             self.envs.workers[0]._env.habitat_env.current_episode.goals[0].position
            #         )]
            #         outputs = self.envs.step(action)
            #     except Exception as e:
            #         logger.error(f"Error following shortest path: {e}")
            #         outputs = self.envs.step([a[0].item() for a in actions])
            # else:
            outputs = self.envs.step([a[0].item() for a in actions])

            observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
            for i in range(self.envs.num_envs):
                if len(self.config.VIDEO_OPTION) > 0:
                    if config.TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE and "intermediate" in observations[i]:
                        for observation in observations[i]["intermediate"]:
                            frame = observations_to_image(observation, infos[i])
                            rgb_frames[i].append(frame)
                        del observations[i]["intermediate"]
                    
                    if "rgb" in observations[i]:
                        observations[i]["rgb"] = np.zeros((self.config.DISPLAY_RESOLUTION, 
                                                           self.config.DISPLAY_RESOLUTION, 3))
                    frame = observations_to_image(observations[i], infos[i])
                    rgb_frames[i].append(frame)
                    audios[i].append(observations[i]["audiogoal"])
                
            if config.DISPLAY_RESOLUTION != model_resolution:
                resize_observation(observations, model_resolution)
            batch = batch_obs(observations, device=self.device)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )
            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            for i in range(self.envs.num_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                if not_done_masks[i].item() == 0:
                    episode_stats = dict()
                    for metric_uuid in self.metric_uuids:
                        episode_stats[metric_uuid] = infos[i][metric_uuid]
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats['geodesic_distance'] = current_episodes[i].info['geodesic_distance']
                    episode_stats['euclidean_distance'] = np.linalg.norm(
                                                                np.array(current_episodes[i].goals[0].position) -
                                                                np.array(current_episodes[i].start_position)
                                                            )
                    logging.debug(episode_stats)
                    current_episode_reward[i] = 0
                    stats_episodes[(
                        current_episodes[i].scene_id,
                        current_episodes[i].episode_id,
                    )] = episode_stats
                    t.update()

                    if len(self.config.VIDEO_OPTION) > 0:
                        fps = int(1 / self.config.TASK_CONFIG.SIMULATOR.STEP_TIME)
                        if "sound" in current_episodes[i].info:
                            sound = current_episodes[i].info["sound"]
                        else:
                            sound = current_episodes[i].sound_id.split('/')[-1][:-4]
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i][:-1],
                            scene_name=current_episodes[i].scene_id.split('/')[3],
                            sound=sound,
                            sr=self.config.TASK_CONFIG.SIMULATOR.AUDIO.RIR_SAMPLING_RATE,
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metric_name='spl',
                            metric_value=infos[i]['spl'],
                            tb_writer=writer,
                            audios=audios[i][:-1],
                            fps=fps
                        )

                        rgb_frames[i] = []
                        audios[i] = []
                    
                    if "top_down_map" in self.config.VISUALIZATION_OPTION:
                        top_down_map = plot_top_down_map(
                            infos[i], dataset=self.config.TASK_CONFIG.SIMULATOR.SCENE_DATASET
                        )
                        scene = current_episodes[i].scene_id.split('/')[-2]
                        writer.add_image(
                                        "{}_{}_{}/{}".format(
                                                        config.EVAL.SPLIT, 
                                                        scene, current_episodes[i].episode_id, 
                                                        config.BASE_TASK_CONFIG_PATH.split('/')[-1][:-5], 
                                                        ), 
                                        top_down_map, 
                                        dataformats='WHC'
                                    )
            
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = sum(
                [v[stat_key] for v in stats_episodes.values()]
            )
        num_episdoes = len(stats_episodes)

        stats_file = os.path.join(
            config.TENSORBOARD_DIR,
            "{}_stats_{}.json".format(
                config.EVAL.SPLIT, config.SEED
            ),
        )
        new_stats_episodes = {",".join(key): value for key, value in stats_episodes.items()}
        with open(stats_file, "w") as f:
            json.dump(new_stats_episodes, f)

        episode_reward_mean = aggregated_stats["reward"] / num_episdoes
        episode_metric_mean = {}
        for metric_uuid in self.metric_uuids:
            episode_metric_mean[metric_uuid] = aggregated_stats[metric_uuid] / num_episdoes
        logger.info(f"Average episode reward: {episode_reward_mean:.6f}")
        for metric_uuid, metric_mean in episode_metric_mean.items():
            logger.info(f"Average episode {metric_uuid}: {metric_mean:.6f}")

        if not config.EVAL.SPLIT.startswith("test"):
            writer.add_scalar(
                "{}/reward".format(config.EVAL.SPLIT),
                episode_reward_mean,
                checkpoint_index
            )
            for metric_uuid, metric_mean in episode_metric_mean.items():
                writer.add_scalar(
                    "{}/{}".format(config.EVAL.SPLIT, metric_uuid),
                    metric_mean,
                    checkpoint_index
                )
        
        self.envs.close()

        result = {
            "episode_reward_mean": episode_reward_mean,
        }
        for metric_uuid, metric_mean in episode_metric_mean.items():
            result["episode_{}_mean".format(metric_uuid)] = metric_mean
        
        return result