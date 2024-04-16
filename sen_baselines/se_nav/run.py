#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import torch

import sen
from ss_baselines.common.baseline_registry import baseline_registry
from sen_baselines.se_nav.config.default import get_config

import tensorflow as tf


def main():
    # 从命令行传入参数构建训练过程参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        # required=True,
        default='train',
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        # required=True,
        default='av_nav/config/pointgoal_rgb.yaml',
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="Evaluation interval of checkpoints",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action='store_true',
        help="Modify config options from command line"
    )
    parser.add_argument(
        "--eval-best",
        default=False,
        action='store_true',
        help="Modify config options from command line"
    )
    parser.add_argument(
        "--prev-ckpt-ind",
        type=int,
        default=-1,
        help="Evaluation interval of checkpoints",
    )
    args = parser.parse_args()

    if args.eval_best:
        best_ckpt_idx = find_best_ckpt_idx(os.path.join(args.model_dir, 'tb'))
        best_ckpt_path = os.path.join(args.model_dir, 'data', f'ckpt.{best_ckpt_idx}.pth')
        print(f'Evaluating the best checkpoint: {best_ckpt_path}')
        args.opts += ['EVAL_CKPT_PATH_DIR', best_ckpt_path]

    # run exp
    config = get_config(args.exp_config, args.opts, args.model_dir, args.run_type, args.overwrite)
    # <class 'type'> train_init只是一个句柄, 其类型就是type
    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    # <class 'ss_baselines.av_nav.ddppo.ddppo_trainer.DDPPOTrainer'> trainer才是真正的trainer类型
    # 这里测试使用的ddppo.yaml, 所以类型是DDPPOTrainer
    # 从DDPPOTrainer看起, 第一步找到eval的方法, 用于SELDnet方法的构建
    trainer = trainer_init(config)
    torch.set_num_threads(1)

    level = logging.DEBUG if config.DEBUG else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    # 下面的train和eval调用的是DDPPOTrainer的train和eval函数
    # 应该需要重写eval函数
    if args.run_type == "train":
        trainer.train()
    elif args.run_type == "eval":
        trainer.eval(args.eval_interval, args.prev_ckpt_ind, config.USE_LAST_CKPT)


def find_best_ckpt_idx(event_dir_path, min_step=-1, max_step=10000):
    events = os.listdir(event_dir_path)

    max_value = 0
    max_index = -1
    for event in events:
        if "events" not in event:
            continue
        iterator = tf.compat.v1.train.summary_iterator(os.path.join(event_dir_path, event))
        for e in iterator:
            if len(e.summary.value) == 0:
                continue
            if not e.summary.value[0].tag.startswith('val'):
                break
            if 'spl' not in e.summary.value[0].tag:
                continue
            if not min_step <= e.step <= max_step:
                continue
            if len(e.summary.value) > 0 and e.summary.value[0].simple_value > max_value:
                max_value = e.summary.value[0].simple_value
                max_index = e.step

    if max_index == -1:
        print('No max index is found in {}'.format(event_dir_path))
    else:
        print('The best index in {} is {}'.format(event_dir_path, max_index))

    return max_index


if __name__ == "__main__":
    main()
