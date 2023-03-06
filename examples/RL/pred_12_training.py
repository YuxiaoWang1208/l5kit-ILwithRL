import argparse
import os

import sys
project_path = "/root/zhufenghua12/wangyuxiao/l5kit-wyx/l5kit"
print("project path: ", project_path)
sys.path.append(project_path)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import SubprocVecEnv

from l5kit.environment.envs.l5_env import SimulationConfigGym
from l5kit.environment.feature_extractor import CustomFeatureExtractor
from l5kit.environment.callbacks import L5KitEvalCallback


# Dataset is assumed to be on the folder specified
# in the L5KIT_DATA_FOLDER environment variable
# Please set the L5KIT_DATA_FOLDER environment variable
os.environ["L5KIT_DATA_FOLDER"] = "/root/zhufenghua12/l5kit/prediction"
os.chdir("/root/zhufenghua12/wangyuxiao/l5kit-wyx/examples/RL")
if "L5KIT_DATA_FOLDER" not in os.environ:
    raise KeyError("L5KIT_DATA_FOLDER environment variable not set")

# print(os.environ.keys())
config_path = os.getcwd() + "/pred_12_config.yaml"
os.environ.setdefault('CONFIG_PATH', config_path)
# print(os.environ["CONFIG_PATH"])

from pred_12 import PRED_12

import datetime

d = datetime.datetime.now() + datetime.timedelta(hours=8)
# date = "0207"
# print(d.strftime('%Y-%m-%d_%H-%M'))
date = d.strftime('%Y-%m-%d_%H-%M')

lr = 3e-4  # 3e-4 1e-3 6e-5
from l5kit.configs import load_config_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='./gym_config.yaml',
                        help='Path to L5Kit environment config file')
    parser.add_argument('-o', '--output', type=str, default='PPO',
                        help='File name for saving model states')
    parser.add_argument('--load', type=str, # default='./logs/PPO_100000_steps.zip',
                        help='Path to load model and continue training')
    parser.add_argument('--simnet', action='store_true',
                        help='Use simnet to control agents')
    parser.add_argument('--simnet_model_path', default=None, type=str,
                        help='Path to simnet model that controls agents')
    parser.add_argument('--tb_log', default='./tb_log_' + date + '/', type=str,
                        help='Tensorboard log folder')
    parser.add_argument('--save_path', default='./logs_' + date + '/', type=str,
                        help='Folder to save model checkpoints')
    parser.add_argument('--save_freq', default=10000, type=int,  # 10000
                        help='Frequency to save model checkpoints')
    parser.add_argument('--eval_freq', default=1000, type=int,  # 100000 1000
                        help='Frequency to evaluate model state')
    parser.add_argument('--n_eval_episodes', default=1, type=int,  # 10
                        help='Number of episodes to evaluate')
    parser.add_argument('--n_steps', default=1000000, type=int,
                        help='Total number of training time steps')
    parser.add_argument('--num_rollout_steps', default=256, type=int,  # 256
                        help='Number of rollout steps per environment per model update')
    parser.add_argument('--n_IL_epochs', default=60, type=int,
                        help='Number of model Imitation training epochs per update')
    parser.add_argument('--n_RL_epochs', default=40, type=int,
                        help='Number of model Reinforcement and Imitation training epochs per update')
    parser.add_argument('--n_epochs', default=250, type=int,
                        help='Number of model training epochs per update')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Mini batch size of model update')
    parser.add_argument('--lr', default=lr, type=float,  # 3e-4
                        help='Learning rate')
    parser.add_argument('--gamma', default=0.95, type=float,
                        help='Discount factor')
    parser.add_argument('--gae_lambda', default=0.90, type=float,
                        help='Factor for trade-off of bias vs variance for Generalized Advantage Estimator')
    parser.add_argument('--clip_start_val', default=0.1, type=float,
                        help='Start value of clipping in PPO')
    parser.add_argument('--clip_end_val', default=0.001, type=float,
                        help='End value of clipping in PPO')
    parser.add_argument('--clip_progress_ratio', default=1.0, type=float,
                        help='Training progress ratio to end linear schedule of clipping')
    parser.add_argument('--model_arch', default='simple_gn', type=str,
                        help='Model architecture of feature extractor')
    parser.add_argument('--features_dim', default=128, type=int,
                        help='Output dimension of feature extractor')
    parser.add_argument('--n_envs', default=4, type=int,
                        help='Number of parallel environments')
    parser.add_argument('--n_eval_envs', default=4, type=int,
                        help='Number of parallel environments for evaluation')
    parser.add_argument('--eps_length', default=128, type=int,  # 32 128
                        help='Episode length of gym rollouts')
    parser.add_argument('--rew_clip', default=15, type=float,
                        help='Reward clipping threshold')
    parser.add_argument('--kinematic', action='store_true',
                        help='Flag to use kinematic model in the environment')
    parser.add_argument('--enable_scene_type_aggregation', action='store_true',
                        help='enable scene type aggregation of evaluation metrics')
    parser.add_argument('--scene_id_to_type_path', default=None, type=str,
                        help='Path to csv file mapping scene id to scene type')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    # get config
    cfg = load_config_data(os.environ["CONFIG_PATH"])
    rand = cfg["gym_params"]["randomize_start_frame"]
    args.config = os.environ["CONFIG_PATH"]

    # Simnet model
    if args.simnet and (args.simnet_model_path is None):
        raise ValueError("simnet_model_path needs to be provided when using simnet")

    # make train env
    train_sim_cfg = SimulationConfigGym()
    train_sim_cfg.num_simulation_steps = args.eps_length + 1
    train_sim_cfg.use_agents_gt = (not args.simnet)
    env_kwargs = {'env_config_path': args.config, 'use_kinematic': args.kinematic, 'train': True,
                  'sim_cfg': train_sim_cfg, 'simnet_model_path': args.simnet_model_path}
    env = make_vec_env("L5-CLE-v0", env_kwargs=env_kwargs, n_envs=args.n_envs,
                       vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "fork"})

    # Custom Feature Extractor backbone
    args.model_arch = 'resnet50'
    pretrained = True
    policy_kwargs = {
        "features_extractor_class": CustomFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": args.features_dim, "model_arch": args.model_arch, "pretrained": pretrained},
        "normalize_images": False
    }

    # define model
    clip_schedule = get_linear_fn(args.clip_start_val, args.clip_end_val, args.clip_progress_ratio)
    if args.load is not None:
        model = PRED_12.load(args.load, env, clip_range=clip_schedule, learning_rate=args.lr)
    else:
        model = PRED_12("MultiInputPredPolicy", env, future_num_frames=cfg['model_params']['future_num_frames'],
                    policy_kwargs=policy_kwargs, verbose=1, n_steps=args.num_rollout_steps,
                    learning_rate=args.lr, gamma=args.gamma, tensorboard_log=args.tb_log,
                    # n_IL_epochs=args.n_IL_epochs, n_RL_epochs=args.n_RL_epochs,
                    n_epochs = args.n_epochs, weights_scaling=[1., 1., 1.],
                    clip_range=clip_schedule, batch_size=args.batch_size, seed=args.seed, gae_lambda=args.gae_lambda)

    # make eval env
    eval_sim_cfg = SimulationConfigGym()
    eval_sim_cfg.num_simulation_steps = None
    eval_sim_cfg.use_agents_gt = (not args.simnet)
    eval_env_kwargs = {'env_config_path': args.config, 'use_kinematic': args.kinematic, 'return_info': True,
                       'train': False, 'sim_cfg': eval_sim_cfg, 'simnet_model_path': args.simnet_model_path}
    eval_env = make_vec_env("L5-CLE-v0", env_kwargs=eval_env_kwargs, n_envs=args.n_eval_envs,
                            vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "fork"})

    # callbacks
    # Note: When using multiple environments, each call to ``env.step()``
    # will effectively correspond to ``n_envs`` steps.
    # To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``
    # Save Model Periodically
    checkpoint_callback = CheckpointCallback(save_freq=(args.save_freq // args.n_envs), save_path=args.save_path,
                                             name_prefix=args.output)
    MODEL_PATH = args.save_path
    # Eval Model Periodically
    eval_callback = L5KitEvalCallback(eval_env, eval_freq=(args.eval_freq // args.n_envs),
                                      n_eval_episodes=args.n_eval_episodes, n_eval_envs=args.n_eval_envs,
                                      prefix='l5_cle_eval', enable_scene_type_aggregation=args.enable_scene_type_aggregation,
                                      scene_id_to_type_path=args.scene_id_to_type_path)

    # train
    model.learn(args.n_steps, callback=[checkpoint_callback, eval_callback])
# 从模型外面写个函数从dataloader读it data