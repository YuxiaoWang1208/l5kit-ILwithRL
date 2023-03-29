import argparse
import os

import sys
project_path = "/root/zhufenghua12/wangyuxiao/l5kit-wyx/l5kit"
print("project path: ", project_path)
sys.path.append(project_path)

from pathlib import Path
import numpy as np
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import SubprocVecEnv

from l5kit.environment.envs.l5_env import SimulationConfigGym
from l5kit.environment.feature_extractor import CustomFeatureExtractor
from l5kit.environment.callbacks import L5KitEvalCallback

from l5kit.planning.rasterized.model import RasterizedPlanningModel
from l5kit.configs import load_config_data

# Dataset is assumed to be on the folder specified
# in the L5KIT_DATA_FOLDER environment variable
# Please set the L5KIT_DATA_FOLDER environment variable
os.environ["L5KIT_DATA_FOLDER"] = "/root/zhufenghua12/l5kit/prediction"
os.chdir("/root/zhufenghua12/wangyuxiao/l5kit-wyx/examples/RL")
if "L5KIT_DATA_FOLDER" not in os.environ:
    raise KeyError("L5KIT_DATA_FOLDER environment variable not set")

config_path = os.getcwd() + "/pred_12_config.yaml"
os.environ.setdefault('CONFIG_PATH', config_path)

from pred_12 import PRED_12
from get_il_eval_data import get_frame_data, un_rescale, rescale


date = "pretrain_withturn1"  # il3_1000 纯12步预测1000场景validate训练 il3 纯12步预测sample39号单场景 il4 纯1步预测sample39号单场景
steps = "5000"  # 100000
scene_id = 14  # 转弯场景：39x 红灯场景：12x 25 绿灯启动场景：13x 15 弯道场景：直道场景：40 58
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='./gym_config.yaml',
                        help='Path to L5Kit environment config file')
    parser.add_argument('-o', '--output', type=str, default='PPO',
                        help='File name for saving model states')
    parser.add_argument('--load', type=str, default='./logs_' + date + '/PPO_' + steps + '_steps.zip',  # './logs_lr0.0003len128/PPO_1000000_steps.zip'
                        help='Path to load model and continue training')
    parser.add_argument('--simnet', action='store_true',
                        help='Use simnet to control agents')
    parser.add_argument('--simnet_model_path', default=None, type=str,
                        help='Path to simnet model that controls agents')
    parser.add_argument('--tb_log', default='./tb_log/', type=str,
                        help='Tensorboard log folder')
    parser.add_argument('--save_path', default='./logs/', type=str,
                        help='Folder to save model checkpoints')
    parser.add_argument('--save_freq', default=1000, type=int,  # 100000 1000
                        help='Frequency to save model checkpoints')
    parser.add_argument('--eval_freq', default=1000, type=int,  # 100000 1000
                        help='Frequency to evaluate model state')
    parser.add_argument('--n_eval_episodes', default=1, type=int,  # 10
                        help='Number of episodes to evaluate')
    parser.add_argument('--n_steps', default=1000000, type=int,
                        help='Total number of training time steps')
    parser.add_argument('--num_rollout_steps', default=256, type=int,
                        help='Number of rollout steps per environment per model update')
    parser.add_argument('--n_epochs', default=10, type=int,
                        help='Number of model training epochs per update')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Mini batch size of model update')
    parser.add_argument('--lr', default=3e-4, type=float,
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
    parser.add_argument('--eps_length', default=128, type=int,
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


    args.config = os.environ["CONFIG_PATH"]

    # Simnet model
    if args.simnet and (args.simnet_model_path is None):
        raise ValueError("simnet_model_path needs to be provided when using simnet")

    # Custom Feature Extractor backbone
    policy_kwargs = {
        "features_extractor_class": CustomFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": args.features_dim, "model_arch": args.model_arch},
        "normalize_images": False
    }

    sys.path.append("/root/zhufenghua12/wangyuxiao/l5kit-wyx/examples/RL")
    from get_il_eval_data import rasterizer
    cfg = load_config_data(args.config)

    # train
    # model.learn(args.n_steps, callback=[checkpoint_callback, eval_callback])

    # visualize
    rollout_sim_cfg = SimulationConfigGym()
    rollout_sim_cfg.num_simulation_steps = None
    rollout_sim_cfg.use_agents_gt = (not args.simnet)
    rollout_env_kwargs = {'env_config_path': args.config, 'use_kinematic': args.kinematic, 'return_info': True,
                       'train': False, 'sim_cfg': rollout_sim_cfg, 'simnet_model_path': args.simnet_model_path}
    rollout_env = make_vec_env("L5-CLE-v0", env_kwargs=rollout_env_kwargs, n_envs=1,
                            vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "fork"})
    
    rescale_action = rollout_env.get_attr('rescale_action')[0]
    use_kinematic = rollout_env.get_attr('use_kinematic')[0]

    # to use transform to avoid the difference of rescale between training and evaluating:
    origin_env_kwargs = {'env_config_path': args.config, 'use_kinematic': args.kinematic, 'return_info': True,
                       'train': True, 'sim_cfg': rollout_sim_cfg, 'simnet_model_path': args.simnet_model_path}
    origin_env = make_vec_env("L5-CLE-v0", env_kwargs=origin_env_kwargs, n_envs=1,
                            vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "fork"})

    # define model
    model_path = "./models_" + date + "/" + str(steps) + ".pt"
    device = th.device("cpu")
    clip_schedule = get_linear_fn(args.clip_start_val, args.clip_end_val, args.clip_progress_ratio)
    if args.load is not None:
        model = PRED_12.load(model_path, rollout_env, device=device, clip_range=clip_schedule, learning_rate=args.lr)
    else:
        print("This is evaluation! Please enter model file path.")

    # saved_model_state_dict = th.load(model_path).state_dict()
    # model.load_state_dict(saved_model_state_dict)
    # model.cpu()
    # model = model.eval()
    th.set_grad_enabled(False)
    model.policy.set_training_mode(False)

    import gym
    from l5kit.visualization.visualizer.zarr_utils import episode_out_to_visualizer_scene_gym_cle
    from l5kit.visualization.visualizer.visualizer import visualize
    from bokeh.io import output_notebook, show, output_file, save
    from l5kit.data import MapAPI
    from l5kit.environment.gym_metric_set import CLEMetricSet
    env_config_path = args.config
    map_env = gym.make("L5-CLE-v0", env_config_path=env_config_path, sim_cfg=rollout_sim_cfg, \
                        use_kinematic=True, train=False, return_info=True)

    def rollout_episode(model, env, idx = 39):  # 18 25 29 50 60 65 75 95
        """Rollout a particular scene index and return the simulation output.

        :param model: the RL policy
        :param env: the gym environment
        :param idx: the scene index to be rolled out
        :return: the episode output of the rolled out scene
        """

        # Set the reset_scene_id to 'idx'
        # env.set_attr('reset_scene_id', idx)
        env.env_method('set_reset_id', idx)
        
        # Rollout step-by-step
        obs = env.reset()
        done = False

        n = 1
        while True:
            # data = get_frame_data(idx, n)
            # xy = data["target_positions"]
            # yaw = data["target_yaws"]
            # action = np.expand_dims(np.concatenate([xy, yaw], axis=-1)[0], axis=0)

            # # get resacle params and rescale the targets
            # rescale_action = env.get_attr('rescale_action')[0]
            # use_kinematic = env.get_attr('use_kinematic')[0]
            # if rescale_action:
            #     if use_kinematic:
            #         kin_rescale = env.get_attr('kin_rescale')[0]
            #         action[..., 0] = action[..., 0] / kin_rescale.steer_scale
            #         action[..., 1] = action[..., 1] / kin_rescale.acc_scale
            #     else:
            #         non_kin_rescale = env.get_attr('non_kin_rescale')[0]
            #         action[..., 0] = (action[..., 0] - non_kin_rescale.x_mu) / non_kin_rescale.x_scale
            #         action[..., 1] = (action[..., 1] - non_kin_rescale.y_mu) / non_kin_rescale.y_scale
            #         action[..., 2] = (action[..., 2] - non_kin_rescale.yaw_mu) / non_kin_rescale.yaw_scale

            # il_action = th.tensor(action)

            action, _ = model.predict(obs, deterministic=True)
            # obs, _ = model.policy.obs_to_tensor(obs)
            # action = model.policy.pred_traj(obs)[..., 0:3]  # pred_traj1
            # action = action.cpu().numpy().reshape((-1,) + model.policy.action_space.shape)

            # target_actions = th.cat((th.tensor(xy), th.tensor(yaw)), dim=-1).view(1, -1)
            # target_actions[..., 0] = (target_actions[..., 0] - non_kin_rescale.x_mu) / non_kin_rescale.x_scale
            # target_actions[..., 1] = (target_actions[..., 1] - non_kin_rescale.y_mu) / non_kin_rescale.y_scale
            # target_actions[..., 2] = (target_actions[..., 2] - non_kin_rescale.yaw_mu) / non_kin_rescale.yaw_scale

            # target_obs = {'image': data["image"]}
            # target_obs, _ = model.policy.obs_to_tensor(target_obs)
            # pred_actions = th.tensor(model.policy.pred_traj(target_obs))
            # criterion = th.nn.MSELoss(reduction="none")
            # loss = th.mean(criterion(pred_actions, target_actions))
            # print(loss)
            # obs_loss = th.mean(criterion(target_obs['image'], obs['image']))
            # print(obs_loss)

            
            # action = model.policy.pred_traj(target_obs)[..., 0:3]
            # action = action.cpu().numpy().reshape((-1,) + model.policy.action_space.shape)

            # action = rescale(origin_env, action)
            # action = un_rescale(env, action)
            obs, _, done, info = env.step(action)
            print(n)
            n += 1
            if done[0]:
                break

        # The episode outputs are present in the key "sim_outs"
        sim_outs = info[0]["sim_outs"]
        sim_out = info[0]["sim_outs"][0]
        return sim_out, sim_outs

    # Rollout one episode
    sim_out, sim_outs = rollout_episode(model, rollout_env, scene_id)

    metric_set = CLEMetricSet()
    metric_set.evaluate(sim_outs)
    from l5kit.environment.callbacks import L5KitEvalCallback
    # Aggregate metrics (ADE, FDE)
    ade_error, fde_error = L5KitEvalCallback.compute_ade_fde(metric_set)
    print("ade_error: ", ade_error, "fde_error: ", fde_error)

    # might change with different rasterizer
    map_API = map_env.dataset.rasterizer.sem_rast.mapAPI

    def visualize_outputs(sim_outs, map_API):
        for sim_out in sim_outs: # for each scene
            vis_in = episode_out_to_visualizer_scene_gym_cle(sim_out, map_API)
            save_path = Path(os.getcwd() + "/plots", "plot_" + date + "_PPO_" + steps + "_" + str(scene_id) + ".html")
            output_file(save_path)
            # show(visualize(sim_out.scene_id, vis_in))
            save(obj=visualize(sim_out.scene_id, vis_in), filename=save_path)

    output_notebook()
    visualize_outputs([sim_out], map_API)




