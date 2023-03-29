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

config_path = os.getcwd() + "/pred_ppo_config.yaml"
os.environ.setdefault('CONFIG_PATH', config_path)

from pred_ppo import PRED_PPO
# from get_il_eval_data import dataset


date = "pretrain_withturn1" # "pretrain_withturn1" # "pretrain_withturn" # "pretrain_1000" # "2023-03-19_10-22"  # "2023-03-16_17-18"
steps = "15000" # "100000"  # 35000  # 30000
scene_id = 25  # 转弯场景：39x 93 红灯场景：12x 25 绿灯启动场景：13x 15 弯道场景：直道场景：40 58 65
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
    args.config = os.environ["CONFIG_PATH"]
    cfg = load_config_data(args.config)

    # train
    # model.learn(args.n_steps, callback=[checkpoint_callback, eval_callback])

    # visualize
    rollout_sim_cfg = SimulationConfigGym()
    rollout_sim_cfg.num_simulation_steps = None
    rollout_sim_cfg.use_agents_gt = (not args.simnet)
    rollout_env_kwargs = {'env_config_path': args.config, 'use_kinematic': args.kinematic, 'return_info': True,
                       'train': False, 'sim_cfg': rollout_sim_cfg, 'simnet_model_path': args.simnet_model_path,
                       'randomize_start': False}
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
    model_path = "./models_" + str(date) + "/" + str(steps) + ".pt"
    device = th.device("cpu")
    clip_schedule = get_linear_fn(args.clip_start_val, args.clip_end_val, args.clip_progress_ratio)
    if args.load is not None:
        model = PRED_PPO.load(model_path, rollout_env, device=device, clip_range=clip_schedule, learning_rate=args.lr)
    else:
        print("This is evaluation! Please enter model file path.")

    # # use pretrained 12 steps prediction model
    # from pred_12 import PRED_12
    # pre_model_path = "./models_pretrain_1000/" + str(100000) + ".pt"
    # pre_model = PRED_12.load(pre_model_path, rollout_env, device=device, clip_range=clip_schedule, learning_rate=args.lr)
    # pre_model_paras = pre_model.get_parameters()
    # model_paras = model.get_parameters()
    # print(pre_model_paras['policy'])
    # print(model_paras['policy'])
    # pre_model.policy.set_training_mode(False)

    # saved_model_state_dict = th.load(model_path).state_dict()
    # model.load_state_dict(saved_model_state_dict)
    # model.cpu()
    # model = model.eval()
    th.set_grad_enabled(False)
    model.policy.set_training_mode(False)


    # close loop evaluation:
    from l5kit.simulation.dataset import SimulationConfig
    from l5kit.simulation.unroll import ClosedLoopSimulator, PredPPOClosedLoopSimulator
    from l5kit.cle.closed_loop_evaluator import ClosedLoopEvaluator, EvaluationPlan
    from l5kit.cle.metrics import (CollisionFrontMetric, CollisionRearMetric, CollisionSideMetric,
                                DisplacementErrorL2Metric, DistanceToRefTrajectoryMetric,SimulatedDrivenMilesMetric,
                                ReplayDrivenMilesMetric,YawErrorMetric,YawErrorCAMetric,SimulatedVsRecordedEgoSpeedMetric)
    from l5kit.cle.validators import RangeValidator, ValidationCountingAggregator
    from l5kit.cle.composite_metrics import (PassedDrivenMilesCompositeMetric,DrivenMilesCompositeMetric,
                                            CompositeMetricAggregator)
    from prettytable import PrettyTable

    from l5kit.data import ChunkedDataset, LocalDataManager
    from l5kit.dataset import EgoDataset
    from l5kit.rasterization import build_rasterizer

    num_simulation_steps = 248

    # ==== DEFINE CLOSED-LOOP SIMULATION
    sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=True, disable_new_agents=True,  # use_agents_gt=False
                            distance_th_far=500, distance_th_close=50, num_simulation_steps=num_simulation_steps,
                            start_frame_index=0, show_info=True)

    # ===== INIT DATASET
    dm = LocalDataManager(None)
    eval_cfg = cfg["val_data_loader"]
    eval_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()
    # rasterisation
    rasterizer = build_rasterizer(cfg, dm)
    raster_size = cfg["raster_params"]["raster_size"][0]
    n_channels = rasterizer.num_channels()
    eval_dataset = EgoDataset(cfg, eval_zarr, rasterizer)
    
    print(eval_dataset)

    # sim_loop = MultimodalClosedLoopSimulator(sim_cfg, eval_dataset, device, model_ego=model4, model_agents=simulation_model)  # model_agents=None
    sim_loop =PredPPOClosedLoopSimulator(sim_cfg, eval_dataset, device, model_ego=model, model_agents=None)


    # ==== UNROLL
    # scenes_to_unroll = list(range(0, len(eval_zarr.scenes), len(eval_zarr.scenes)//num_scenes_to_unroll))
    # turns_scenes_to_unroll = [8, 23, 28, 39, 43, 57, 74, 77, 82, 93, 97]  # left and right turns
    test_list = [18,
                25, 29, 35, 50, 60, 65, 75, 95
                ]
    test_list = list([6,13,14,18,35,38,63,85,91,97])
    scenes_to_unroll_1 = list(range(0, 1))
    scenes_to_unroll_100 = list(range(0, 100))
    sim_outs = sim_loop.unroll(origin_env, test_list)


    # ==== CLOSE LOOP METRICS
    metrics_1 = [DisplacementErrorL2Metric(),
            DistanceToRefTrajectoryMetric(),
            CollisionFrontMetric(),
            CollisionRearMetric(),
            CollisionSideMetric(),
            ]

    validators_1 = [RangeValidator("displacement_error_l2", DisplacementErrorL2Metric, max_value=30),
                RangeValidator("distance_ref_trajectory", DistanceToRefTrajectoryMetric, max_value=4),
                RangeValidator("collision_front", CollisionFrontMetric, max_value=0),
                RangeValidator("collision_rear", CollisionRearMetric, max_value=0),
                RangeValidator("collision_side", CollisionSideMetric, max_value=0),
                ]

    intervention_validators_1 = ["displacement_error_l2",
                            "distance_ref_trajectory",
                            "collision_front",
                            "collision_rear",
                            "collision_side",
                            ]

    metrics_2 = [
            SimulatedDrivenMilesMetric(),
            ReplayDrivenMilesMetric(),
            SimulatedVsRecordedEgoSpeedMetric(),
            ]

    validators_2 = [
                RangeValidator("simulated_driven_miles", SimulatedDrivenMilesMetric, max_value=0),
                RangeValidator("replay_driven_miles", ReplayDrivenMilesMetric, max_value=0),
                RangeValidator("simulated_minus_recorded_ego_speed", SimulatedVsRecordedEgoSpeedMetric, max_value=0),
                ]

    intervention_validators_2 = [
                            "simulated_driven_miles",
                            "replay_driven_miles",
                                "simulated_minus_recorded_ego_speed",
                            ]


    composite_metrics=[
    PassedDrivenMilesCompositeMetric(composite_metric_name='PassedDrivenMilesCompositeMetric',intervention_validators=intervention_validators_2),
    DrivenMilesCompositeMetric(composite_metric_name='DrivenMilesCompositeMetric')
    ]

    cle_evaluator_1 = ClosedLoopEvaluator(EvaluationPlan(metrics=metrics_1,
                                                    validators=validators_1,
                                                    composite_metrics=[],
                                                    intervention_validators=intervention_validators_1))

    cle_evaluator_2 = ClosedLoopEvaluator(EvaluationPlan(metrics=metrics_2,
                                                    validators=validators_2,
                                                    composite_metrics=composite_metrics,
                                                    intervention_validators=intervention_validators_2))
    composite_metric_aggregator=CompositeMetricAggregator()


    # ==== QUANTITATIVE EVALUATION
    cle_evaluator_1.evaluate(sim_outs)
    validation_results_1 = cle_evaluator_1.validation_results()
    scene_id = 0
    for _, validator_dict in validation_results_1.items():
        for validator_name, validator_output in validator_dict.items():
            if not validator_output.is_valid_scene:
                print(scene_id)
                print(validator_name)
        scene_id += 1
    agg = ValidationCountingAggregator().aggregate(validation_results_1)

    cle_evaluator_2.evaluate(sim_outs)
    validation_results_2 = cle_evaluator_2.validation_results()
    agg_2 = ValidationCountingAggregator().aggregate(validation_results_2)
    composite_metrics_results_scenes=cle_evaluator_2.composite_metric_results()
    composite_metric_agg=composite_metric_aggregator.aggregate(composite_metrics_results_scenes)



    fields = ["metric", "value"]
    table = PrettyTable(field_names=fields)

    values = []
    names = []

    # copute ades and fdes
    ades = []
    for metric_result in cle_evaluator_1.scene_metric_results.values():
        ade = th.mean(metric_result["displacement_error_l2"])
        ades.append(ade)
    ades = sum(ades) / len(ades)
    table.add_row(["ade", ades])
    values.append(ades)
    names.append("ade")

    fdes = []
    for metric_result in cle_evaluator_1.scene_metric_results.values():
        fde = metric_result["displacement_error_l2"][-1]
        fdes.append(fde)
    fdes = sum(fdes) / len(fdes)
    table.add_row(["fde", fdes])
    values.append(fdes)
    names.append("fde")

    cle_evaluator_1.reset()
    cle_evaluator_2.reset()

    for metric_name in agg:
        table.add_row([metric_name, agg[metric_name].item()])
        values.append(agg[metric_name].item())
        names.append(metric_name)

    for metric_name in agg_2:
        table.add_row([metric_name, agg_2[metric_name].item()])
        values.append(agg_2[metric_name].item())
        names.append(metric_name)

    for metric_name in composite_metric_agg:
        table.add_row([metric_name, composite_metric_agg[metric_name].item()])
        values.append(composite_metric_agg[metric_name].item())
        names.append(metric_name)

    print(table)



    from l5kit.environment.callbacks import L5KitEvalCallback
    from l5kit.visualization.visualizer.zarr_utils import simulation_out_to_visualizer_scene
    # # Aggregate metrics (ADE, FDE)
    # ade_error, fde_error = L5KitEvalCallback.compute_ade_fde(metric_set)
    # print("ade_error: ", ade_error, "fde_error: ", fde_error)

    import gym
    from bokeh.io import output_notebook, show, output_file, save
    from l5kit.visualization.visualizer.visualizer import visualize

    # might change with different rasterizer
    env_config_path = args.config
    map_env = gym.make("L5-CLE-v0", env_config_path=env_config_path, sim_cfg=rollout_sim_cfg, \
                        use_kinematic=True, train=False, return_info=True)
    map_API = map_env.dataset.rasterizer.sem_rast.mapAPI

    def visualize_outputs(sim_outs, map_API):
        id = 0
        for sim_out in sim_outs: # for each scene
            # if id != scene_id:
            #     id += 1
            #     continue
            vis_in = simulation_out_to_visualizer_scene(sim_out, map_API)
            save_path = Path(os.getcwd() + "/plots", "plot_" + date + "_ilPPO_" + steps + "_" + str(sim_out.scene_id) + ".html")
            id += 1
            print("plot results")
            print(save_path)
            output_file(save_path)
            # show(visualize(sim_out.scene_id, vis_in))
            save(obj=visualize(sim_out.scene_id, vis_in), filename=save_path)

    output_notebook()
    visualize_outputs(sim_outs, map_API)