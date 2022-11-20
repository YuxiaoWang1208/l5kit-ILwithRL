from l5kit.simulation.unroll import *


class MultimodalClosedLoopSimulator:
    def __init__(self, sim_cfg: SimulationConfig, dataset: EgoDataset,
                 device: torch.device,
                 model_ego: Optional[torch.nn.Module] = None,
                 model_agents: Optional[torch.nn.Module] = None,
                 keys_to_exclude: Tuple[str] = ("image",),
                 mode: int = ClosedLoopSimulatorModes.L5KIT):
        """
        Create a simulation loop object capable of unrolling ego and agents
        :param sim_cfg: configuration for unroll
        :param dataset: EgoDataset used while unrolling
        :param device: a torch device. Inference will be performed here
        :param model_ego: the model to be used for ego
        :param model_agents: the model to be used for agents
        :param keys_to_exclude: keys to exclude from input/output (e.g. huge blobs)
        :param mode: the framework that uses the closed loop simulator
        """
        self.sim_cfg = sim_cfg
        if not sim_cfg.use_ego_gt and model_ego is None and mode == ClosedLoopSimulatorModes.L5KIT:
            raise ValueError("ego model should not be None when simulating ego")
        if not sim_cfg.use_agents_gt and model_agents is None and mode == ClosedLoopSimulatorModes.L5KIT:
            raise ValueError("agents model should not be None when simulating agent")
        if sim_cfg.use_ego_gt and mode == ClosedLoopSimulatorModes.GYM:
            raise ValueError("ego has to be simulated when using gym environment")

        self.model_ego = torch.nn.Sequential().to(device) if model_ego is None else model_ego.to(device)
        self.model_agents = torch.nn.Sequential().to(device) if model_agents is None else model_agents.to(device)

        self.device = device
        self.dataset = dataset

        self.keys_to_exclude = set(keys_to_exclude)


    def unroll(self, scene_indices: List[int]) -> List[SimulationOutput]:
        """
        Simulate the dataset for the given scene indices
        :param scene_indices: the scene indices we want to simulate
        :return: the simulated dataset
        """
        sim_dataset = SimulationDataset.from_dataset_indices(self.dataset, scene_indices, self.sim_cfg)

        agents_ins_outs: DefaultDict[int, List[List[UnrollInputOutput]]] = defaultdict(list)
        ego_ins_outs: DefaultDict[int, List[UnrollInputOutput]] = defaultdict(list)

        for frame_index in tqdm(range(len(sim_dataset)), disable=not self.sim_cfg.show_info):
        # for frame_index in tqdm(range(10), disable=not self.sim_cfg.show_info):
            next_frame_index = frame_index + 1
            should_update = next_frame_index != len(sim_dataset)

            # AGENTS
            if not self.sim_cfg.use_agents_gt:
                agents_input = sim_dataset.rasterise_agents_frame_batch(frame_index)
                if len(agents_input):  # agents may not be available
                    agents_input_dict = default_collate(list(agents_input.values()))
                    agents_output_dict = self.model_agents.inference(move_to_device(agents_input_dict, self.device))

                    # for update we need everything as numpy
                    agents_input_dict = move_to_numpy(agents_input_dict)
                    agents_output_dict = move_to_numpy(agents_output_dict)

                    if should_update:
                        self.update_agents(sim_dataset, next_frame_index, agents_input_dict, agents_output_dict)

                    # update input and output buffers
                    agents_frame_in_out = self.get_agents_in_out(agents_input_dict, agents_output_dict,
                                                                 self.keys_to_exclude)
                    for scene_idx in scene_indices:
                        agents_ins_outs[scene_idx].append(agents_frame_in_out.get(scene_idx, []))

            # EGO
            if not self.sim_cfg.use_ego_gt:
                ego_input = sim_dataset.rasterise_frame_batch(frame_index)
                ego_input_dict = default_collate(ego_input)
                ego_output_dict = self.model_ego.inference(move_to_device(ego_input_dict, self.device))
                # ego_output_dict = self.model_ego(move_to_device(ego_input_dict, self.device))

                if not ("positions" in ego_output_dict.keys() and "yaws" in ego_output_dict.keys()):
                    ego_output_dict.update(dict(positions=ego_output_dict["ego_positions"], yaws=ego_output_dict["ego_yaws"]))

                ego_input_dict = move_to_numpy(ego_input_dict)
                ego_output_dict = move_to_numpy(ego_output_dict)
                # print(ego_output_dict["ego_positions_all"].shape)

                if should_update:
                    self.update_ego(sim_dataset, next_frame_index, ego_input_dict, ego_output_dict)

                ego_frame_in_out = self.get_ego_in_out(ego_input_dict, ego_output_dict, self.keys_to_exclude)
                for scene_idx in scene_indices:
                    ego_ins_outs[scene_idx].append(ego_frame_in_out[scene_idx])

        simulated_outputs: List[SimulationOutput] = []
        for scene_idx in scene_indices:
            simulated_outputs.append(SimulationOutput(scene_idx, sim_dataset, ego_ins_outs, agents_ins_outs))
        return simulated_outputs


    @staticmethod
    def update_agents(dataset: SimulationDataset, frame_idx: int, input_dict: Dict[str, np.ndarray],
                      output_dict: Dict[str, np.ndarray]) -> None:
        """Update the agents in frame_idx (across scenes) using agents_output_dict

        :param dataset: the simulation dataset
        :param frame_idx: index of the frame to modify
        :param input_dict: the input to the agent model
        :param output_dict: the output of the agent model
        :return:
        """

        agents_update_dict: Dict[Tuple[int, int], np.ndarray] = {}

        world_from_agent = input_dict["world_from_agent"]
        yaw = input_dict["yaw"]
        pred_trs = transform_points(output_dict["positions"][:, :1], world_from_agent)[:, 0]
        pred_yaws = yaw + output_dict["yaws"][:, 0, 0]

        next_agents = np.zeros(len(yaw), dtype=AGENT_DTYPE)
        next_agents["centroid"] = pred_trs
        next_agents["yaw"] = pred_yaws
        next_agents["track_id"] = input_dict["track_id"]
        next_agents["extent"] = input_dict["extent"]

        next_agents["label_probabilities"][:, PERCEPTION_LABEL_TO_INDEX["PERCEPTION_LABEL_CAR"]] = 1

        for scene_idx, next_agent in zip(input_dict["scene_index"], next_agents):
            agents_update_dict[(scene_idx, next_agent["track_id"])] = np.expand_dims(next_agent, 0)
        dataset.set_agents(frame_idx, agents_update_dict)


    @staticmethod
    def get_agents_in_out(input_dict: Dict[str, np.ndarray],
                          output_dict: Dict[str, np.ndarray],
                          keys_to_exclude: Optional[Set[str]] = None) -> Dict[int, List[UnrollInputOutput]]:
        """Get all agents inputs and outputs as a dict mapping scene index to a list of UnrollInputOutput

        :param input_dict: all agent model inputs (across scenes)
        :param output_dict: all agent model outputs (across scenes)
        :param keys_to_exclude: if to drop keys from input/output (e.g. huge blobs)
        :return: the dict mapping scene index to a list UnrollInputOutput. Some scenes may be missing
        """
        key_required = {"track_id", "scene_index"}
        if len(key_required.intersection(input_dict.keys())) != len(key_required):
            raise ValueError(f"track_id and scene_index not found in keys {input_dict.keys()}")

        keys_to_exclude = keys_to_exclude if keys_to_exclude is not None else set()
        if len(key_required.intersection(keys_to_exclude)) != 0:
            raise ValueError(f"can't drop required keys: {keys_to_exclude}")

        ret_dict = defaultdict(list)
        for idx_agent in range(len(input_dict["track_id"])):
            agent_in = {k: v[idx_agent] for k, v in input_dict.items() if k not in keys_to_exclude}
            agent_out = {k: v[idx_agent] for k, v in output_dict.items() if k not in keys_to_exclude}
            ret_dict[agent_in["scene_index"]].append(UnrollInputOutput(track_id=agent_in["track_id"],
                                                                       inputs=agent_in,
                                                                       outputs=agent_out))
        return dict(ret_dict)


    @staticmethod
    def get_ego_in_out(input_dict: Dict[str, np.ndarray],
                       output_dict: Dict[str, np.ndarray],
                       keys_to_exclude: Optional[Set[str]] = None) -> Dict[int, UnrollInputOutput]:
        """Get all ego inputs and outputs as a dict mapping scene index to a single UnrollInputOutput

        :param input_dict: all ego model inputs (across scenes)
        :param output_dict: all ego model outputs (across scenes)
        :param keys_to_exclude: if to drop keys from input/output (e.g. huge blobs)
        :return: the dict mapping scene index to a single UnrollInputOutput.
        """
        key_required = {"track_id", "scene_index"}
        if len(key_required.intersection(input_dict.keys())) != len(key_required):
            raise ValueError(f"track_id and scene_index not found in keys {input_dict.keys()}")

        keys_to_exclude = keys_to_exclude if keys_to_exclude is not None else set()
        if len(key_required.intersection(keys_to_exclude)) != 0:
            raise ValueError(f"can't drop required keys: {keys_to_exclude}")

        ret_dict = {}
        scene_indices = input_dict["scene_index"]
        if len(np.unique(scene_indices)) != len(scene_indices):
            raise ValueError(f"repeated scene_index for ego! {scene_indices}")

        for idx_ego in range(len(scene_indices)):
            ego_in = {k: v[idx_ego] for k, v in input_dict.items() if k not in keys_to_exclude}
            ego_out = {k: v[idx_ego] for k, v in output_dict.items() if k not in keys_to_exclude}
            ret_dict[ego_in["scene_index"]] = UnrollInputOutput(track_id=ego_in["track_id"],
                                                                inputs=ego_in,
                                                                outputs=ego_out)
        return ret_dict


    @staticmethod
    def update_ego(dataset: SimulationDataset, frame_idx: int, input_dict: Dict[str, np.ndarray],
                   output_dict: Dict[str, np.ndarray]) -> None:
        """Update ego across scenes for the given frame index.

        :param dataset: The simulation dataset
        :param frame_idx: index of the frame to modify
        :param input_dict: the input to the ego model
        :param output_dict: the output of the ego model
        :return:
        """
        world_from_agent = input_dict["world_from_agent"]
        yaw = input_dict["yaw"]
        pred_trs = transform_points(output_dict["positions"][:, :1], world_from_agent)
        pred_yaws = np.expand_dims(yaw, -1) + output_dict["yaws"][:, :1, 0]

        dataset.set_ego(frame_idx, 0, pred_trs, pred_yaws)

