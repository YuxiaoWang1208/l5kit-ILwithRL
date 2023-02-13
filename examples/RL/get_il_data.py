from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
import torch as th
from torch.utils.data import DataLoader


device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

# IL env config
dm = LocalDataManager(None)
config_path = './gym_config.yaml'
cfg = load_config_data(config_path)
batch_size = 64

# rasterisation
rasterizer = build_rasterizer(cfg, dm)
raster_size = cfg["raster_params"]["raster_size"][0]
n_channels = rasterizer.num_channels()

# load dataset for imitation learning
loader_key = cfg["train_data_loader"]["key"]
dataset_zarr = ChunkedDataset(dm.require(loader_key)).open()
dataset = EgoDataset(cfg, dataset_zarr, rasterizer)
dataset = dataset.get_scene_dataset(scene_index=39)  # 单场景数据集
train_cfg = cfg["train_data_loader"]
dataloader = DataLoader(dataset, shuffle=train_cfg["shuffle"], batch_size=int(batch_size/4),
                                num_workers=train_cfg["num_workers"])
print("Imitation data has been loaded.")
data_it = iter(dataloader)


def get_data():
    global data_it
    try:
        il_data_buffer = next(data_it)
    except StopIteration:
        data_it = iter(dataloader)
        il_data_buffer = next(data_it)
    # to cuda device                  
    il_data_buffer = {k: v.to(device) for k, v in il_data_buffer.items()}

    return il_data_buffer
