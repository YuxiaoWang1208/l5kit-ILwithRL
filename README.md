## 2022.11.20 王宇霄：
### 多模态和价值评价的预测和规划
scripts/vect_multimodal_task_model.py : 多模态模型Module/n
scripts/vect_multimodal_task_learn.py : 训练文件/n
scripts/vect_multimodal_task_config.yaml : 配置文件/n
scripts/vect_multimodal_task_CLtest.py : 闭环测试文件/n
tmpxxxx : 训练好的模型/n
logxxxx : 训练日志文件


Welcome to **L5Kit**. L5Kit is a Python library with functionality for the development and training of *learned prediction, planning and simulation* models for autonomous driving applications.

[Click here for documentation](https://woven-planet.github.io/l5kit)

## Model-based offline reinforcement learning

### Run imitation learning and motion prediction (model learning)

```bash
python scripts/offline_rl.py --imitate_loss_weight 1 --pred_loss_weight 1 --cuda_id 1
```

**Trained models**

该模型同时输出两个向量，模仿学习策略输出的自车轨迹(12个点, 维度3*12)，对其他障碍物预测(30辆车，每辆车12个点，维度30*12*3)

`/mnt/share_disk/user/public/l5kit/model/scene_13-il_weight_1.0-pred_weight_0.5-iter_0124000.pt`

该模型在scene 13训练，该场景可由以下代码得到

```python
train_zarr = ChunkedDataset(dataset_path).open()
vectorizer = build_vectorizer(cfg, dm)
train_dataset = EgoDatasetVectorized(cfg, train_zarr, vectorizer)

val_dataset = train_dataset.get_scene_dataset(13)
```


