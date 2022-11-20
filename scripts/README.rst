L5Kit Scripts
===============================================================================
In this folder, we provide scripts to extract metadata from the L5Kit dataset.

Categorize Scenes
-------------------------------------------------------------------------------
The file :code:`categorize_scenes.py` categorizes the various scenes based on the turn taken by the ego.
Currently, there exist three categories: :code:`left`, :code:`right` and :code:`straight`. The script takes as
arguments:

* :code:`data_path`: the path to L5Kit dataset to categorize
* :code:`output`: the CSV file name for writing the metadata

2022.11.20 王宇霄：
多模态和价值评价的预测和规划
vect_multimodal_task_model.py : 多模态模型Module
vect_multimodal_task_learn.py : 训练文件
vect_multimodal_task_config.yaml : 配置文件
vect_multimodal_task_CLtest.py : 闭环测试文件
../tmpxxxx : 训练好的模型
../logxxxx : 训练日志文件
