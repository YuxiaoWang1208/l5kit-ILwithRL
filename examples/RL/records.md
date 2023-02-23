记录实验设置：
    sample数据集39号场景：
        2023-02-07_16-05：纯PPO，loss = policy_loss + 0.0 * entropy_loss + 0.5 * value_loss；不能处理路口转弯。
        2023-02-13_10-09：PPO但读取了IL数据，loss = 0.0 + policy_loss + 0.0 * entropy_loss + 0.005 * value_loss；停在300k，路口转弯效果不好。
        2023-02-13_10-46：一步IL(MSE)+PPO，loss = 1.0 * il_loss + policy_loss + 0.0 * entropy_loss + 0.005 * value_loss；有一定效果，不过应该不如单场景纯IL，依旧不能很好地在路口转弯行驶。
        2023-02-13_15-59：一步IL(MSE)，loss = 1.0 * il_loss + 0.0*(policy_loss + 0.0 * entropy_loss + 0.005 * value_loss)；为了验证是不是因为IL是对actions_mean训练而不是采样后的actions而导致效果不好；或者ILloss本身存在问题？
        2023-02-13_16-51:一步IL(MSE)，loss相同，但是是对采样后的最终actions计算il_loss，还是不行，loss下降但是reward和闭环测试表现差，会不会是action不一样格式的问题？经验证，直接用target actions控制也不能跑好，应该就是用于il训练的target不对。
        
        后再经调试证实，是由于l5_env.py line:291 action = self._rescale_action(action)考虑到PPO[-1,1]的动作空间限制，在环境内仿真时擅自进行了额外的rescale，导致直接从数据集读取的target不再正确。由于不方便从SubprocVecEnv环境读取rescale参数，因此只能考虑放弃rescale并适当放宽PPO action_space。※实际上，[-1,1]的完全够用，不会clip掉取值。以下是非rescale的：

        2023-02-14_15-50：以action_mean模仿，std下降，一步IL(MSE)+PPO，loss = 1.0 * il_loss + policy_loss + 0.0 * entropy_loss + 0.005 * value_loss；效果还是不太好，直接取消rescale会对PPO造成较大影响，不好训练，必须想办法得到rescale参数从而对il的target进行修改！
        2023-02-14_16-10：以action模仿，std下降，一步IL(MSE)，loss = 1.0 * il_loss + 0.0*(policy_loss + 0.0 * entropy_loss + 0.005 * value_loss)；同上。

        2023-02-15_09-28：rescale版本，以action_mean模仿，一步IL(MSE)+PPO，loss = 1.0 * il_loss + policy_loss + 0.0 * entropy_loss + 0.005 * value_loss；由于下一条所述的纯模仿都没能实现，混合效果自然无法提升。
        2023-02-15_09-30：rescale版本，以action模仿，一步IL(MSE)，loss = 1.0 * il_loss + 0.0*(policy_loss + 0.0 * entropy_loss + 0.005 * value_loss)；还是不能实现路口转弯，可能是由于一步模仿不容易学习？
        2023-02-15_16-01：rescale版本，以action模仿，deterministic=True，一步IL(MSE)，loss = 10.0 * il_loss + 0.0*(policy_loss + 0.0 * entropy_loss + 0.005 * value_loss)；
        2023-02-15_18-35：rescale版本，以action模仿，加扰动，一步IL(MSE)，loss = 10.0 * il_loss + 0.0*(policy_loss + 0.0 * entropy_loss + 0.005 * value_loss)；

        2023-02-15_19-50：rescale版本，以action模仿，不加扰动，一步IL(MSE)，50 纯IL_n_epochs + 50 混合RL_n_epochs，相当于增加了IL的比重，混合loss = 10.0 * il_loss + 0.0*(policy_loss + 0.0 * entropy_loss + 0.005 * value_loss)；有没有一种可能，是单独39号场景不太好训练？从闭环测试中可以看出，智能车在转弯的那几帧action loss较大，但是在前后由于都是跑直线action loss其实不大，因此或许是在这样的图像观测和单场景训练集下，学会在那里转弯真的太难了？对于这个39号场景的246帧构成的数据集只有30帧是路口转弯...解决办法：1.换一个稍微简单的弯道场景再验证一下！ 2.直接用原来的轨迹规划网络试一次39号场景。
        
        经过验证，不是网络模型的问题，是规划步数的问题！！！12步能够学到预测未来的轨迹，因此比1步学习的更善于转弯。

        2023-02-20_16-59：12步轨迹预测模仿训练+模仿奖励强化学习训练，256*4步collects+128步eps_length；loss波动大需要调整分开loss记录，action net, pred net, 其他RL loss各自记录；此外训练结构也不够好，40次RL+60次Pred训练突变太大。

        更改了循环训练结构为10epochs每个epoch一次RL一次IL，并换用预训练的resnet50作为Feature Extractor。此前il loss波动不下降其实是因为pred_net添加的参数没有被optimizer纳入考虑！！！2023-02-21_14-03

        2023-02-22_12-19，2023-02-22_13-10，训练时用[batch_size, future_num_frames*3]的方式乘上availabilities求loss，并且rescale相当于只对第1步的进行，后面11步轨迹训练时不rescale target。
        2023-02-22_15-17，直接只用resnet50网络的效果可以，原本的action net涉及的概率采样可能有点问题。
        2023-02-22_17-53，为了验证上一条，在resnet50网络的基础上把其他几层网络加上但是不用action dist采样，并且降低了学习率为3e-4，看看效果。效果很好，证明多加几层网络不影响轨迹学习，而且由于调低了学习率还使学习变快，750个updates就有沿着车道开的相当不错的效果！！！下面只需要验证action dist的部分是直接去掉还是添加logit的loss。

        2023-02-23_16-44，12步pred+PPO，deterministic=True，可能是由于Value Net 和 Pred共用了特征提取网络？还是由于Value Net训练频率太低没收敛，总之结果是先经历了一个变差的过程。