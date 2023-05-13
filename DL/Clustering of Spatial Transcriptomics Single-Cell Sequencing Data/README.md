# README

### 项目结构

- model文件中存储了所有python代码
- save_model文件中保存了AE训练出的模型结果
- spca_dat文件中是12个h5数据集

<img src="C:\Users\Shawn\Desktop\Typora\DL大作业.assets\image-20230205204654639.png" alt="image-20230205204654639" style="zoom:50%;" />

### 代码测试说明

请进入model文件中的k_means.py中的主函数部分

<img src="C:\Users\Shawn\AppData\Roaming\Typora\typora-user-images\image-20230205222421963.png" alt="image-20230205222421963" style="zoom:50%;" />

指定读取的数据集、读取已训练好的模型

如果需要重新训练模型，将参数train_model设为True

如果不用AE提取特征直接进行聚类，设置参数AE=False