# SJTU-M3DV-Medical-3D-Voxel-Classification
This repository is the solution of EE369 Machine Learning 2019 Autumn Class project.
## 基于卷积神经网络的3D医学图像分类

### 问题描述 —— 肺部结节的分类问题
(1) 数据集<br>
a. 训练集-465个样本(标签已知)<br>
b. 测试集-117个样本(标签未知)<br>
(PS: 每个样本都包括voxel、seg、label三部分。voxel和seg大小是100 x 100 x 100)<br>
<br>
(2) 任务<br>
预测测试集样本的标签，以AUC值作为评判标准。<br>

### 方法简述
使用[densesharpe](https://github.com/duducheng/DenseSharp)的神经网络结构并结合裁剪、旋转、mixup的数据增强方法，最终把两个使用不同验证集训练的模型集成，取其预测结果的平均值。

### 运行环境
* Keras2.2.4
* Tensorflow1.14.0（GPU版本）

### 运行方法
已经将训练好的权重放在了./tmp/test的路径下，直接使用即可。<br>
预测测试集的样本标签：运行`test.py`文件即可得到Submission.csv<br>

注：本课程项目代码使用的参考代码是[densesharpe](https://github.com/duducheng/DenseSharp)(作者：[@duducheng](https://github.com/duducheng))
