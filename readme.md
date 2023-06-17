# ACGAN-Mnist
## 文件说明
* aigc_mn.py：实现接口类AiGcMn，类的初始化函数中完成模型加载等初始化工作。

  接口类AiGcMn提供一个接口函数generate，该函数的参数是一个整数型n维tensor（n是batch的大小，每个整数在0~9范围内，代表需要生成的数字），输出是`n*1*28*28`的tensor（n是batch的大小，每个`1*28*28`的tensor表示随机生成的数字图像），支持在CPU和GPU上运行随机产生输出图像。

  若直接运行`aigc_mn.py`，则会读取模型后随机生成10个不同的数字，将其分别保存在`./results`文件夹中。

* models：模型文件夹，存有训练好的模型。

* results：结果文件夹，保存生成的图片。

* train：训练部分，保存用于训练模型的代码与数据。

  运行其中`main.py`即可进行训练，默认进行25轮训练，训练结果会产生图片保存在`./train/results`中，产生的模型保存在`./train/models`文件夹中。

* report：报告。
