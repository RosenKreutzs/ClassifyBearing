### 1.主线逻辑(清晰)
- 目的：轴承故障一维信号分类；
- 途径：使用LSTM模型对于轴承故障一维信号进行分类；
- 预期效果：准确率达到95%；
- 要求：作为轴承故障诊断和分类的入门项目；
- 数据来源：凯斯西储大学轴承故障数据集([西储大学轴承数据中心对应该数据的网站](https://link.csdn.net/?target=https%3A%2F%2Fengineering.case.edu%2Fbearingdatacenter%2Fwelcome))；
### 2.anconda环境
- 打包环境

``conda env export > environment.yml``
- 加载环境

``conda env create -f environment.yml``

### 3.突出点
- 对于LSTM模型，一维对象的训练数据输入形状的堆叠处理方案是 先将一维对象[堆叠处理](LSTM/TRY.ipynb)为二维对象，直接用单个二维对象的形状为输入形状；
### 4.完成成事项
- ExplainationOfOriginDatas.md的内容填写
- 处理方法的扩展
