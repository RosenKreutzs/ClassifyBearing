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

- 对于LSTM模型，一维对象的训练数据输入形状的堆叠处理方案是 先将一维对象[堆叠处理](./LSTM/TRY.ipynb)为二维对象，直接用单个二维对象的形状为输入形状；
- [KANs的个人理解](./KANs/EXPLAINATION.md);
- [连续小波变换](./VGG/EXPLAINATION_CWT.md)
- [短时傅里叶变换](./VGG/EXPLAINATION_STFT.md)
- [快速傅里叶变换](./VGG/EXPLAINATION_FFT.md)
- [VGG16的个人理解](./VGG/EXPLAINATION_CNN.md)
- [CNN的个人理解](./VGG/EXPLAINATION_CNN.md)
- [RNN的个人理解](./LSTM/EXPLAINATION_RNN.md)

### 4.未完成事项

- 经验模态分解;
- 变分模态分解；
- XGBoost的个人理解；
- svm的个人理解；
- 随机森林的个人理解；
- transformer的个人理解；(Cross attention机制)
- GRU的个人理解；(BiGRU的个人理解)
- KANs模型的外迁；
- 未连接WPS笔记；
