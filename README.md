### 1.主线逻辑(清晰)

- 目的：轴承故障一维信号分类；
- 途径：使用LSTM模型对于轴承故障一维信号进行分类；
- 预期效果：准确率达到95%；
- 要求：作为轴承故障诊断和分类的入门项目；
- 数据来源：凯斯西储大学轴承故障数据集([西储大学轴承数据中心对应该数据的网站](https://link.csdn.net/?target=https%3A%2F%2Fengineering.case.edu%2Fbearingdatacenter%2Fwelcome))；
- ADS文件夹的数据太多了，上传过于麻烦，存储在Kaggle的服务器中；
### 2.anconda环境

- 打包环境

``conda env export > environment.yml``

- 加载环境

``conda env create -f environment.yml``

### 3.突出点

- 对于LSTM模型，一维对象的训练数据输入形状的堆叠处理方案是 先将一维对象[堆叠处理](./LSTM/TRY.ipynb)为二维对象，直接用单个二维对象的形状为输入形状；
- [连续小波变换](./VGG/EXPLAINATION_CWT.md)
- [短时傅里叶变换](./VGG/EXPLAINATION_STFT.md)
- [快速傅里叶变换](./VGG/EXPLAINATION_FFT.md)
- [VGG16的个人理解](./VGG/EXPLAINATION_CNN.md)
- [CNN的个人理解](./VGG/EXPLAINATION_CNN.md)
- [RNN的个人理解](./LSTM/EXPLAINATION_RNN.md)
- [LSTM的个人理解](./LSTM/EXPLAINATION_LSTM.md)
- [GRU的个人理解](./LSTM/EXPLAINATION_GRU.md)
- [经验模态分解](./LSTM/EXPLAINATION_EMD.md)
- [变分模态分解](./LSTM/EXPLAINATION_VMD.md)
- [XGBoost的个人理解](./XGBoost/EXPLAINATION_XGBoost.md)
- [RandomForest的个人理解](./XGBoost/EXPLAINATION_RandomForest.md)
- [svm的个人理解](./XGBoost/EXPLAINATION_SVM.md)
- [transformer的个人理解](./EXPLAINATION_Transformer.md)
### 4.未完成事项
