### 泰坦尼克生还预测
* 编写环境：python3.9 package on conda

##### 文件说明
* RandomForest.pkl 由TrainModelExport.py程序导出的训练模型
* test.csv 预测数据
* train.csv 原始训练数据  
* Titanic.ipynb 从其他博主那下载下来的，解题思路
* TitanicFeatureSelection.py 特征贡献度
* TitanicLinearRegression.py 利用线性回归算法训练预测
* TitanicLogisticRegression.py 利用线性逻辑算法训练预测
* TitanicRandomForest.py 利用随机森林算法训练预测
* TitanicIntegrate.py 多种算法聚合训练预测
* TrainModelExport.py 训练模型导出
* TrainModelImport.py 训练模型导入

##### Run
我写的时候使用的是python3.9 

依赖包由sklearn joblib numpy 可以自行百度安装

又一个坑就是M1处理器的MAC电脑在装pip包的时候有很多保错，可以百度搜索安装Conda来托管python

##### Video
https://www.bilibili.com/video/av840265491