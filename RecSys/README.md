# Recommend Syste项目说明
本项目文件结构如下：  
```
PageRank
│
├─实验结果
│    ├─result_best.txt
│    └─other_results.txt
│       ├─result_baseline.txt
│       ├─result_attribute_SVD.txt
│       ├─result_basic_SVD.txt
│       └─result_bias_SVD.txt
│ 
├─源码
│    ├─main.py
│    ├─grid_search.py
│    ├─dataProcessing.ipynb
│    ├─UserCF.py
|    ├─ItemCF.py
│    ├─SVD.py
│    └─utils.py
│ 
├─实验报告
│    └─实验报告.pdf
│ 
└─可执行文件
    ├─models
    │    └─baseline_estimator.pkl
    └─main.exe
```
项目分为4个部分：
1. 实验结果：包其中result_best.txt是最好的结果，other_results.txt是其他算法的结果，如baseline、基础SVD、带偏置的SVD、带属性的SVD，由于协同过滤效果较差同时运行时间过长，未给出结果。
2. 源码：包含了所有的源代码，其中main.py是主程序，grid_search.py是对SVD进行性能调优程序，dataProcessing.ipynb是数据处理程序，UserCF.py是基于用户的协同过滤算法，ItemCF.py是基于物品的协同过滤算法，SVD.py是基于SVD的推荐算法，utils.py是一些工具函数。
3. 实验报告：包含了实验报告的pdf文件。
4. 可执行文件：包含了main.exe和models文件夹，main.exe是可执行文件，models文件夹中存放了baseline模型，供可执行文件mian.exe读取。