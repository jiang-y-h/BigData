# PageRank
本项目文件结构如下：  
```
PageRank
│
├─results
│    ├─result.txt
│    ├─result_basic.txt
│    ├─result_block.txt
│    ├─result_block_stripe.txt
│    ├─result_sparse.txt
│
└─src
│    ├─main.py
│    ├─main.spec
│    ├─pagerank.py
│    ├─page_rank.ipynb
│    ├─utils.py
│
├─Data.txt
└─main.exe
```
其中main.exe是打包好的可执行文件，可以直接运行。  
运行结果会输出到results文件夹下的result.txt

## main.exe运行过程：
1. 选择数据输入，默认为同一目录下的Data.txt文件，显示“读取数据成功”即可完成数据的输入。  
![alt text](img\image-1.png)
2. 输入阻尼系数$\beta$，默认为0.85  
![alt text](img\image-2.png)
3. 然后可以选择使用哪一个PageRank算法，有Basic基础算法，Sparse稀疏矩阵的优化算法，Block分块算法，Block+Stripe分块条带算法。Block+Stripe+Parallel的分块条带并行算法。  
![alt text](img\image-3.png)
4. 选择完算法后，等待程序运行一段时间，即可得到每一轮迭代的误差，当误差低于1e-6时，迭代结束，输出前100个节点，以及其对应的PageRank值。
同时会将其与调用networkx库运行的结果进行比较，测试代码是否排序正确，以及得到的PageRank值误差。  
![alt text](img\image-4.png)
5. 最终将结果保存到result/result.txt
