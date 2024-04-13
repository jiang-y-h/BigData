import numpy as np
import time

# 读数据
def read_data():
    file = open('Data.txt', 'r')
    graph=[] 
    node_set=set()
    for line in file:
        data=line.split()
        edge=(int(data[0]),int(data[1]))  # 以tuple存储两个点/边
        node_set.add(edge[0])
        node_set.add(edge[1])
        graph.append(edge)

    node_num=len(node_set) # 点的个数
    return graph, node_num, node_set

G, node_num, node_set = read_data()
scores = np.ones((node_num))/node_num  # 存储在disk
new_scores = np.zeros((node_num))  # 分块存储在ram
block_size = 2000
beta = 0.85

def get_stripes(G, node_num, block_size):
    block_num = node_num//block_size
    remain = node_num%block_size
    if remain != 0:  # +1是因为最后一个stripe也得存
        block_num += 1

    stripes = [ {} for _ in range(block_num)]  # [0,1,2,...,block_num-1]
    length = [0 for _ in range(node_num)]

    # 初始化稀疏矩阵
    for edge in G:
        to_node = edge[1]-1
        from_node = edge[0]-1
        index = to_node//block_size  # dest所在块的编号
        if from_node not in stripes[index].keys():
            # 将from_node加入stripes
            stripes[index][from_node] = []
        stripes[index][from_node].append(to_node)  # 将to_node加入stripes
        length[from_node] += 1  # 记录每个节点的出度
    return stripes, length

stripes, length = get_stripes(G, node_num, block_size)

def deal_dead_end(stripes, length, node_num, block_size):
    block_num = node_num//block_size
    remain = node_num%block_size
    if remain != 0:
        block_num += 1
    for i in range(node_num):
        # 没有出度则为dead-end
        if length[i] == 0:
            length[i] = node_num
            for j in range(block_num):
                stripes[j][i] = [k for k in range(j*block_size, min((j+1)*block_size, node_num))]
    return stripes, length


def power_interation_block_stripe(stripes, length, node_num, block_size, beta):
    global scores, new_scores    
    block_num = node_num//block_size
    remain = node_num%block_size
    if remain != 0:  # +1是因为最后一个stripe也得存
        block_num += 1
    end_block_index = block_num-1
        
    e = 1  # 两次迭代之间的误差
    interation_num = 0
    while e > 1e-3:
        e = 0
        # 每次处理一块
        for i in range(end_block_index):
            new_scores[i*block_size:(i+1)*block_size] = (1-beta)/node_num
            for from_node in stripes[i]:  # 遍历当前块下的所有源节点(stripe)
                for to_node in stripes[i][from_node]:  # 对应的目标节点
                    new_scores[to_node] += beta*scores[from_node]/length[from_node]
            e += sum(abs(new_scores[i*block_size:(i+1)*block_size]-scores[i*block_size:(i+1)*block_size]))
        
        # 处理剩余部分
        if remain != 0:
            new_scores[end_block_index*block_size:] = (1-beta)/node_num
            for from_node in stripes[end_block_index]:
                for to_node in stripes[end_block_index][from_node]:
                    new_scores[to_node] += beta*scores[from_node]/length[from_node]
            e+=sum(abs(new_scores[end_block_index*block_size:]-scores[end_block_index*block_size:]))
        scores=np.copy(new_scores)
        interation_num += 1
        print('interation_num:', interation_num, ' e:', e)


if __name__ == '__main__':
    
    stripes, length = deal_dead_end(stripes, length, node_num, block_size)
    print('finish deal dead end')
    start_time = time.time()
    power_interation_block_stripe(stripes, length, node_num, block_size, beta)
    end_time = time.time()
    print("时间：", end_time - start_time)
    print("scores_sum:",sum(scores))
    # 只取最大的前100个
    sorted_indices = np.argsort(scores)[::-1][:100]
    sorted_scores = scores[sorted_indices]
    print('Top 100:', sorted_scores)
    print('Top 100的点:', sorted_indices+1)  # 点的序号从1开始