import numpy as np
from utils import *
import time
import multiprocessing

class PageRankBasic():
    '''
    内存充足, 不管是邻接矩阵还是稀疏矩阵都可以存下
    r和r_new都存储在内存中
    '''

    def __init__(self, G, node_num, log=False, beta=0.85, tol=1e-6):
        self.beta = beta
        self.tol = tol
        self.G = G
        self.node_num = node_num
        self.log = log

    def get_stochastic_matrix(self):
        # 初始化邻接矩阵全存进
        matrix = np.zeros((self.node_num,self.node_num))
        # 统计邻接矩阵
        for edge in self.G:
            matrix[edge[1]-1][edge[0]-1] = 1  # 入度0->1
        # 计算
        for j in range(self.node_num):
            sum_of_col = sum(matrix[:,j])  # 出度之和(d)
            # 如果发现dead-end，将其转为随机跳转
            if sum_of_col == 0:
                matrix[:,j] = 1/self.node_num
                continue
            for i in range(self.node_num):  # 1/d
                matrix[i,j] /= sum_of_col
        return matrix

    def power_interation(self):
        # 全存在ram
        matrix = self.get_stochastic_matrix()
        scores = np.ones((self.node_num))/self.node_num  # 用 1/node_num 初始化rank vector
        new_scores = np.zeros((self.node_num))

        interation_num = 0  # 迭代次数
        e = self.node_num # 两次迭代之间的误差

        while e > self.tol:
            new_scores = self.beta*np.dot(matrix,scores)+(1-self.beta)/self.node_num  # β随机游走
            e = sum(abs(new_scores-scores))
            scores = np.copy(new_scores)
            interation_num += 1

            if self.log == True:
                print(f"第{interation_num}次迭代, 误差为{e}")

        return scores, interation_num
    

class PageRankSparse():
    '''
    内存较充足
    可以完整存放r_new
    r和M存在disk, 为每个page读取disk
    '''

    def __init__(self, G, node_num, log=False, beta=0.85, tol=1e-6):
        self.beta = beta
        self.tol = tol
        self.G = G
        self.node_num = node_num
        self.log = log
        # on disk
        self.scores = np.ones((self.node_num))/self.node_num  # 1/N
        self.sparse_matrix = self.get_sparse_matrix()
    
    def get_sparse_matrix(self):
        # 初始化稀疏矩阵
        sparse_matrix = [[] for _ in range(self.node_num)]
        for edge in self.G:
            sparse_matrix[edge[0]-1].append(edge[1]-1)  # 出度0->1
        return sparse_matrix
    
    def read_disk(self, index):
        return self.sparse_matrix[index], self.scores[index]
    
    def write_disk(self, new_scores):
        self.scores = np.copy(new_scores)

    def power_interation(self):
        e = self.node_num  # 两次迭代之间的误差
        interation_num = 0  # 迭代次数

        while e > self.tol:
            new_scores = (1-self.beta)*np.ones((self.node_num))/self.node_num  # ram

            # 挨个读入稀疏矩阵
            for i in range(self.node_num):
                # 读取disk
                matrix_line, scores_i = self.read_disk(i)

                # 如果是dead-end
                if len(matrix_line) == 0:  # 没有出度，为所有节点分配
                    new_scores += self.beta*scores_i/self.node_num  # 因为稀疏矩阵，所以在这里处理
                    continue
                for j in matrix_line:  # i->j
                    new_scores[j] += self.beta*scores_i/len(matrix_line)

            e = sum(abs(new_scores-self.scores))
            self.write_disk(new_scores)
            interation_num += 1

            if self.log == True:
                print(f"第{interation_num}次迭代, 误差为{e}")

        return self.scores, interation_num

    def power_interation(self):
        e = self.node_num  # 两次迭代之间的误差
        interation_num = 0  # 迭代次数

        while e > self.tol:
            new_scores = np.zeros((self.node_num))

            for i in range(self.node_num):
                # 读取disk
                matrix_line, scores_i = self.read_disk(i)

                for j in matrix_line:  # i->j
                    new_scores[j] += self.beta*scores_i/len(matrix_line)

            # re-insert the leaked PageRank
            new_scores += (1-sum(new_scores))/self.node_num
            e = sum(abs(new_scores-self.scores))
            self.write_disk(new_scores)
            interation_num += 1

            if self.log == True:
                print(f"第{interation_num}次迭代, 误差为{e}")

        return self.scores, interation_num


class PageRankBlock():
    '''
    内存不足，需要分块计算
    r_new分块放入内存
    为每个块扫描磁盘M和r
    '''

    def __init__(self, G, node_num, block_size=2000, log=False, beta=0.85, tol=1e-6):
        self.beta = beta
        self.tol = tol
        self.G = G
        self.node_num = node_num
        self.log = log
        self.block_size = block_size
        # on disk
        self.scores = np.ones((self.node_num))/self.node_num  # 1/N
        self.sparse_matrix = self.get_sparse_matrix()
        self.new_scores = np.zeros((node_num))  # 分块存储在ram

    def get_sparse_matrix(self):
        # 初始化稀疏矩阵
        sparse_matrix = [[] for _ in range(self.node_num)]
        for edge in self.G:
            sparse_matrix[edge[0]-1].append(edge[1]-1)  # 出度0->1
        return sparse_matrix
    
    def read_disk(self, index):
        return self.sparse_matrix[index], self.scores[index]
    
    def write_disk(self, new_scores):
        self.scores = np.copy(new_scores)

    def read_block(self, begin, end):
        self.new_scores[begin:end] = (1-self.beta)/self.node_num

    def write_block(self, begin, end, value):
        self.new_scores[begin:end] += value

    def power_interation(self):
        # 分块
        block_num = self.node_num//self.block_size
        e = 1  # 两次迭代之间的误差
        interation_num = 0  # 迭代次数

        while e > self.tol:
            e = 0
            # 每次处理一块
            for i in range(block_num):
                # 读进一个块
                self.read_block(i*self.block_size, (i+1)*self.block_size)

                # scan M and r_old once for each block
                for j in range(self.node_num):
                    matrix_line, scores_j = self.read_disk(j)

                    # 遇到dead-end
                    if len(matrix_line) == 0:
                        self.write_block(i*self.block_size, (i+1)*self.block_size, self.beta*scores_j/self.node_num)
                        continue
                    for m in matrix_line:
                        if m>=i*self.block_size and m<(i+1)*self.block_size:
                            self.write_block(m, m+1, self.beta*scores_j/len(matrix_line))

                e += sum(abs(self.new_scores[i*self.block_size:(i+1)*self.block_size]-self.scores[i*self.block_size:(i+1)*self.block_size]))
            
            # 处理剩余部分
            self.read_block(block_num*self.block_size, self.node_num)

            for j in range(self.node_num):
                matrix_line, scores_j = self.read_disk(j)

                if len(matrix_line) == 0:
                    self.write_block(block_num*self.block_size, self.node_num, self.beta*scores_j/self.node_num)
                    continue
                for m in matrix_line:
                    if m>=block_num*self.block_size:
                        self.write_block(m, m+1, self.beta*scores_j/len(matrix_line))

            e += sum(abs(self.new_scores[block_num*self.block_size:]-self.scores[block_num*self.block_size:]))
            self.write_disk(self.new_scores)
            interation_num += 1

            if self.log == True:
                print(f"第{interation_num}次迭代, 误差为{e}")
            
        return self.scores, interation_num

class PRBlockStripe():
    '''
    内存不足, 优化读取磁盘的次数
    对状态转换矩阵进行分块处理
    每次读取r_new的一个块, 同时读取状态转换矩阵的一个块
    '''
    def __init__(self, G, node_num, block_size=2000, log=False, beta=0.85, tol=1e-6):
        self.beta = beta
        self.tol = tol
        self.G = G
        self.node_num = node_num
        self.log = log
        self.block_size = block_size
        # on disk
        self.scores = np.ones((self.node_num))/self.node_num  # 1/N
        self.stripes,self.length= self.get_stripes()
        self.new_scores = np.zeros((node_num))  # 分块存储在ram
    
    def deal_dead_end(self,stripes, length, node_num, block_size):
        block_num = self.node_num//block_size
        remain = self.node_num%block_size
        if remain != 0:
            block_num += 1
        for i in range(node_num):
            # 没有出度则为dead-end
            if length[i] == 0:
                length[i] = node_num
                for j in range(block_num):
                    stripes[j][i] = [k for k in range(j*block_size, min((j+1)*block_size, node_num))]
        return stripes, length

    def get_stripes(self):
        block_num = self.node_num//self.block_size
        remain = self.node_num%self.block_size
        if remain != 0:  # +1是因为最后一个stripe也得存
            block_num += 1

        stripes = [ {} for _ in range(block_num)]  # [0,1,2,...,block_num-1]
        length = [0 for _ in range(self.node_num)]

        # 初始化稀疏矩阵
        for edge in self.G:
            to_node = edge[1]-1
            from_node = edge[0]-1
            index = to_node//self.block_size  # dest所在块的编号
            if from_node not in stripes[index].keys():
                # 将from_node加入stripes
                stripes[index][from_node] = []
            stripes[index][from_node].append(to_node)  # 将to_node加入stripes
            length[from_node] += 1  # 记录每个节点的出度

        return self.deal_dead_end(stripes, length, self.node_num, self.block_size)
    
    def read_disk_stripe(self, index):
        return self.stripes[index]
    
    def read_disk_score(self, index):
        return self.scores[index]
    
    def write_disk(self, new_scores):
        self.scores = np.copy(new_scores)

    def read_block(self, begin, end):
        self.new_scores[begin:end] = (1-self.beta)/self.node_num

    def write_scores(self, index, value):
        self.new_scores[index] += value


    def power_interation(self):  
        block_num = self.node_num//self.block_size
        remain = self.node_num%self.block_size
        block_size = self.block_size
        if remain != 0:  # +1是因为最后一个stripe也得存
            block_num += 1
        end_block_index = block_num-1
            
        e = 1  # 两次迭代之间的误差
        interation_num = 0
        while e > 1e-3:
            e = 0
            # 每次处理一块
            for i in range(end_block_index):
                self.read_block(i*block_size, (i+1)*block_size)
                stripe=self.read_disk_stripe(i)
                for from_node in stripe:  # 遍历当前块下的所有源节点(stripe)
                    for to_node in stripe[from_node]:  # 对应的目标节点
                        self.write_scores(to_node,self.beta*self.read_disk_score(from_node)/self.length[from_node])
                e += sum(abs(self.new_scores[i*block_size:(i+1)*block_size]-self.scores[i*block_size:(i+1)*block_size]))
            
            # 处理剩余部分
            if remain != 0:
                self.read_block(end_block_index*block_size, node_num)
                stripe=self.read_disk_stripe(end_block_index)
                for from_node in stripe:
                    for to_node in stripe[from_node]:
                        self.write_scores(to_node,self.beta*self.read_disk_score(from_node)/self.length[from_node])
                e+=sum(abs(self.new_scores[end_block_index*block_size:]-self.scores[end_block_index*block_size:]))
            self.write_disk(self.new_scores)
            interation_num += 1
            
            if self.log == True:
                print(f"第{interation_num}次迭代, 误差为{e}")
        return self.scores, interation_num

class PRBlockStripeParallel():
    '''
    内存不足, 优化读取磁盘的次数
    对状态转换矩阵进行分块处理
    每次读取r_new的一个块, 同时读取状态转换矩阵的一个块
    对读取r_new进行计算时并行处理, 同时处理多个块
    '''
    def __init__(self, G, node_num, block_size=2000, log=False, beta=0.85, tol=1e-6):
        self.beta = beta
        self.tol = tol
        self.G = G
        self.node_num = node_num
        self.log = log
        self.block_size = block_size
        # on disk
        self.scores = np.ones((self.node_num))/self.node_num  # 1/N
        self.stripes,self.length= self.get_stripes()
        self.new_scores = np.zeros((node_num))  # 分块存储在ram
    
    def deal_dead_end(self,stripes, length, node_num, block_size):
        block_num = self.node_num//block_size
        remain = self.node_num%block_size
        if remain != 0:
            block_num += 1
        for i in range(node_num):
            # 没有出度则为dead-end
            if length[i] == 0:
                length[i] = node_num
                for j in range(block_num):
                    stripes[j][i] = [k for k in range(j*block_size, min((j+1)*block_size, node_num))]
        return stripes, length

    def get_stripes(self):
        block_num = self.node_num//self.block_size
        remain = self.node_num%self.block_size
        if remain != 0:  # +1是因为最后一个stripe也得存
            block_num += 1

        stripes = [ {} for _ in range(block_num)]  # [0,1,2,...,block_num-1]
        length = [0 for _ in range(self.node_num)]

        # 初始化稀疏矩阵
        for edge in self.G:
            to_node = edge[1]-1
            from_node = edge[0]-1
            index = to_node//self.block_size  # dest所在块的编号
            if from_node not in stripes[index].keys():
                # 将from_node加入stripes
                stripes[index][from_node] = []
            stripes[index][from_node].append(to_node)  # 将to_node加入stripes
            length[from_node] += 1  # 记录每个节点的出度

        return self.deal_dead_end(stripes, length, self.node_num, self.block_size)
    
    def read_disk_stripe(self, index):
        return self.stripes[index]
    def read_disk_scores(self, index):
        return self.scores[index]

    
    def write_disk(self, new_scores):
        self.scores = np.copy(new_scores)

    def read_block(self, begin, end):
        self.new_scores[begin:end] = (1-self.beta)/self.node_num

    def write_block(self, begin, end, value):
        self.new_scores[begin:end] = value

    def process_block(self,block_index,scores,stripe,length,is_last=False):

        start_index = block_index * self.block_size
        if is_last:
            remain = len(scores) - start_index
            new_scores_temp = (1 - self.beta) / self.node_num*np.ones((remain))
        else:
            new_scores_temp = (1 - self.beta) / self.node_num*np.ones((self.block_size))
        
        for from_node in stripe:
            for to_node in stripe[from_node]:
                new_scores_temp[to_node-start_index] += self.beta * self.read_disk_scores(from_node)/ length[from_node]
        
        if is_last:
            e=sum(abs(new_scores_temp-scores[start_index:]))
        else :
            e=sum(abs(new_scores_temp-scores[start_index:start_index+self.block_size]))
        return (new_scores_temp.tolist(),e)


    def power_interation(self): 
        
        block_num = self.node_num//self.block_size
        remain = self.node_num%self.block_size
        if remain != 0:  # +1是因为最后一个stripe也得存
            block_num += 1
        end_block_index = block_num-1
            
        e = 1  # 两次迭代之间的误差
        interation_num = 0
        
        while e > 1e-3:
            e=0

            # 进程函数的输入
            block_input = [(i,self.scores,self.read_disk_stripe(i),self.length) for i in range(end_block_index)]

            if remain != 0:
                block_input.append((end_block_index,self.scores,self.read_disk_stripe(end_block_index),self.length,True))
            
            temp_scores = []
            
            # 每个进程处理一个block
            pool = multiprocessing.Pool()
            temp_scores=pool.starmap(self.process_block, block_input)
            pool.close()
            pool.join()
            
            # 将进程函数处理的结果写回
            if remain != 0:
                for i, (slice_,slice_e) in enumerate(temp_scores[:-1]):
                    self.write_block(i * self.block_size, (i + 1) * self.block_size, slice_)
                    e+=slice_e
                self.new_scores[end_block_index*self.block_size:] = temp_scores[-1][0]
            else:
                for i, (slice_,slice_e) in enumerate(temp_scores):
                    self.write_block(i * self.block_size, (i + 1) * self.block_size, slice_)
                    e+=slice_e

            self.write_disk(self.new_scores)
            interation_num += 1
            if self.log == True:
                print(f"第{interation_num}次迭代, 误差为{e}")
        return self.scores, interation_num



if __name__ == "__main__":
    path = 'PageRank\Data.txt'
    graph, node_num, node_set = read_data(path)
    if check_continuous(node_set, node_num) == True:
        #prb = PageRankBasic(graph, node_num, log=True)
        #prs = PageRankSparse(graph, node_num, log=True)
        #prb = PageRankBlock(graph, node_num, block_size=2000, log=True)
        prbs=PRBlockStripeParallel(graph,node_num,block_size=2000,log=True)
        start_time = time.time()
        scores, interation_num = prbs.power_interation()
        end_time = time.time()
        print("时间：", end_time - start_time)
        sorted_indices, sorted_scores = sort_scores(scores)
        standard_result = standard_answer(graph)
        # print(sorted_indices[0], sorted_scores[0])
        # print(standard_result[sorted_indices[0]+1])
        check_result(sorted_indices, sorted_scores, standard_result)

