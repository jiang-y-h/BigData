import numpy as np
from tqdm import tqdm
import pickle

class SVD_attribute:
    def __init__(self, baseline_data, similar_nodes, k = 3, portion = 0.8, factor = 50, lambda_p = 1e-2, lambda_q = 1e-2, 
                 lambda_bx = 1e-2, lambda_bi = 1e-2):
        """
        初始化SVD模型
        Args:
            baseline_data: dict, baseline数据
            similar_nodes: dict, 每个节点的相似节点
            k: int, 使用的相似节点个数
            factor: int, 隐向量的维度
            lambda_p: float, 正则化参数
            lambda_q: float, 正则化参数
            lambda_bx: float, 正则化参数
            lambda_bi: float, 正则化参数
        """
        self.factor = factor  # 隐向量的维度
        self.portion = portion
        # 正则化参数
        self.lambda_p = lambda_p
        self.lambda_q = lambda_q
        self.lambda_bx = lambda_bx
        self.lambda_bi = lambda_bi
        # 用户与物品偏置
        self.global_avg = baseline_data["global_avg"]
        self.bx = baseline_data["user_bias"]
        self.bi = baseline_data["item_bias"]
        # overall max_item_id: 624960 max_user_id: 19834
        max_item_id = 624960
        max_user_id = 19834
        # 随机初始化P(user) Q(item)矩阵
        self.P = np.random.normal(0, 0.1, size=(factor, max_user_id + 1))
        self.Q = np.random.normal(0, 0.1, size=(factor, max_item_id + 1))
        # 相似节点
        self.similar_nodes = similar_nodes
        self.k = k

    def predict(self, user_id, item_id):
        """
        预测用户user对物品item的评分
        Args:
            user_id: 用户id
            item_id: 物品id
        Returns:
            预测评分
        """
        if user_id in self.bx.keys():
            bx = self.bx[user_id]
        else:
            bx = 0
        if item_id in self.bi.keys():
            bi = self.bi[item_id]
        else:
            bi = 0
        p = self.P[:, user_id]
        q = self.Q[:, item_id]
        # 直接得分由SVD模型得到
        direct_score = self.global_avg + bx + bi + np.dot(p, q)
        # indrect_score由相似节点得分平均得到
        indirect_score, count = 0, 0
        if item_id in self.similar_nodes:
            for node_id in self.similar_nodes[item_id]:
                temp_q = self.Q[:, node_id]
                indirect_score += np.dot(p, temp_q)
                count += 1
                if count == self.k:
                    break
        if count == 0:
            score = direct_score
        else:
            score = direct_score * self.portion + (indirect_score / count) * (1 - self.portion)
        score = min(score, 100)
        score = max(score, 0)
        return score
    
    def loss(self, data):
        """
        计算loss
        Args:
            data: dict, 训练数据
        Returns:
            loss
        """
        loss, count = 0.0, 0
        for user_id, rate_data in data.items():
            for item_id, score in rate_data.items():
                predict = self.predict(user_id, item_id)
                loss += (predict - score) ** 2
                count += 1
        # 添加正则化项
        loss += self.lambda_p * np.linalg.norm(self.P) ** 2
        loss += self.lambda_q * np.linalg.norm(self.Q) ** 2
        loss += self.lambda_bx * np.linalg.norm(list(self.bx.values())) ** 2
        loss += self.lambda_bi * np.linalg.norm(list(self.bi.values())) ** 2
        return np.sqrt(loss / count)

    def train(self, epoches, lr, data, valid_data):
        """
        训练模型
        Args:
            epoches: int, 迭代次数
            lr: float, 学习率
            data: dict, 训练数据
        """
        for epoch in range(epoches):
            # 使用tqdm显示训练进度
            for user_id, rate_data in tqdm(data.items(), desc="Epoch {}".format(epoch)):
                for item_id, score in rate_data.items():
                    bx = self.bx[user_id]
                    bi = self.bi[item_id]
                    p = self.P[:, user_id]
                    q = self.Q[:, item_id]
                    # 计算梯度
                    error = score - self.predict(user_id, item_id)
                    self.bx[user_id] += lr * (error - self.lambda_bx * bx)
                    self.bi[item_id] += lr * (error - self.lambda_bi * bi)
                    self.P[:, user_id] += lr * (error * q - self.lambda_p * p)
                    self.Q[:, item_id] += lr * (error * p - self.lambda_q * q)
            # 计算loss
            epoch_loss = self.loss(valid_data)
            print("Epoch {} finished: validate loss={}".format(epoch, epoch_loss))
            # 学习率衰减
            lr *= 0.95
        

def RMSE(data, model):
    rmse, count = 0.0, 0
    for user_id, rate_data in data.items():
        for item_id, score in rate_data.items():
            predict = model.predict(user_id, item_id)
            rmse += (predict - score) ** 2
            count += 1
    rmse = np.sqrt((rmse / count))
    return rmse 


def read_data(path):
    # 读取train.txt格式的数据，返回字典
    data = {}
    with open(path, 'r') as f:
        while True:
            line = f.readline().strip()
            if not line:  # EOF
                break
            # 读取user_id和rate_num
            user_id, rate_num = line.split('|')
            rate_num = int(rate_num)
            user_id = int(user_id)
            # 读取用户的评分数据
            rate_data = {}
            for i in range(rate_num):
                item_id, score = f.readline().strip().split()
                item_id = int(item_id)
                score = int(score)
                rate_data[item_id] = score
            # 保存该用户的数据
            data[user_id] = rate_data
    return data


if __name__ == "__main__":
    with open("RecSys/data/similar_nodes.pkl", "rb") as f:
        similar_nodes = pickle.load(f)
    print("len(similar_nodes):", len(similar_nodes))
    train_path ="RecSys/data/train_data.txt"
    train_data = read_data(train_path)
    print("len(train_data):", len(train_data))
    valid_path ="RecSys/data/validate_data.txt"
    valid_data = read_data(valid_path)
    print("len(valid_data):", len(valid_data))
    # 网格搜索
    factors = [50, 100, 200]
    ks = [3, 5, 9]
    lambdas = [1e-1, 1e-2, 1e-3]
    portions = [0.8, 0.7, 0.6, 0.5]

    min_rmse = float('inf')
    best_params = (0, 0, 0, 0)
    for factor in factors:
        for k in ks:
            for lambda_ in lambdas:
                for portion in portions:
                    with open("RecSys/data/baseline_data.pkl", "rb") as f:
                        baseline_data = pickle.load(f)
                    model = SVD_attribute(baseline_data, similar_nodes, k, portion, factor, lambda_, lambda_, lambda_, lambda_)
                    model.train(10, 0.0005, train_data, valid_data)
                    rmse = RMSE(valid_data, model)
                    if rmse < min_rmse:
                        min_rmse = rmse
                        best_params = (factor, k, lambda_, portion)
    
    best_factor, best_k, best_lambda, best_portion = best_params
    print(f"Best RMSE={min_rmse} for factor={best_factor}, k={best_k}, lambda={best_lambda}, portion={best_portion}")