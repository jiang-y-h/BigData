import pickle
import numpy as np


class BaselineEstimator:
    '''
    baseline：使用全局平均分+用户偏差+物品偏差来估计评分
    '''
    def __init__(self, global_avg, user_bias, item_bias):
        self.global_avg = global_avg
        self.user_bias = user_bias
        self.item_bias = item_bias
        self.baseline_estimator = {}

    def save_model(self, path="models/baseline_estimator.pkl"):
        # 保存自身模型
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def predict(self, user_id, item_id):
        score = self.global_avg
        if user_id in self.user_bias:
            score += self.user_bias[user_id]
        if item_id in self.item_bias:
            score += self.item_bias[item_id]
        score = max(0, score)
        score = min(100, score)
        return score


def read_test_data(path):
    '''
    读取test.txt格式的数据，返回字典
    '''
    data = {}
    with open(path, 'r') as f:
        while True:
            line = f.readline().strip()
            if not line:
                break
            user_id, item_num = line.split('|')
            item_num = int(item_num)
            user_id = int(user_id)
            # 读取待预测的物品
            item_list = []
            for i in range(item_num):
                item_id = int(f.readline().strip())
                item_list.append(item_id)
            data[user_id] = item_list
    return data


def RMSE(data, model):
    '''
    计算RMSE
    '''
    rmse, count = 0.0, 0
    for user_id, rate_data in data.items():
        for item_id, score in rate_data.items():
            predict = model.predict(user_id, item_id)
            rmse += (predict - score) ** 2
            count += 1
    rmse = np.sqrt((rmse / count))
    return rmse 


def read_data(path):
    '''
    读取train.txt格式的数据，返回字典
    '''
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


def cal_global_avg(data):
    '''
    计算全局平均分
    '''
    sum_score = 0
    sum_num = 0
    for user_id, rate_data in data.items():
        sum_score += sum(rate_data.values())
        sum_num += len(rate_data)
    return sum_score / sum_num


def cal_user_bias(data, average_score):
    '''
    统计每个用户的平均评分，用户偏差
    '''
    # 每个用户的平均评分
    user_average_score = {}
    for user_id, rate_data in data.items():
        total_score = 0
        for score in rate_data.values():
            total_score += score
        user_average_score[user_id] = total_score / len(rate_data)
    # 每个用户与全局平均评分的偏差
    user_bias = {}
    for user_id, u_ave_score in user_average_score.items():
        user_bias[user_id] = u_ave_score - average_score
    # 最小偏差，最大偏差，平均偏差
    max_bias = max(user_bias.items(), key=lambda x: x[1])
    min_bias = min(user_bias.items(), key=lambda x: x[1])
    total_bias = 0
    for bias in user_bias.values():
        total_bias += bias
    average_bias = total_bias / len(user_bias)
    return user_average_score, user_bias, max_bias, min_bias, average_bias


def cal_item_bias(data, average_score):
    '''
    统计每个物品的平均评分，物品偏差
    '''
    # 统计物品得分
    item_scores = {}
    for user_id, rate_data in data.items():
        for item_id, score in rate_data.items():
            if item_id in item_scores:
                item_scores[item_id].append(score)
            else:
                item_scores[item_id] = [score]
    # 计算物品平均得分
    item_average_score = {}
    for item_id, scores in item_scores.items():
        item_average_score[item_id] = sum(scores) / len(scores)
    # 计算物品偏差
    item_bias = {}
    for item_id, i_ave_score in item_average_score.items():
        item_bias[item_id] = i_ave_score - average_score
    # 最大偏差，最小偏差，平均偏差
    max_bias = max(item_bias.items(), key=lambda x: x[1])
    min_bias = min(item_bias.items(), key=lambda x: x[1])
    total_bias = 0
    for bias in item_bias.values():
        total_bias += bias
    average_bias = total_bias / len(item_bias)
    return item_average_score, item_bias, max_bias, min_bias, average_bias