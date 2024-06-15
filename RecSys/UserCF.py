from tqdm import tqdm
from utils import *
import pickle

class UserCF():
    def __init__(self, baseline_data,train_data,neighbor_k=20):
        self.baseline_data=baseline_data
        self.similarity=self.cal_similarity(train_data)
        self.neighbor_k=neighbor_k
        self.train_data=train_data

    def pearson(self,x, y,x_id,y_id):
        shared_items = set(x.keys()) & set(y.keys())
        # 如果没有共同元素，返回0
        if not shared_items:
            return 0
        # 计算pearson相关系数
        sim, sum1, sum2 = 0, 0, 0
        for item in shared_items:
            temp1 = x[item] - self.baseline_data.global_avg-self.baseline_data.user_bias[x_id]
            temp2 = y[item] - self.baseline_data.global_avg-self.baseline_data.user_bias[y_id]
            # 为了避免分母为0的情况，对将打分值做一个微调
            if temp1 == 0:
                temp1 = 0.1
            if temp2 == 0:
                temp2 = 0.1
            sim += temp1 * temp2  # 分子
            # 计算分母
            sum1 += temp1**2
            sum2 += temp2**2
        sim = sim / ((sum1**0.5) * (sum2**0.5))
        return sim
    
    def cal_similarity(self,train_set):
        """
        calculate the similarity between users
        Args:
            train_set: the train data
        Returns:
            similarity: the similarity matrix
        """
        similarity = {key:{} for key in train_set.keys()}
        for i, user1 in tqdm(enumerate(train_set.keys()), desc="similarity calculation"):
            for j, user2 in enumerate(list(train_set.keys())[i+1:], start=i+1):
                pearson_sim = self.pearson(train_set[user1], train_set[user2], user1, user2)
                similarity[user1][user2] = pearson_sim
                similarity[user2][user1] = pearson_sim
        return similarity
    
    
    def cf_user_predict(self,user_id, item_list, data,use_baseline=False):
        neighbor = sorted(self.similarity[user_id], key=lambda x: self.similarity[user_id][x], reverse=True)
        score={}
        for item_id in item_list:
            num=0
            index=0
            predict=0
            sum=0
            while index<len(neighbor):
                if num>=self.neighbor_k:
                    break
                if item_id in data[neighbor[index]]:
                    if self.similarity[user_id][neighbor[index]]<0:
                        break
                    num+=1
                    # temp = self.baseline_data.predict(neighbor[index],item_id) if use_baseline else 0
                    predict+=self.similarity[user_id][neighbor[index]]*(data[neighbor[index]][item_id])
                    sum+=self.similarity[user_id][neighbor[index]]
                index+=1
            if sum!=0:
                predict=predict/sum
            else:
                predict=self.baseline_data.global_avg+self.baseline_data.user_bias[user_id]
            
            if use_baseline:
                predict=0.5*predict+0.5*self.baseline_data.predict(user_id,item_id)
            score[item_id]=min(predict,100)
            score[item_id]=max(predict,0)
        return score
    
    def cal_rmse(self,train_data,valid_data,use_baseline=False):
        rmse, count = 0.0, 0
        for user_id, rate_data in tqdm(valid_data.items(), desc="Predicting users"):
            item_list = rate_data.keys()
            predict_score = self.cf_user_predict(user_id, item_list,train_data,use_baseline)
            rmse +=  sum((predict_score[key] - rate_data[key]) ** 2 for key in item_list)
            count += len(item_list)
        rmse = (rmse / count)**0.5
        return rmse
    
    def predict(self,user_id,item_id):
        neighbor = sorted(self.similarity[user_id], key=lambda x: self.similarity[user_id][x], reverse=True)
        num=0
        index=0
        predict_score=0
        sum=0
        while index<len(neighbor):
            if num>=self.neighbor_k:
                break
            if item_id in self.train_data[neighbor[index]]:
                if self.similarity[user_id][neighbor[index]]<0:
                    break
                num+=1
                predict_score+=self.similarity[user_id][neighbor[index]]*(self.train_data[neighbor[index]][item_id])
                sum+=self.similarity[user_id][neighbor[index]]
            index+=1
        if sum!=0:
            predict_score=predict_score/sum
        else:
            predict_score=self.baseline_data.global_avg+self.baseline_data.user_bias[user_id]
        
        predict_score=0.2*predict_score+0.8*self.baseline_data.predict(user_id,item_id)
        predict_score=min(predict_score,100)
        predict_score=max(predict_score,0)
        return predict_score





if __name__ == "__main__":
    train_path ="data/train_data.txt"
    train_data = read_data(train_path)
    print("len(train_data):", len(train_data))

    valid_path ="data/validate_data.txt"
    valid_data = read_data(valid_path)
    print("len(valid_data):", len(valid_data))

    baseline_path = 'models/baseline_estimator.pkl'
    with open(baseline_path, 'rb') as f:
        baseline_data = pickle.load(f)
    print("baseline_data:", baseline_data.global_avg)

    ucf=UserCF(baseline_data,train_data)
    rmse=ucf.cal_rmse(train_data,valid_data,use_baseline=True)
    print("rmse:",rmse)