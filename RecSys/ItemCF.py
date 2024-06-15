from tqdm import tqdm
from utils import *
import pickle
import random

class itemCF():
    def __init__(self,baseline_data,train_data_path,similar_nodes):
        self.baseline=baseline_data
        self.train_data=self.read_data(train_data_path)
        self.similar_nodes=similar_nodes
    
    def read_data(self,path):
        train_data={}
        with open(path, "r") as f:
            while True:
                data=f.readline()
                if not data:
                    break
                data=data.split('|')
                user_id,rate_nums= int(data[0]),data[1]
                for i in range(int(rate_nums)):
                    rate=f.readline()
                    rate=rate.split()
                    item_id=int(rate[0])
                    if item_id not in train_data:
                        train_data[item_id]={}
                        train_data[item_id][user_id]=float(rate[1])
        return train_data

    def pearson(self,item1,item2,item1_id,item2_id):
        # 获得共有的item
        shared=set(item1.keys()) & set(item2.keys())
        # 如果没有共同元素，返回无穷
        if not shared:
            return 0
        # 计算pearson相关系数
        sim=0
        sum1=0
        sum2=0
        for user in shared:
            # print()
            # print(item1_id,item2_id)
            temp1=item1[user]-self.baseline.global_avg-self.baseline.item_bias[item1_id]
            temp2=item2[user]-self.baseline.global_avg-self.baseline.item_bias[item2_id]
            # 为了避免分母为0的情况，对将打分值做一个微调
            if temp1==0:
                temp1=0.1
            if temp2==0:
                temp2=0.1
            sim+=temp1*temp2
            sum1+=temp1**2
            sum2+=temp2**2
        sim=sim/((sum1**0.5)*(sum2**0.5))
        return sim
    
    def predict(self,item_id,user_id):
        neighbor=set()
        if item_id in self.similar_nodes:
            neighbor=self.similar_nodes[item_id]
        predict=self.baseline.predict(user_id,item_id)
        if item_id in self.train_data:
            sum=0
            temp_predict=0
            # 只计算拥有共同属性的item
            for item in neighbor:
                if item not in self.train_data:
                    continue
                pear=self.pearson(self.train_data[item_id],self.train_data[item],item_id,item)
                if pear<0.5:
                    continue
                if user_id not in self.train_data[item]:
                    continue
                temp_predict+=pear*(self.train_data[item][user_id]-self.baseline.predict(user_id,item))
                sum+=pear
            # 属性相同的item没有被user评分，就计算与其他item的相似度
            if sum==0:
                # 计算与其他物品的相似度，并排序
                sim={}
                sim_num=0
                random_list = list(self.train_data.keys())
                random.shuffle(random_list)
                for item in random_list:
                    if sim_num>=100:
                        break
                    if item==item_id:
                        continue
                    if user_id not in self.train_data[item]:
                        continue
                    pear=self.pearson(self.train_data[item_id],self.train_data[item],item_id,item)
                    sim_num+=1
                    sim[item]=pear
                if len(sim)!=0:
                    neighbor_k = sorted(sim, key=lambda x:sim[x], reverse=True)
                    index=0
                    sum_pearson=0
                    sim_predict=0
                    while index<len(neighbor_k):
                        if index>=20:
                            break
                        temp_item=neighbor_k[index]
                        sim_predict+=sim[temp_item]*(self.train_data[temp_item][user_id]-self.baseline.predict(user_id,temp_item))
                        sum_pearson+=sim[temp_item]
                        index+=1
                    if sum_pearson!=0:
                        sim_predict=sim_predict/sum_pearson
                    predict+=sim_predict
            else:
                temp_predict=temp_predict/sum
                predict+=temp_predict
        return predict
    
    def cal_rmse(self,valid_data_path):
        
        valid_data=self.read_data(valid_data_path)
        print(len(valid_data))
        print(len(self.train_data))
        print(len(valid_data)+len(self.train_data))
        sum=0
        count=0
        for item in tqdm(valid_data.keys(), desc="Processing items"):
            for user in valid_data[item].keys():
                predict=self.predict(item,user)
                sum+=(predict-valid_data[item][user])**2
            count+=len(valid_data[item])
            temp_rmse = (sum / count)**0.5
            print("curr_rmse:",temp_rmse)
        return (sum/count)**0.5

if __name__ == "__main__":
    train_path ="data/train_data.txt"
    valid_path ="data/validate_data.txt"

    baseline_path = 'models/baseline_estimator.pkl'
    with open(baseline_path, 'rb') as f:
        baseline_data = pickle.load(f)
    similar_nodes_path='data/similar_nodes.pkl'
    with open(similar_nodes_path, 'rb') as f:
        similar_nodes = pickle.load(f)
    icf=itemCF(baseline_data,train_path,similar_nodes)
    rmse=icf.cal_rmse(valid_path)
    print("rmse:",rmse)
    