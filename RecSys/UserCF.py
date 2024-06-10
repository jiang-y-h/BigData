from tqdm import tqdm
import utils

class userCF():
    def __init__(self, data):
        self.data = data
        self.user_sim = self.user_similarity()
        self.user_sim_sorted = self.user_similarity_sorted()

    def pearson(x, y, x_ave, y_ave):
        shared_items = set(x.keys()) & set(y.keys())
        # 如果没有共同元素，返回0
        if not shared_items:
            return 0
        # 计算pearson相关系数
        sim, sum1, sum2 = 0, 0, 0
        for item in shared_items:
            temp1 = x[item] - x_ave
            temp2 = y[item] - y_ave
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
    
    def cal_similarity(self,train_set, user_average_score):
        """
        calculate the similarity between users
        Args:
            train_set: the train data
        Returns:
            similarity: the similarity matrix
        """
        similarity = {key:{} for key in train_set.keys()}
        for i, user1 in tqdm(enumerate(train_set.keys()), desc="Outer Loop"):
            for j, user2 in enumerate(list(train_set.keys())[i+1:], start=i+1):
                pearson_sim = self.pearson(train_set[user1], train_set[user2], user_average_score[user1], user_average_score[user2])
                similarity[user1][user2] = pearson_sim
                similarity[user2][user1] = pearson_sim
        return similarity
    
    def cf_user_predict(user_id, item_list, data, similarity, user_average_score, neighbor_k):
        neighbor = sorted(similarity[user_id], key=lambda x: similarity[user_id][x], reverse=True)
        score={}
        for item_id in item_list:
            num=0
            index=0
            predict=0
            sum=0
            while index<len(neighbor):
                if num>=neighbor_k:
                    break
                if item_id in data[neighbor[index]]:
                    if similarity[user_id][neighbor[index]]<0:
                        break
                    num+=1
                    predict+=similarity[user_id][neighbor[index]]*(data[neighbor[index]][item_id])
                    sum+=similarity[user_id][neighbor[index]]
                index+=1
            if sum!=0:
                predict=predict/sum
            else:
                predict=user_average_score[user_id]
            score[item_id]=predict
        return score


if __name__ == "__main__":
    train_path="data/train.txt"
    
    ucf = userCF()
    similarity = ucf.cal_similarity(train_data, user_average_score)
    test_data={}
    test_path="data/test.txt"

    with open(test_path,"r") as f:
        while True:
            data=f.readline()
            if not data:
                break
            data=data.split('|')
            user_id,rate_nums=int(data[0]),int(data[1])
            user_rate={}
            for i in range(int(rate_nums)):
                rate=int(f.readline())
                user_rate[rate]=0
            test_data[user_id]=user_rate

    for user in test_data.keys():
        item_list=list(test_data[user].keys())
        predict=ucf.cf_user_predict(user,item_list,train_data,similarity,user_average_score,10)
        for item in item_list:
            test_data[user][item]=predict[item]