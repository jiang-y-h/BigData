attribute_path = "data\itemAttribute.txt"

# 读取item属性
def read_attribute(attribute_path):
    item_attribute = {}
    with open(attribute_path, "r") as f:
        while True:
            line = f.readline().strip()
            if not line:
                break
            item_id, attribute1, attribute2= line.split('|')
            item_id = int(item_id)
            if attribute1 == "None":
                attribute1 = 0
            else:
                attribute1 = int(attribute1)
            if attribute2 == "None":
                attribute2 = 0
            else:
                attribute2 = int(attribute2)
            item_attribute[item_id] = (attribute1, attribute2)
    return item_attribute

def get_all_attribute(item_attribute):
    all_attribute = {}
    for item_id, attribute in item_attribute.items():
        if attribute[0]!=0:
            if attribute[0] not in all_attribute:
                all_attribute[attribute[0]] = set()
                all_attribute[attribute[0]].add(item_id)
                all_attribute[attribute[0]].add(attribute[0])
            else:
                all_attribute[attribute[0]].add(item_id)

        if attribute[1]!=0:
            if attribute[1] not in attribute:
                all_attribute[attribute[1]] = set()
                all_attribute[attribute[1]].add(item_id)
                all_attribute[attribute[1]].add(attribute[1])
            else:
                all_attribute[attribute[1]].add(item_id)


train_path ="data/train_data.txt"
valid_path ="data/validate_data.txt"

def read_data(path):
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

class itemCF():
    def __init__(self,user_average_score,item_attribute,all_attribute,item_average_score,user_bias,train_data):
        self.user_average_score=user_average_score
        self.item_attribute=item_attribute
        self.all_attribute=all_attribute
        self.item_average_score=item_average_score
        self.user_bias=user_bias
        self.train_data=train_data

    def pearson(item1,item2,average1,average2):
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
            temp1=item1[user]-average1
            temp2=item2[user]-average2
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
        atrribute=(0,0)
        if item_id in self.item_attribute:
            atrribute=self.item_attribute[item_id]
        neighbor=set()
        if atrribute[0]!=0:
            for item in self.all_attribute[atrribute[0]]:
                if item in self.train_data and user_id in self.train_data[item]:
                    neighbor.add(item)
        if atrribute[1]!=0:
            for item in self.all_attribute[atrribute[1]]:
                if item in self.train_data and user_id in self.train_data[item]:
                    neighbor.add(item)

        predict=self.global_avg+self.user_bias[user_id]
        if item_id in self.train_data:
            predict+=self.item_bias[item_id]
            sum=0
            temp_predict=0
            # 只计算拥有共同属性的item
            for item in neighbor:
                if item not in self.train_data:
                    continue
                pear=self.pearson(self.train_data[item_id],self.train_data[item],self.item_average_score[item_id],self.item_average_score[item])
                if pear<0:
                    continue
                if user_id not in self.train_data[item]:
                    continue
                temp_predict+=pear*(self.train_data[item][user_id]-self.item_average_score[item])
                sum+=pear
            # 属性相同的item没有被user评分，就计算与其他item的相似度
            if sum==0:
                # 计算与其他物品的相似度，并排序
                sim={}
                sim_num=0
                for item in self.train_data.keys():
                    if sim_num>=30:
                        break
                    if item==item_id:
                        continue
                    if user_id not in self.train_data[item]:
                        continue
                    pear=self.pearson(self.train_data[item_id],self.train_data[item],self.item_average_score[item_id],self.item_average_score[item])
                    sim_num+=1
                    sim[item]=pear
                if len(sim)!=0:
                    neighbor_k = sorted(sim, key=lambda x:sim[x], reverse=True)
                    index=0
                    sum_pearson=0
                    sim_predict=0
                    while index<len(neighbor_k):
                        if index>=10:
                            break
                        temp_item=neighbor_k[index]
                        sim_predict+=sim[temp_item]*(self.train_data[temp_item][user_id]-self.item_average_score[temp_item])
                        sum_pearson+=sim[temp_item]
                        index+=1
                    if sum_pearson!=0:
                        sim_predict=sim_predict/sum_pearson
                    predict+=sim_predict
            else:
                temp_predict=temp_predict/sum
                predict+=temp_predict
        return predict