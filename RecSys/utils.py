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

