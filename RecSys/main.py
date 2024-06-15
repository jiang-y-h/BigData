from utils import *
from SVD import *
from UserCF import *
from ItemCF import *


def cal_util_data(data):
    '''
    计算全局平均分，用户平均分，用户偏差，物品平均分，物品偏差
    '''
    average_score = cal_global_avg(data)
    user_average_score, user_bias, max_user_bias, min_user_bias, average_user_bias = cal_user_bias(data, average_score)
    item_average_score, item_bias, max_item_bias, min_item_bias, average_item_bias = cal_item_bias(data, average_score)
    return average_score, user_average_score, user_bias, max_user_bias, min_user_bias, average_user_bias, item_average_score, item_bias, max_item_bias, min_item_bias, average_item_bias


if __name__ == '__main__':
    print("input the path of the train data:")
    path = input()
    train_data = read_data(path)

    print("input the model you want to use:")
    print("  1.baseline model")
    print("  2.UserCF model (may take a long time)")
    print("  3.ItemCF model (may take a long long time)")
    print("  4.SVD model")
    print("  5.SVD + bias model")
    print("  6.SVD + bias + attribute model")

    model = None
    model_name = input()

    global_avg, user_average_score, user_bias, max_user_bias, min_user_bias, average_user_bias, item_average_score, item_bias, max_item_bias, min_item_bias, average_item_bias = cal_util_data(train_data)
    baseline_data = {}
    baseline_data["global_avg"] = global_avg
    baseline_data["user_bias"] = user_bias
    baseline_data["item_bias"] = item_bias

    if model_name == '1':
        model = BaselineEstimator(global_avg, user_bias, item_bias)
    elif model_name == '2':
        estimator = BaselineEstimator(global_avg, user_bias, item_bias)
        model= UserCF(estimator,train_data)
    elif model_name == '3':
        with open("data/similar_nodes.pkl", "rb") as f:
            similar_nodes = pickle.load(f)
        estimator = BaselineEstimator(global_avg, user_bias, item_bias)
        model = ItemCF(estimator, path,similar_nodes)
    elif model_name == '4':
        model = SVD()
        model.train(5, 0.0005, train_data)
    elif model_name == '5':
        model = SVD_bias(baseline_data)
        model.train(5, 0.0005, train_data)
    elif model_name == '6':
        with open("data/similar_nodes.pkl", "rb") as f:
            similar_nodes = pickle.load(f)
        print("len(similar_nodes):", len(similar_nodes))
        model = SVD_attribute(baseline_data, similar_nodes)
        model.train(5, 0.0005, train_data)
    else:
        print("invalid input")
        exit()
    
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("model saved")

    print("input the path of the test data:")
    path = input()
    test_data = read_test_data(path)

    results = {}
    for user_id, item_list in test_data.items():
        for item_id in item_list:
            if results.get(user_id) is None:
                results[user_id] = {}
            results[user_id][item_id] = model.predict(user_id, item_id)
    print("predict done")

    with open("results.txt", "w") as f:
        for user_id, rate_data in results.items():
            f.write(str(user_id) + "|" + str(len(rate_data)) + "\n")
            for item_id, score in rate_data.items():
                f.write(str(item_id) + "\n")
    