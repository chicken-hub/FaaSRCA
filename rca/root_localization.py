import pickle
import json
from create_dataset import MyOwnDataset
import numpy as np

def ndcg_at_k(scores, k):
    # 计算 DCG 值
    dcg = scores[0]
    for i in range(1, k):
        dcg += scores[i] / np.log2(i + 1)

    # 计算 IDCG 值
    ideal_scores = sorted(scores, reverse=True)
    idcg = ideal_scores[0]
    for i in range(1, min(k, len(ideal_scores))):
        idcg += ideal_scores[i] / np.log2(i + 1)

    # 得到 NDCG 值
    return dcg / idcg if idcg > 0 else 0.0


if __name__=='__main__':
# 加载模型
    layers_num = 4
    hidden_num = 32
    epoch_num = 100
    backbone = "GAT"
    ablation = ""
    dataset = "train-ticket"
    time = "1"

    with open("xxx/" + dataset + "-train" + backbone + "_model_" + str(layers_num) + "layers_" + str(hidden_num) + "dim_" + str(epoch_num) + "epoch-" + time + ablation + ".pkl", "rb") as f:
        loaded_model = pickle.load(f)

# 读取正常时的异常分数分布
    with open("xxx/" + dataset + backbone + "_graphType2nodeDistribution_" + str(layers_num) + "layers_" + str(hidden_num) + "dim_" + str(epoch_num) + "epoch-" + time + ablation + ".json","r") as f:
        graphType2nodeDistribution = json.load(f)
    
# 从文件中加载字典
    with open('xxx/name2nodeNum_dict.json', 'r') as f:
        name2nodeNum_dict = json.load(f)

# 读取待根因定位的异常数据
    dataset_data_abnormal = []
    for i in range(0, 8):
        dataset_data_abnormal.append(MyOwnDataset(root="xxx/train-ticket-test-abnormal-"+ablation+str(i)))
    with open('xxx/abnormal_file_list.pkl', 'rb') as file:
        abnormal_file_list = pickle.load(file)

# 第几张图，分别进行异常检测
    hit_ratio_dict = {}
    ndcg_dict = {}
    for i in range(6):
        hit_ratio_dict[i] = []
        ndcg_dict[i] = []
    topk_cnt = 0
    truth_label = []
    
    for test_index, test_data in enumerate(dataset_data_abnormal): 
        for i in range(len(test_data)):
            data = test_data[i] 
            abnormal_cnt += 1
            truth_label.append(1)

            fileName = abnormal_file_list[i + test_index*10000]
            trace_type = fileName.split("nodeAttr_dict_")[1].split("-000")[0]
            
            # 对测试集数据进行预测，返回预测结果 pred，得分 score
            pred, score, emb = loaded_model.predict(data, return_pred=True, return_score=True, return_emb=True)
            score_list = score.tolist()
            
            abnormal_score_deviation_dict = {}
            # 一个一个节点对齐分数分布
            for j in range(data.num_nodes):
                mean = graphType2nodeDistribution[trace_type][str(j)]["mean"]
                std = graphType2nodeDistribution[trace_type][str(j)]["std"]
                z_score = (score_list[j] - mean) / std
                # 计算 z_score 得到和正常时分数的偏差
                abnormal_score_deviation_dict[j] = z_score
                
            # 排序得到分数偏差最大的顺序以及对应的节点 
            sorted_dict = sorted(abnormal_score_deviation_dict.items(), key=lambda x: -x[1])
            sorted_list = [item[0] for item in sorted_dict]
            
            root_cnt = sum(data.y).item() 
            root_list = []
            for i in range(len(data.y)):
                if data.y[i] == 1:
                    root_list.append(i)
            
            for topk in range(root_cnt, root_cnt+6):
                topk_cnt = topk - root_cnt
                # topk list就是分数偏差列表的前topk项
                topk_list = sorted_list[:topk]
                topk_set = set(topk_list)
                root_set = set(root_list)
                # 计算交集的大小
                hit_count = len(topk_set.intersection(root_set))
                # 计算 hit ratio，命中率，预测的根因个数和实际的
                root_hit_ratio = hit_count / root_cnt
                hit_ratio_dict[topk_cnt].append(root_hit_ratio)
                
                # 计算ndcg，评估排序列表的质量
                scores = []
                for k in topk_list:
                    if k in root_list:
                        scores.append(1)
                    else:
                        scores.append(0)
                root_ndcg = ndcg_at_k(scores, k=len(scores))
                ndcg_dict[topk_cnt].append(root_ndcg)
            
    # 所有图中的平均结果
        for topk_cnt in hit_ratio_dict.keys():
            average_hit_ratio = sum(hit_ratio_dict[topk_cnt]) / len(hit_ratio_dict[topk_cnt])
            average_ndcg = sum(ndcg_dict[topk_cnt]) / len(ndcg_dict[topk_cnt])
                
                