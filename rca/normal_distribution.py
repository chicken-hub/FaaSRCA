import json
import numpy as np

if __name__=='__main__':
    layers_num = 4
    hidden_num = 32
    epoch_num = 100
    backbone = "GAT"
    ablation = ""
    dataset = "train-ticket"
    time = "1"

    with open("xxx/" + dataset + backbone + "_graphType2nodeScore_" + str(layers_num) + "layers_" + str(hidden_num) + "dim_" + str(epoch_num) + "epoch-" + time + ablation + ".json","r") as f:
        graphType2nodeScore = json.load(f)
        
    # 保存每种图的每个节点的分数分布
    graphType2nodeDistribution = {}
    for graphType, node in graphType2nodeScore.items():
        graphType2nodeDistribution[graphType] = {}
        for index, score in node.items():
            graphType2nodeDistribution[graphType][index] = {}
            mean = np.mean(score)
            std = np.std(score)
            graphType2nodeDistribution[graphType][index]["mean"] = mean
            graphType2nodeDistribution[graphType][index]["std"] = std

    with open("xxx/" + dataset + backbone + "_graphType2nodeDistribution_" + str(layers_num) + "layers_" + str(hidden_num) + "dim_" + str(epoch_num) + "epoch-" + time + ablation + ".json","w") as f:
        json.dump(graphType2nodeDistribution, f)