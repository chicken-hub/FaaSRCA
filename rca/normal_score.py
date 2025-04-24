import pickle
import json
from create_dataset import MyOwnDataset

if __name__=='__main__':
    layers_num = 4
    hidden_num = 32
    epoch_num = 100
    backbone = "GAT"
    ablation = ""
    dataset = "train-ticket"
    time = "1"

# 加载模型
    with open("xxx/" + dataset + "-train" + backbone + "_model_" + str(layers_num) + "layers_" + str(hidden_num) + "dim_" + str(epoch_num) + "epoch-" + time + ablation + ".pkl", "rb") as f:
        loaded_model = pickle.load(f)
    
    with open('xxx/name2nodeNum_dict.json', 'r') as f:
        name2nodeNum_dict = json.load(f)

# 保存每种图的每个节点的分数
    graphType2nodeScore = {}
    for key in name2nodeNum_dict.keys():
        graphType2nodeScore[key] = {}

# 读取正常的数据集
    dataset_data_normal = []
    for i in range(0, 8):
        dataset_data_normal.append(MyOwnDataset(root="xxx/train-ticket-test-normal-" + ablation + str(i)))
# 读取正常文件列表
    with open('xxx/normal_file_list.pkl', 'rb') as file:
        normal_file_list = pickle.load(file)

# 分类后得到每个类的正常时的分数
    for index, dataset_data in enumerate(dataset_data_normal):
        data_size = len(dataset_data)
        for i in range(data_size):
            data = dataset_data[i] 
            fileName = normal_file_list[i + index*10000]
            trace_type = fileName.split("nodeAttr_dict_")[1].split("-000")[0]
            # 返回预测结果 pred，得分 score
            pred, score, emb = loaded_model.predict(data, return_pred=True, return_score=True, return_emb=True)
            
            for j in range(data.num_nodes):
                if j not in graphType2nodeScore[trace_type]:
                    graphType2nodeScore[trace_type][j] = []
                graphType2nodeScore[trace_type][j].append(score.tolist()[j])   

    
    with open("xxx/" + dataset + backbone + "_graphType2nodeScore_" + str(layers_num) + "layers_" + str(hidden_num) + "dim_" + str(epoch_num) + "epoch-" + time + ablation + ".json","w") as f:
        json.dump(graphType2nodeScore, f)