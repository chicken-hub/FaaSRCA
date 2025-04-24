import torch
from torch_geometric.data import Data, InMemoryDataset
import json
import os

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    # 生成数据集所用的方法
    def process(self):
        folder_path = ''
        # 获取文件夹中的所有文件和文件夹列表
        file_list = os.listdir(folder_path)
        sorted_file_list = file_list
        data_list = []
        cnt = 0
        # 遍历文件列表，每个文件做成一张图，然后放入图数据集
        for file_name in sorted_file_list:
            file_path = os.path.join(folder_path, file_name)
            # Read data into huge `Data` list.
            cnt += 1

            # 这里用于构建datas
            X, Y, Edge_index = graph_construction(file_path)
            # 放入datalist
            data = Data(x=X, y=Y, edge_index=Edge_index)
            # 创建包含多个图的图数据集
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# 接受一个文件路径作为参数，并返回读取到的邻接矩阵。
def read_adjacency_matrix(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        # 迭代读取每一行的方式，逐行处理邻接矩阵数据
        for line in file:
            row = list(map(int, line.strip().split()))
            matrix.append(row)
    return matrix 


def graph_construction(file_path):
    nodeName2nodeIndexDict = {}
    index = 0
    functionNameDict = {}
    feature_list = []
    label_list = []
    
    with open(file_path, 'r') as f:
        nodeAttributeDict = json.load(f)
    # 将每个节点的属性进行拼接，得到节点向量
    for nodeName in nodeAttributeDict.keys():
        node_embedding = []
        for key, value in nodeAttributeDict[nodeName].items():
            if "vector" in key:
                node_embedding += value
        feature_list.append(node_embedding)
        # 打标签
        if nodeAttributeDict[nodeName]["label"] == "normal":
            label_list.append(0)
        else:
            label_list.append(1)
        nodeName2nodeIndexDict[nodeName] = index
        index += 1
        functionNameDict[nodeName.split("_")[0]] = True
    
    # 每个节点的特征 
    X = torch.tensor(feature_list, dtype=torch.float)
    # 每个节点的标签 
    Y = torch.tensor(label_list, dtype=torch.long)    
    nodeIndex2nodeNameDict = {value:key for key, value in nodeName2nodeIndexDict.items()}
    
# 添加节点之间的边 
    forward_edge_list = []
    backward_edge_list = []
    # 在单个函数内部添加平台侧组件节点的边，同一函数前缀的不同组件节点之间有连边
    nodeNameEnd = ["_deployment", "_replicaset", "_pod", "_container"]
    for functionName in functionNameDict.keys():
        for i in range(len(nodeNameEnd) - 1):
            from_node = nodeName2nodeIndexDict[functionName + nodeNameEnd[i]]
            to_node = nodeName2nodeIndexDict[functionName + nodeNameEnd[i+1]]
            forward_edge_list.append(from_node)
            backward_edge_list.append(to_node)

# 从adjMatrix和serviceName2nodeIndex中提取边
    adjMatrix_path = "xxx/adjMatrix"
    nodeIndex_path = "xxx/serviceName2nodeIndex"
    processed_string = file_path.split("/vector/")[1]
    processed_string = processed_string.split("_vector")[0]
    adjMatrix_file = os.path.join(adjMatrix_path, "adjMatrix_"+processed_string+".txt")
    nodeIndex_file = os.path.join(nodeIndex_path, "serviceName2nodeIndex_"+processed_string+".json")
    # 读取邻接矩阵和函数节点对应表
    adjacency_matrix = read_adjacency_matrix(adjMatrix_file)
    with open(nodeIndex_file, 'r') as f2:
        nodeIndex_data = json.load(f2)
    indexNode_data = {value:key for key, value in nodeIndex_data.items()}
    # 根据 trace 逐个函数添加边
    for node, index in nodeIndex_data.items() :
        for col_index in range(len(adjacency_matrix[index])):
            element = adjacency_matrix[index][col_index]
            # 如果存在调用则会有有向边，从调用的 container 到被调用的 deployment
            if element > 0 :
                from_node = nodeName2nodeIndexDict[node + "_container"]
                to_node = nodeName2nodeIndexDict[indexNode_data[col_index] + "_deployment"]
                forward_edge_list.append(from_node)
                backward_edge_list.append(to_node)

    # 边的索引，[2,num_edges]从上指向下一一对应 
    Edge_index = torch.tensor([forward_edge_list,
                               backward_edge_list], dtype=torch.long)


    return X, Y, Edge_index


if __name__=='__main__':
    train_ticket_data = MyOwnDataset("xxx/train-ticket-train")
