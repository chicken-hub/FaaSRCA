import torch
import json
import numpy as np
import os
from embedding_model import Embedding
from torch import nn
import pickle
import random


# 嵌入 event 和 auditLog
def log_embedding(json_data, event_embedding_data, auditLog_embedding_data, k8s_event_template_miner, k8s_event, k8s_auditLog_template_miner, k8s_auditLog):
    event_vector = {}
    auditLog_vector = {}
    for functionName, attribute in json_data.items() :
        deployment_event = []
        replicaset_event = []
        pod_event = []
        # 对每条 event 原文找它们匹配的事件模版
        for log in attribute["Deployment_message"]:
            result = k8s_event_template_miner.match(log)
            preprocess = k8s_event[result.get_template()]
            deployment_event.append(preprocess)
        for log in attribute["Replicaset_message"]:
            result = k8s_event_template_miner.match(log)
            preprocess = k8s_event[result.get_template()]
            replicaset_event.append(preprocess)
        for log in attribute["Pod_message"]:
            result = k8s_event_template_miner.match(log)
            preprocess = k8s_event[result.get_template()]
            pod_event.append(preprocess)

        deployment_auditLog = []
        replicaset_auditLog = []
        pod_auditLog = []
        # 对每条 auditLog 原文找它们匹配的模版
        for log in attribute["Deployment_auditLog"]:
            result = k8s_auditLog_template_miner.match(log)
            preprocess = k8s_auditLog[result.get_template()]
            deployment_auditLog.append(preprocess)
        for log in attribute["Replicaset_auditLog"]:
            result = k8s_auditLog_template_miner.match(log)
            preprocess = k8s_auditLog[result.get_template()]
            replicaset_auditLog.append(preprocess)
        for log in attribute["Pod_auditLog"]:
            result = k8s_auditLog_template_miner.match(log)
            preprocess = k8s_auditLog[result.get_template()]
            pod_auditLog.append(preprocess)
        
        event = [deployment_event, replicaset_event, pod_event]
        auditLog = [deployment_auditLog, replicaset_auditLog, pod_auditLog]
        string = ["_deployment", "_replicaset", "_pod"]
        
# event 先查看该日志是否出现过，有则跳过，没有则将其语义向量添加到计算列表里，最后做池化得到文档向量  
        for index, attribute in enumerate(event):
            sentence_vectors = []
            sentence_dict = {}
            for event in attribute:
                # 提取每条日志的单词列表，将单词拼回成字符串
                sentence = ' '.join(event)
                if sentence not in sentence_dict.keys():
                    # 找到对应的句子语义向量，将NumPy数组转换回Tensor对象
                    sentence_vectors.append(torch.tensor(event_embedding_data[sentence]))
                    sentence_dict[sentence] = True
            # 将句子向量列表转换为Tensor类型
            tensor_vectors = torch.cat(sentence_vectors, dim=0)
            # 对句子向量进行sum，得到文档向量
            document_vector = torch.sum(tensor_vectors, dim=0)
            # 将向量转换为NumPy数组
            document_vector_np = document_vector.numpy()
            # 保存该函数下该文档的语义向量
            event_vector[functionName + string[index]] = document_vector_np.tolist()
            
# auditLog
        for index, attribute in enumerate(auditLog):
            sentence_vectors = []
            sentence_dict = {}
            for auditLog in attribute:
                # 提取每条日志的单词列表，将单词拼回成字符串
                sentence = ' '.join(auditLog)
                # 如果这条日志还没有出现过
                if sentence not in sentence_dict.keys():
                    sentence_vectors.append(torch.tensor(auditLog_embedding_data[sentence]))
                    sentence_dict[sentence] = True
            # 将句子向量列表转换为Tensor类型
            tensor_vectors = torch.cat(sentence_vectors, dim=0)
            # 对句子向量进行池化，得到文档向量
            document_vector = torch.sum(tensor_vectors, dim=0)
            # 将向量转换为NumPy数组
            document_vector_np = document_vector.numpy()
            # 保存该函数下该文档的语义向量
            auditLog_vector[functionName + string[index]] = document_vector_np.tolist()
            
    return event_vector, auditLog_vector


def time_embedding(json_data, time_model, nodeName2cleanTime):
    time_list = []
    time_list_cnt = 0
    node2TimeIndex = {}
    for functionName in json_data.keys() :
        one_function_list = [nodeName2cleanTime[functionName+"_deployment"], nodeName2cleanTime[functionName+"_replicaset"], nodeName2cleanTime[functionName+"_pod"], nodeName2cleanTime[functionName+"_container"]]
        time_list.append(one_function_list)
        node2TimeIndex[functionName] = time_list_cnt
        time_list_cnt += 1

    # 将列表转换为张量
    time_tensor = torch.tensor(time_list).to(torch.double)
    # 使用model的forward
    time_embedding = time_model(time_tensor)

    time_vector = {}
    for functionName in json_data.keys():
        time_vector[functionName + "_deployment"] = time_embedding[node2TimeIndex[functionName]][0].tolist()
        time_vector[functionName + "_replicaset"] = time_embedding[node2TimeIndex[functionName]][1].tolist()
        time_vector[functionName + "_pod"] = time_embedding[node2TimeIndex[functionName]][2].tolist()
        time_vector[functionName + "_container"] = time_embedding[node2TimeIndex[functionName]][3].tolist()

    return time_vector


def metric_embedding(json_data, cpu_usage_model, mem_usage_model, nodeName2standard_cpu_usage, nodeName2standard_mem_usage):
    cpu_usage_list = []
    cpu_usage_list_cnt = 0
    node2cpu_usageIndex = {}
    for functionName in json_data.keys():
        one_function_list = [nodeName2standard_cpu_usage[functionName+"_container"]]
        cpu_usage_list.append(one_function_list)
        node2cpu_usageIndex[functionName] = cpu_usage_list_cnt
        cpu_usage_list_cnt += 1
    # 将列表转换为张量
    cpu_usage_tensor = torch.tensor(cpu_usage_list).to(torch.double)
    # 使用model的forward
    cpu_usage_embedding = cpu_usage_model(cpu_usage_tensor)
    cpu_usage_vector = {}
    for functionName in json_data.keys():
        cpu_usage_vector[functionName + "_container"] = cpu_usage_embedding[node2cpu_usageIndex[functionName]][0].tolist()

    mem_usage_list = []
    mem_usage_list_cnt = 0
    node2mem_usageIndex = {}
    for functionName in json_data.keys():
        one_function_list = [nodeName2standard_mem_usage[functionName+"_container"]]
        mem_usage_list.append(one_function_list)
        node2mem_usageIndex[functionName] = mem_usage_list_cnt
        mem_usage_list_cnt += 1
    # 将列表转换为张量
    mem_usage_tensor = torch.tensor(mem_usage_list).to(torch.double)
    # 使用model的forward
    mem_usage_embedding = mem_usage_model(mem_usage_tensor)
    mem_usage_vector = {}
    for functionName in json_data.keys():
        mem_usage_vector[functionName + "_container"] = mem_usage_embedding[node2mem_usageIndex[functionName]][0].tolist()

    return cpu_usage_vector, mem_usage_vector


def findAllFile(path):
    for root, ds, fs in os.walk(path):
        for f in fs:
            if f.endswith('.json'):
                fullname = os.path.join(root, f)
                yield fullname
                
                
def all_embedding():
    # 读取drain模版文件
    with open("xxx/k8s_event_drain_template_miner.bin",'rb') as f:
        k8s_event_template_miner = pickle.load(f)
    # 读取预处理模版后的文件
    with open("xxx/k8s_event.json", "r") as file:
        k8s_event = json.load(file)
    # 读取预处理后模版对应向量的文件
    with open("xxx/k8s_event_sentence_embedding.json", "r") as f:
        event_embedding_data = json.load(f)
    with open("xxx/k8s_auditLog_drain_template_miner.bin",'rb') as f:
        k8s_auditLog_template_miner = pickle.load(f)
    with open("xxx/k8s_auditLog.json", "r") as file:
        k8s_auditLog = json.load(file)
    with open("xxx/k8s_auditLog_sentence_embedding.json", "r") as f:
        auditLog_embedding_data = json.load(f)

    batch_size = 32  
    hidden_dim = 128   
    output_dim = 768
    # 创建和初始化model
    time_model = Embedding(batch_size, hidden_dim, output_dim)
    # 读取模型
    time_model.load_state_dict(torch.load('xxx/time_model.pth'))
    with open("xxx/time.json", 'r') as f:
        nodeName2cleanTime = json.load(f)

    cpu_usage_model = Embedding(batch_size, hidden_dim, output_dim)
    cpu_usage_model.load_state_dict(torch.load('xxx/cpu_usage_model.pth'))
    with open("xxx/cpu_usage.json", 'r') as f:
        nodeName2standard_cpu_usage = json.load(f)

    mem_usage_model = Embedding(batch_size, hidden_dim, output_dim)
    mem_usage_model.load_state_dict(torch.load('xxx/mem_usage_model.pth'))
    with open("xxx/mem_usage.json", 'r') as f:
        nodeName2standard_mem_usage = json.load(f)

# 文件目录 trainticket
# filePath_list = ["xxx"]

# 提取文件
    file_list = []
    for filePath in filePath_list:
        # 获取文件夹中的所有文件和文件夹列表
        sorted_file_list = sorted(os.listdir(filePath))
        for file_path in sorted_file_list:
            fileName = filePath + file_path   
            file_list.append(fileName)

# 遍历文件建图 
    file_cnt = 0
    random.shuffle(file_list)
    total_cnt = 10000
    normal_file_list = []

    # 遍历文件列表
    for file_name in file_list:
        if file_cnt == total_cnt:
            break
        file_cnt += 1
        
        with open(file_name, 'r') as f:
            json_data = json.load(f)
        nodeNameDict = {}
        for functionName, attribute in json_data.items():
            # 一个函数对应四个节点
            nodeNameDict[functionName + "_deployment"] = "normal"
            nodeNameDict[functionName + "_replicaset"] = "normal"
            nodeNameDict[functionName + "_pod"] = "normal"
            nodeNameDict[functionName + "_container"] = "normal"
            if "abnormal" in attribute:
                # 添加异常标签
                if "deployment" in attribute["abnormal"]:
                    nodeNameDict[functionName + "_deployment"] = attribute["abnormal"]
                if "replicaset" in attribute["abnormal"]:
                    nodeNameDict[functionName + "_replicaset"] = attribute["abnormal"]
                if "pod" in attribute["abnormal"]:
                    nodeNameDict[functionName + "_pod"] = attribute["abnormal"]
                if "container" in attribute["abnormal"]:
                    nodeNameDict[functionName + "_container"] = attribute["abnormal"]

        processed_string = file_name.split("nodeAttr_dict_")[1]
        processed_string = processed_string.split(".json")[0]
        
        # 文本嵌入
        event_vector, auditLog_vector = log_embedding(json_data, event_embedding_data, auditLog_embedding_data, k8s_event_template_miner, k8s_event, k8s_auditLog_template_miner, k8s_auditLog)  
        # 时间嵌入
        time_vector = time_embedding(json_data, time_model, nodeName2cleanTime)
        # 指标嵌入
        cpu_usage_vector, mem_usage_vector = metric_embedding(json_data, cpu_usage_model, mem_usage_model, nodeName2standard_cpu_usage, nodeName2standard_mem_usage)

        # 给每个节点赋属性 
        nodeAttributeDict = {}
        for nodeName, nodeLabel in nodeNameDict.items():
            nodeAttributeDict[nodeName] = {}
            nodeAttributeDict[nodeName]["label"] = nodeLabel
            if nodeName in event_vector:
                nodeAttributeDict[nodeName]["event_vector"] = event_vector[nodeName]
            if nodeName in auditLog_vector:
                nodeAttributeDict[nodeName]["auditLog_vector"] = auditLog_vector[nodeName]
            if nodeName in cpu_usage_vector:
                nodeAttributeDict[nodeName]["cpu_usage_vector"] = cpu_usage_vector[nodeName]
            if nodeName in mem_usage_vector:
                nodeAttributeDict[nodeName]["mem_usage_vector"] = mem_usage_vector[nodeName]
            if nodeName in time_vector:
                nodeAttributeDict[nodeName]["time_vector"] = time_vector[nodeName]

        folder_path = "xxx/vector/"
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        with open(folder_path + processed_string + "_vector.json", "w") as f:
            json.dump(nodeAttributeDict, f)


if __name__=='__main__':
    all_embedding()

    