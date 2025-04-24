import torch
import json
from transformers import BertModel, BertTokenizer

# 得到所有事件模版的 bert 语义向量
if __name__ == '__main__':
    # 指定本地路径，base-768，large-1024
    local_path = "xxx/bert-base-uncased"
    # 加载预训练的BERT模型和分词器（model，tokenizer）
    tokenizer = BertTokenizer.from_pretrained(local_path, local_files_only=True)
    model = BertModel.from_pretrained(local_path, local_files_only=True)
    logTypeList = ["event", "auditLog"]
    
    for logType in logTypeList:
        with open("xxx/k8s_"+logType+".json", "r") as f:
            json_data = json.load(f)
        sentence_embedding = {}
        
        for key, value in json_data.items():
            # 提取每条日志的单词列表，将单词拼回成字符串
            sentence = ' '.join(value)
            # 使用加载的分词器对句子进行分词，并将分词后的结果转换为模型所需的输入张量。
            inputs = tokenizer(sentence, return_tensors='pt')
            # 使用加载的模型对输入张量进行前向传播，获取语义向量。
            with torch.no_grad():
                outputs = model(**inputs)
            # [CLS]经过全连接+tanh激活函数后的tensor张量，作为句子语义向量表示
            sentence_vector = torch.Tensor(outputs.pooler_output)
            sentence_vector_np = sentence_vector.numpy()
            sentence_embedding[sentence] = sentence_vector_np.tolist()
            
        with open("/xxx/k8s_"+logType+"_sentence_embedding.json", "w") as f:
            json.dump(sentence_embedding, f)


