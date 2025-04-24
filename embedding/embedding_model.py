import torch
from torch import nn

# Embedding 是一个继承自nn.Module的PyTorch模型类 
class Embedding(nn.Module):
    def __init__(self, batch_size, hidden_embedding_size, output_dim, dtype=torch.double):
        super(Embedding, self).__init__()
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.hidden_embedding_size = hidden_embedding_size
        self.params = nn.ParameterDict({
            'weights': nn.Parameter(torch.randn(1, hidden_embedding_size)),
            'biases': nn.Parameter(torch.rand(hidden_embedding_size)),
            'embedding_matrix': nn.Parameter(torch.rand(hidden_embedding_size, output_dim).double())
        })
        self.init_weights()
        self.pred = nn.Softmax(dim=2)
        
    def init_weights(self):
        weights = (param.data for name, param in self.named_parameters() if 'weights' in name)
        bias = (param.data for name, param in self.named_parameters() if 'bias' in name)
        embedding_matrix = (param.data for name, param in self.named_parameters() if 'embedding_matrix' in name)
        for k in weights:
            nn.init.xavier_uniform_(k)
        for k in bias:
            nn.init.zeros_(k)
        for k in embedding_matrix:
            nn.init.xavier_uniform_(k)
    
    # 将输入数据 t 通过模型进行处理并生成输出 embed
    def forward(self, t):
        output = []
        t = t.unsqueeze(2) 
        # tW+b：将 t 与随机初始化的权重向量 W 相乘，然后添加随机初始化的偏置向量 b
        projection = torch.mul(t, self.params['weights']) + self.params['biases']
        s = projection 
        # Etime = sEs，得到时间嵌入向量 Etime
        embed = torch.einsum('bsv,vi->bsi', s, self.params['embedding_matrix'])
        return embed


if __name__=='__main__':
    batch_size = 32
    hidden_dim = 128 
    embedding_dim = 768

    # 创建和初始化model
    time_model = Embedding(batch_size, hidden_dim, embedding_dim)
    cpu_usage_model = Embedding(batch_size, hidden_dim, embedding_dim)
    mem_usage_model = Embedding(batch_size, hidden_dim, embedding_dim)
    
    torch.save(time_model.state_dict(), 'xxx/time_model.pth')
    torch.save(cpu_usage_model.state_dict(), 'xxx/cpu_usage_model.pth')
    torch.save(mem_usage_model.state_dict(), 'xxx/mem_usage_model.pth')