import torch
import pickle
from create_dataset import MyOwnDataset
from torch_geometric.nn import GAT
from torch_geometric.loader import DataLoader
from pygod.detector.base import DeepDetector
from pygod.nn.decoder import DotProductDecoder
from pygod.nn.functional import double_recon_loss

class GAEBase(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim=32,
                 num_layers=4,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 sigmoid_s=False,
                 backbone=GAT,
                 **kwargs):
        super(GAEBase, self).__init__()

        # split the number of layers for the encoder and decoders
        assert num_layers >= 2, \
            "Number of layers must be greater than or equal to 2."
        self.encoder_layers = math.floor(num_layers / 2)
        self.decoder_layers = math.ceil(num_layers / 2)

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        
        self.shared_encoder = backbone(in_channels=self.in_dim,
                                       hidden_channels=self.hid_dim,
                                       num_layers=self.encoder_layers,
                                       out_channels=hid_dim,
                                       dropout=dropout,
                                       act=act,
                                       **kwargs)

        self.attr_decoder = backbone(in_channels=self.hid_dim,
                                     hidden_channels=self.hid_dim,
                                     num_layers=self.decoder_layers,
                                     out_channels=in_dim,
                                     dropout=dropout,
                                     act=act,
                                     **kwargs)
        
        self.struct_decoder = DotProductDecoder(in_dim=self.hid_dim,
                                                hid_dim=self.hid_dim,
                                                num_layers=self.decoder_layers - 1,
                                                dropout=dropout,
                                                act=act,
                                                sigmoid_s=sigmoid_s,
                                                backbone=backbone,
                                                **kwargs)
        
        self.loss_func = double_recon_loss
        self.emb = None

        
    def forward(self, x, edge_index):
        # # encode feature matrix
        self.emb = self.shared_encoder(x, edge_index)

        # # reconstruct feature matrix
        x_ = self.attr_decoder(self.emb, edge_index)

        # decode adjacency matrix
        s_ = self.struct_decoder(self.emb, edge_index)

        return x_, s_


    @staticmethod
    def process_graph(data):
        data.s = to_dense_adj(data.edge_index)[0]


class GAE(DeepDetector):
    def __init__(self,
                 hid_dim=32,
                 num_layers=4,
                 dropout=0.,
                 weight_decay=0.,
                 act=torch.nn.functional.relu,
                 sigmoid_s=False,
                 backbone=GAT,
                 contamination=0.1,
                 lr=4e-3,
                 epoch=100,
                 gpu=-1,
                 batch_size=0,
                 num_neigh=-1,
                 weight=1,
                 verbose=0,
                 save_emb=False,
                 compile_model=False,
                 **kwargs):

        super(GAE, self).__init__(hid_dim=hid_dim,
                                       num_layers=num_layers,
                                       dropout=dropout,
                                       weight_decay=weight_decay,
                                       act=act,
                                       backbone=backbone,
                                       contamination=contamination,
                                       lr=lr,
                                       epoch=epoch,
                                       gpu=gpu,
                                       batch_size=batch_size,
                                       num_neigh=num_neigh,
                                       verbose=verbose,
                                       save_emb=save_emb,
                                       compile_model=compile_model,
                                       **kwargs)

        self.weight = weight
        self.sigmoid_s = sigmoid_s

    def process_graph(self, data):
        GAEBase.process_graph(data)

    def init_model(self, **kwargs):
        if self.save_emb:
            self.emb = torch.zeros(self.num_nodes, self.hid_dim)
        return GAEBase(in_dim=self.in_dim,
                            hid_dim=self.hid_dim,
                            num_layers=self.num_layers,
                            dropout=self.dropout,
                            act=self.act,
                            sigmoid_s=self.sigmoid_s,
                            backbone=self.backbone,
                            **kwargs).to(self.device)
    
    
    def forward_model(self, data):
        batch_size = data.batch_size
        node_idx = data.n_id

        x = data.x.to(self.device)
        s = data.s.to(self.device)
        edge_index = data.edge_index.to(self.device)
        
        x_, s_ = self.model(x, edge_index)

        score = self.model.loss_func(x[:batch_size],
                                     x_[:batch_size],
                                     s[:batch_size, node_idx],
                                     s_[:batch_size],
                                     self.weight)

        loss = torch.mean(score)

        return loss, score.detach().cpu()



if __name__=='__main__':
    layers_num = 4
    hidden_num = 32
    epoch_num = 100
    backbone = "GAT"
    ablation = ""
    dataset = "train-ticket"
    time = "1"

    dataset_data = MyOwnDataset("xxx/" + dataset + "-train" + ablation)
    data = dataset_data._data
    contamination_ = sum(data.y).item() / len(data.y)

    # 创建 GAT auto-encoder，设置参数，例如隐藏层维度 hid_dim、层数 num_layers 和训练周期 epoch
    detector = GAE(hid_dim=hidden_num, num_layers=layers_num, lr=0.004, epoch=epoch_num, contamination=contamination_, backbone=GAT, gpu=0, verbose=3, weight=1.0)

    data_loader = DataLoader(dataset_data, batch_size=128, shuffle=True)
    i = 0
    for step, data in enumerate(data_loader):
        i += 1
        detector.fit(data)


    with open("xxx/" + dataset + "-train" + backbone + "_model_" + str(layers_num) + "layers_" + str(hidden_num) + "dim_" + str(epoch_num) + "epoch-" + time + ablation + ".pkl", "wb") as f:
        pickle.dump(detector, f)
