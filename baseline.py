import torch
from torch import nn 
from layers import MLP

class BaseLine(nn.Module):
    def __init__(self, 
                 ego_input_dim,
                 veh_input_dim, 
                 ped_input_dim, 
                 hidden_dim, 
                 layers=5,) -> None:
        super(BaseLine, self).__init__()
        self.encoder = MLP(ego_input_dim + veh_input_dim + ped_input_dim, hidden_dim, hidden_dim, layers)
        self.decoder = MLP(hidden_dim, hidden_dim, 6, layers)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, ego_feat, adj_veh_feat, adj_ped_feat):
        bs,_,_ = ego_feat.shape
        #ego_feat = ego_feat.repeat(1,3,1)
        adj_veh_feat = torch.mean(adj_veh_feat,dim=1).unsqueeze(1)
        adj_ped_feat = torch.mean(adj_ped_feat,dim=1).unsqueeze(1)
        feat = torch.cat((ego_feat, adj_veh_feat, adj_ped_feat), dim=-1)
        feat_ = self.encoder(feat)
        feat_ = feat_.view(bs, -1)
        prob = self.decoder(feat_)
        prob = self.softmax(prob)
        return prob