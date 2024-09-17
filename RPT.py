import torch
from torch import nn
from layers import MLP, TR_layer

class Adj_embedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layer) -> None:
        super(Adj_embedding, self).__init__()
        self.adj_embed = MLP(input_dim, hidden_dim, hidden_dim, hidden_layer)
        self.shortcut = nn.Linear(input_dim, hidden_dim)
    def forward(self, x):
        x = self.adj_embed(x) + self.shortcut(x)
        return x
    

class Ego_embed(nn.Module):
    def __init__(self,input_dim, hidden_dim, adj_veh_embed_dim, adj_ped_embed_dim, num_heads,dropout) -> None:
        super(Ego_embed, self).__init__()
        self.ego_embed = nn.Linear(input_dim, hidden_dim)
        self.ego2adj_veh = TR_layer(hidden_dim, adj_veh_embed_dim, adj_veh_embed_dim, hidden_dim, num_heads, dropout)
        self.ego2adj_ped = TR_layer(hidden_dim, adj_ped_embed_dim, adj_ped_embed_dim, hidden_dim, num_heads, dropout)
    def forward(self, ego_feat, adj_veh_embed, adj_ped_embed):
        ego_embed = self.ego_embed(ego_feat)
        ego2veh = self.ego2adj_veh(ego_embed, adj_veh_embed, adj_veh_embed)
        ego2ped = self.ego2adj_ped(ego_embed, adj_ped_embed, adj_ped_embed)
        return ego2veh, ego2ped, ego_embed
    

class Encoder(nn.Module):
    def __init__(self, 
                 ego_input_dim,
                 veh_input_dim, 
                 ped_input_dim, 
                 hidden_dim,
                 num_heads,
                 dropout,
                 adj_veh_hidden_layer = 1,
                 adj_ped_hidden_layer = 1):
        super(Encoder, self).__init__()
        self.adj_veh_embed = Adj_embedding(veh_input_dim, hidden_dim, adj_veh_hidden_layer)
        self.adj_ped_embed = Adj_embedding(ped_input_dim, hidden_dim, adj_ped_hidden_layer)
        self.ego_feat = Ego_embed(ego_input_dim, hidden_dim, adj_veh_embed_dim=hidden_dim, adj_ped_embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
    def forward(self, x, veh_feat, ped_feat):
        adj_veh_embed = self.adj_veh_embed(veh_feat)
        adj_ped_embed = self.adj_ped_embed(ped_feat)
        ego2veh, ego2ped, ego_embed = self.ego_feat(x, adj_veh_embed, adj_ped_embed)
        ego_feat = ego_embed + ego2ped + ego2veh
        return ego_feat, adj_veh_embed, adj_ped_embed
    
class Decoder(nn.Module):
    def __init__(self, hidden_dim, prob_class=6) -> None:
        super(Decoder, self).__init__()
        self.ego_feat = nn.Linear(hidden_dim,hidden_dim)
        self.danger_prob = MLP(hidden_dim, hidden_dim, prob_class, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, ego_feat, adj_veh_embed, adj_ped_embed):
        ego_feat = self.ego_feat(ego_feat)
        adj_veh_feat = torch.mean(adj_veh_embed,dim=1).unsqueeze(1)
        adj_ped_feat = torch.mean(adj_ped_embed,dim=1).unsqueeze(1)
        feat = ego_feat + adj_veh_feat + adj_ped_feat
        danger_prob = self.danger_prob(feat).squeeze(-2)
        danger_prob = self.softmax(danger_prob)
        return danger_prob


class RPT(nn.Module):
    def __init__(self, 
                 ego_input_dim,
                 veh_input_dim, 
                 ped_input_dim, 
                 hidden_dim,
                 num_heads,
                 dropout,
                 adj_veh_hidden_layer = 1,
                 adj_ped_hidden_layer = 1):
        super(RPT, self).__init__()
        self.encoder = Encoder(ego_input_dim,
                               veh_input_dim, 
                               ped_input_dim, 
                               hidden_dim,
                               num_heads,dropout,
                               adj_veh_hidden_layer = 1,
                               adj_ped_hidden_layer = 1)
        self.decoder = Decoder(hidden_dim)

    def forward(self, ego_feat, adj_veh_feat, adj_ped_feat):
        ego_feat, adj_veh_embed, adj_ped_embed = self.encoder(ego_feat, adj_veh_feat, adj_ped_feat)
        danger_prob = self.decoder(ego_feat, adj_veh_embed, adj_ped_embed)
        return danger_prob