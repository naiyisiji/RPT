import torch
from torch import nn
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_layer) -> None:
        super(MLP, self).__init__()
        self.hidden_layer = hidden_layer

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU(True)
        self.hidden_layer = nn.ModuleList([
            nn.Sequential(self.layer2,self.norm,self.relu) for _ in range(hidden_layer)])
    def forward(self, x):
        x = self.relu(self.norm(self.layer1(x)))
        for each_hidden_layer in self.hidden_layer:
            x = each_hidden_layer(x)
        x = self.layer3(x)
        return x
    
class TR_layer(nn.Module):
    def __init__(self, 
                 q_input_dim, 
                 k_input_dim, 
                 v_input_dim, 
                 hidden_dim,
                 num_heads,
                 dropout = 0.1) -> None:
        super(TR_layer, self).__init__()
        embed_dim = hidden_dim//16
        self.q_embed = nn.Linear(q_input_dim, embed_dim)
        self.k_embed = nn.Linear(k_input_dim, embed_dim)
        self.v_embed = nn.Linear(v_input_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Linear(embed_dim, hidden_dim)
    def forward(self, q, k, v):
        q = self.q_embed(q)
        k = self.k_embed(k)
        v = self.v_embed(v)
        attn_output,_ =self.attn(q,k,v)
        q = self.norm(q + attn_output)
        ffn_out = self.ffn(q)
        return ffn_out