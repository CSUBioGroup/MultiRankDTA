import torch
import torch.nn as nn
import torch.nn.functional as F
from graph import MPNNModel

class TransformerModel(nn.Module):
    def __init__(self, n_output=1, embed_dim=128, graph_dim=70, esm_dim=1280, num_layers=2, dropout=0.2):
        super(TransformerModel, self).__init__()
        

        self.graph_encoder = MPNNModel(in_dim=graph_dim, edge_dim=6, emb_dim=embed_dim, num_layers=4)


        self.protein_fc = nn.Sequential(
            nn.Linear(esm_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim)
        )


        self.cross_attn = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
        self.reduce_dim = nn.Linear(embed_dim * 2, embed_dim)


        self.gateLinear = nn.Sequential(nn.Linear(512, 3), nn.Sigmoid())


        self.fc1 = nn.Linear(embed_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.pairwise_fc = nn.Linear(512, 1)
        self.listwise_fc = nn.Linear(512, 1)
        self.out = nn.Linear(512, n_output)  

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, drug, protein):

        drug_embed = self.graph_encoder(drug)
        protein_embed = self.protein_fc(protein.float()) 
        combined_features = torch.cat((drug_embed, protein_embed), dim=1)
        combined_features = self.reduce_dim(combined_features)
        combined_features = combined_features.unsqueeze(1)
        attn_output, _ = self.cross_attn(combined_features, combined_features, combined_features)
        combined_features = combined_features.squeeze(1) + attn_output.squeeze(1)

        # 共享 FC 层
        shared_fc = self.relu(self.fc2(self.dropout(self.relu(self.fc1(combined_features)))))

        y1 = self.pairwise_fc(shared_fc) 
        y2 = self.listwise_fc(shared_fc)  
        y3 = self.out(shared_fc)

        # MoE 门控融合
        gate = self.gateLinear(shared_fc)  
        y_all = torch.cat([y1, y2, y3], dim=1).unsqueeze(-1)  
        out = torch.sum(y_all * gate.unsqueeze(-1), dim=1)  
        moe_loss = 0



        return out, moe_loss, y1, y2, y3, shared_fc
