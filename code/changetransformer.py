import torch
import torch.nn as nn
import torch.nn.functional as F
from common.graph import MPNNModel

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


        self.num_drug_tokens = 4
        self.num_protein_tokens = 8
        self.drug_token_proj = nn.Linear(embed_dim, embed_dim * self.num_drug_tokens)
        self.protein_token_proj = nn.Linear(embed_dim, embed_dim * self.num_protein_tokens)
        self.drug_to_protein_attn = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
        self.protein_to_drug_attn = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
        self.fusion_fc = nn.Linear(embed_dim * 2, embed_dim)


        self.gateLinear = nn.Sequential(nn.Linear(512, 3), nn.Softmax(dim=1))


        self.fc1 = nn.Linear(embed_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.pairwise_tower = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.listwise_tower = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.pointwise_tower = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU()
        )

        self.pairwise_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.listwise_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        self.out = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_output)
        )
        self.calib_y1 = nn.Linear(1, 1)
        self.calib_y2 = nn.Linear(1, 1)
        self.calib_y3 = nn.Linear(1, 1)

        self.expert_scale = nn.Parameter(torch.ones(3))
        self.expert_bias = nn.Parameter(torch.zeros(3))
        self.register_buffer("ema_importance", torch.ones(3))
        self.ema_decay = 0.9
        self.importance_mix = 0.5

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, drug, protein):

        drug_embed = self.graph_encoder(drug)



        protein_embed = self.protein_fc(protein.float())

        batch_size = drug_embed.size(0)

        drug_tokens = self.drug_token_proj(drug_embed).view(batch_size, self.num_drug_tokens, -1)
        protein_tokens = self.protein_token_proj(protein_embed).view(batch_size, self.num_protein_tokens, -1)
        drug_ctx, _ = self.drug_to_protein_attn(drug_tokens, protein_tokens, protein_tokens)
        protein_ctx, _ = self.protein_to_drug_attn(protein_tokens, drug_tokens, drug_tokens)

        fused_features = torch.cat(
            [drug_ctx.mean(dim=1), protein_ctx.mean(dim=1)],
            dim=1,
        )
        combined_features = self.fusion_fc(fused_features)
        combined_features = combined_features + 0.5 * (drug_embed + protein_embed)


        shared_fc = self.relu(self.fc2(self.dropout(self.relu(self.fc1(combined_features)))))

        pair_feat = self.pairwise_tower(shared_fc)
        list_feat = self.listwise_tower(shared_fc)
        point_feat = self.pointwise_tower(shared_fc)
        y1_raw = self.pairwise_fc(pair_feat)
        y2_raw = self.listwise_fc(list_feat)
        y3_raw = self.out(point_feat)
        y1 = self.calib_y1(y1_raw)
        y2 = self.calib_y2(y2_raw)
        y3 = self.calib_y3(y3_raw)



        gate = self.gateLinear(shared_fc)

        y_all = torch.cat([y1, y2, y3], dim=1)
        y_all = y_all * self.expert_scale.unsqueeze(0) + self.expert_bias.unsqueeze(0)
        out = torch.sum(y_all * gate, dim=1, keepdim=True)



        batch_importance = gate.sum(dim=0).float()
        if self.training:
            detached_importance = batch_importance.detach().to(self.ema_importance.dtype)
            self.ema_importance.mul_(self.ema_decay).add_((1 - self.ema_decay) * detached_importance)
        importance = self.importance_mix * batch_importance + (1 - self.importance_mix) * self.ema_importance
        cv = importance.std(unbiased=False) / (importance.mean() + 1e-6)
        moe_loss = cv ** 2



        return out, moe_loss, y1, y2, y3, shared_fc
