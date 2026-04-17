import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item):
        u = self.user_embedding(user)
        i = self.item_embedding(item)

        x = torch.cat([u, i], dim=1)
        x = self.mlp(x)

        return self.sigmoid(x).squeeze(-1)
