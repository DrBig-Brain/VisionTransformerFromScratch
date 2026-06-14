import torch
import torch.nn as nn
from models.mlp import MLP
from models.multi_head_attention import MultiHeadAttention

class Block(nn.Module):
    def __init__(self,embedding_dim:int = 64, num_heads:int = 8, dropout = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads,embedding_dim = embedding_dim, dropout = dropout)
        self.mlp = MLP(embedding_dim=embedding_dim, dropout = dropout)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self,x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
def test():
    image = torch.randn(1,64,64)
    model = Block(embedding_dim=64)
    res = model(image)
    print(res.shape)

if __name__ == "__main__":
    test()
