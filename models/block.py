import torch
import torch.nn as nn
from mlp import MLP
from multi_head_attention import MultiHeadAttention

class Block(nn.Module):
    def __init__(self,embedding_dim:int = 64):
        super().__init__()
        self.mha = MultiHeadAttention(embedding_dim = embedding_dim)
        self.mlp = MLP(embedding_dim=64)
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