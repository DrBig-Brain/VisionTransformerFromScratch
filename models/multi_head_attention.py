import torch
import torch.nn as nn
from models.self_attention import SelfAttention

class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads:int = 8, embedding_dim:int = 64, dropout = 0.1):
        super().__init__()
        self.multihead = nn.ModuleList([SelfAttention(embedding_dim = embedding_dim//num_heads, dropout = dropout) for _ in range(num_heads)]) # 8 * B, N, 64//8
        self.proj = nn.Linear(embedding_dim,embedding_dim) #64x64
        self.num_heads = num_heads

    def forward(self,x):
        chunks = torch.chunk(x, self.num_heads, dim=-1)
        out = []
        for head, chunk in zip(self.multihead,chunks):
            out.append(head(chunk))
        x = torch.cat(out, dim = -1)
        out = self.proj(x) #384x64 -> 64x64
        return out
    
def test():
    image = torch.randn(1,64,64)
    print(image.shape)
    model = MultiHeadAttention(num_heads=8, n_embeddings=64)
    image_ = model(image)
    print(image_.shape)

if __name__ == "__main__":
    test()
