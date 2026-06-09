import torch
import torch.nn as nn

#x=(B,N,D)

class SelfAttention(nn.Module):
    def __init__(self,embedding_dim:int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.W_q = nn.Linear(self.embedding_dim,self.embedding_dim)
        self.W_k = nn.Linear(self.embedding_dim,self.embedding_dim)
        self.W_v = nn.Linear(self.embedding_dim,self.embedding_dim)

    def forward(self,x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        attn_score = Q@K.transpose(-2,-1)
        attn_score = attn_score/torch.sqrt(torch.tensor([self.embedding_dim]))

        weights = torch.softmax(attn_score, dim = -1)
        
        return weights @ V
    
def test():
    image = torch.rand(1,257,64)
    model = SelfAttention()
    result = model(image)
    print(result.shape)

if __name__ == "__main__":
    test()