import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,embedding_dim:int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim,embedding_dim*4),
            nn.ReLU(),
            nn.Linear(embedding_dim*4,embedding_dim),
            nn.Dropout(0.2)
        )

    def forward(self,x):
        return self.mlp(x)
    

def test():
    image = torch.randn(1,64,64)
    model = MLP()
    resutl = model(image)
    print(resutl.shape)

if __name__ == "__main__":
    test()