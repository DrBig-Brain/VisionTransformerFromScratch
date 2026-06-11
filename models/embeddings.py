import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self,patch_dim=48,embedding_dim=64):
        super().__init__()

        self.projection = nn.Linear(patch_dim, embedding_dim)

    def forward(self, x):
        return self.projection(x)
    
def test():
    image = torch.randn(1,64,48)
    model = PatchEmbedding()
    result = model(image)
    print(result.shape) #output = (B, 64,64)

if __name__ == "__main__":
    test()