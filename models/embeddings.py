import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self,patch_dim=12,embedding_dim=64):
        super().__init__()

        self.projection = nn.Linear(patch_dim, embedding_dim)

    def forward(self, x):
        return self.projection(x)
    
def test():
    image = torch.randn(1,256,12)
    model = PatchEmbedding()
    result = model(image)
    print(result.shape)

if __name__ == "__main__":
    test()