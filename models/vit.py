import torch
import torch.nn as nn
from create_patches import CreatePatches
from embeddings import PatchEmbedding
from block import Block

class VisionTransformer(nn.Module):
    def __init__(self, image_height, image_widht, image_channels, patch_size, embedding_dim,num_blocks):
        super().__init__()
        self.creaet_patches = CreatePatches(image_width=image_widht, image_heigh=image_height, image_channels=image_channels, patch_size=patch_size)
        self.embeddings = PatchEmbedding(patch_dim =(patch_size**2)*image_channels, embedding_dim=embedding_dim)
        self.positional_embedding = nn.Parameter(torch.randn(1,((image_height//patch_size)*(image_widht//patch_size)) + 1, embedding_dim))
        self.cls_token = nn.Parameter(torch.randn(1,1,embedding_dim))
        self.blocks = nn.Sequential(*[Block(embedding_dim=embedding_dim) for _ in range(num_blocks)])
        self.final_layernorm = nn.LayerNorm(embedding_dim)
        self.prediction_head = nn.Linear(embedding_dim, 4)

    def forward(self,x):
        patches = self.creaet_patches(x)
        patch_embeddings = self.embeddings(patches)
        cls_token = self.cls_token.expand(x.shape[0],-1,-1)
        patch_embeddings = torch.cat([cls_token, patch_embeddings],dim=1)
        patch_embeddings = patch_embeddings + self.positional_embedding
        out = self.blocks(patch_embeddings)
        out = self.final_layernorm(out)

        cls_out = out[:,0]

        logits = self.prediction_head(cls_out)
        return logits
    
def test():
    device = "xpu" if torch.xpu.is_available() else "cpu"
    print(f"Device: {device}")
    image = torch.randn(8,354,442,3).to(device)
    model = VisionTransformer(image_widht=354,image_height=442,image_channels=3,patch_size=8,embedding_dim=64,num_blocks=6)
    model.to(device)
    res = model(image)
    print(res.shape)

if __name__ == "__main__":
    test()