import torch
import torch.nn as nn

class CreatePatches(nn.Module):
    def __init__(self,image_heigh:int, image_width:int, image_channels:int, patch_size:int):
        super().__init__()
        self.image_height = image_heigh
        self.image_width = image_width
        self.patch_size = patch_size
        self.image_channels = image_channels
    def forward(self,x):
        batch_size = x.shape[0]
        n_h = self.image_height//self.patch_size
        n_w = self.image_width//self.patch_size
        x = torch.reshape(x,(batch_size, n_h, self.patch_size, n_w, self.patch_size, self.image_channels))
        x = torch.permute(x,(0,1,3,2,4,5))
        return torch.reshape(x,(batch_size,n_h*n_w,(self.patch_size**2)*self.image_channels))
'''
def CreatePatches(image:torch.tensor):
    assert image.shape == (1,32,32,3)
    (_,H,W,C) = image.shape
    patch_size = 2
    n_h = H//patch_size
    n_w = W//patch_size
    patches = torch.reshape(image,
        (_,n_h,patch_size,n_w,patch_size,C)
    )
    patches = torch.permute(patches,(0,1,3,2,4,5)) #(B,n_H,P,n_W,P,C) -> (B,n_H,n_W,P,P,C)
    patches = torch.reshape(patches,(_,n_h*n_w,(patch_size**2)*C)) #(B,n_H,n_W,P,P,C) -> (B,n_H*n_W,p*p*C) -> (B,N,P^2*C)
    return patches
'''

    

if __name__ == "__main__":
    image = torch.randn(1,32,32,3)
    print(f"before = {image.shape}")
    model = CreatePatches(32,32,3,4)
    image_ = model(image)
    print(f"after = {image_.shape}") # should be Batch_size, 64, 48
    