import torch
import numpy as np

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

    

if __name__ == "__main__":
    image = torch.randn(1,32,32,3)
    print(f"before = {image.shape}")
    image_ = CreatePatches(image)
    print(f"after = {image_.shape}") # should be Batch_size, 256, 12
    