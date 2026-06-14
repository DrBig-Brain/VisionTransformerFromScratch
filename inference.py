import torch
import torch.nn as nn
from dataset import train_loader
from models.vit import VisionTransformer
import matplotlib.pyplot as plt
import numpy as np
def inference():
    classes = ["glioma","meningioma","notumor","pituitary"]
    image_dim = 224
    patch_size = 16
    embedding_dim = 64
    num_heads = 4
    num_block=4
    device = "xpu" if torch.xpu.is_available() else "cpu"
    model = VisionTransformer(image_height=image_dim,image_widht=image_dim,image_channels=3, patch_size=patch_size, embedding_dim=embedding_dim, num_heads=num_heads, num_blocks=num_block, dropout = 0.1)
    model.to(device)
    checkpoint = torch.load("best_model.pth",map_location = device)
    model.load_state_dict(checkpoint["model_state_dict"])
    with torch.no_grad():
        for features, label in train_loader:
            features = features.to(device)
            label = label.to(device)
            logits = model(features)
            prediction = torch.argmax(logits,dim = 1)
            for i in range(64):
                image = features[i].permute(1,2,0).to("cpu").numpy()
                label = label.to("cpu")
                prediction = prediction.to("cpu")
                fig, (ax_img,ax_txt) = plt.subplots(1,2,figsize=(7,5))
                ax_img.imshow(image)
                ax_img.set_title("Brain Tumor Classification")
                ax_img.axis("off")

                ax_txt.axis("off")
                ax_txt.text(0.1, 0.8, f"Target: {classes[label[i].item()]}", fontsize=12)
                ax_txt.text(0.1, 0.6, f"Prediction: {classes[prediction[i].item()]}", fontsize=12)
                plt.tight_layout()
                plt.show()

            break
if __name__ == "__main__":
    inference()

