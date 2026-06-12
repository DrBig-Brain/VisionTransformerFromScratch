import torch
import torch.nn as nn
from tqdm import tqdm
from models.vit import VisionTransformer
from dataset import train_loader, test_loader
from test import evaluate

device = "xpu" if torch.xpu.is_available() else "cpu"
print(f"training on: {device}")
num_epochs = 100
lr = 1e-4
image_dim = 224
patch_size = 16
embedding_dim = 32
num_heads = 4
num_block=4

model = VisionTransformer(image_height=image_dim,image_widht=image_dim,image_channels=3, patch_size=patch_size, embedding_dim=embedding_dim, num_heads=num_heads, num_blocks=num_block)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay=0.03)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

def save_model(model):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "accuracy": accuracy,
        },
        "best_model.pth"
    )
    print(f"saved model to: best_model.pth")

best_accuracy = 0.0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    print(f"Training: {epoch+1}")
    loop = tqdm(train_loader,leave=True)

    for batch_idx, (features, labels) in enumerate(loop):
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        out = model(features)

        loss = criterion(out,labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_( # Gradient Clipping from original Paper
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss+=loss.item()
    scheduler.step()
    print(f"Training Loss: {total_loss/len(train_loader)}")
    print(f"Evaluating: {epoch+1}")
    avg_loss, accuracy =  evaluate(model,test_loader,criterion,device)
    print(f"Validation Loss: {avg_loss}, Validation accuracy: {accuracy}")
    print("\n")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        print(f"best accuracy: {best_accuracy}")
        save_model(model)

