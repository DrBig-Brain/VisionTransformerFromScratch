import torch
import torch.nn as nn
from tqdm import tqdm
from models.vit import VisionTransformer
from dataset import train_loader, test_loader
from test import evaluate
import datetime

device = "xpu" if torch.xpu.is_available() else "cpu"
print(f"training on: {device}")
num_epochs = 300
lr = 2e-3
image_dim = 224
patch_size = 16
embedding_dim = 64
num_heads = 4
num_block=4
dropout = 0.1

model = VisionTransformer(image_height=image_dim,image_widht=image_dim,image_channels=3, patch_size=patch_size, embedding_dim=embedding_dim, num_heads=num_heads, num_blocks=num_block, dropout = 0.1)
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay=0.01)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

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
    print(f"Training: {epoch+1}/{num_epochs}")
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
    #scheduler.step()
    print(f"Training Loss: {total_loss/len(train_loader)}")
    print(f"Evaluating: {epoch+1}/{num_epochs}")
    avg_loss, accuracy =  evaluate(model,test_loader,criterion,device)
    print(f"Validation Loss: {avg_loss}, Validation accuracy: {accuracy}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        print(f"best accuracy: {best_accuracy}")
        save_model(model)
    if best_accuracy > 95:
        print(f"TRAINING TERMINATED DESIRED ACCURACY ACHIEVED")
        with open(f"docs/training_log_{datetime.now()}.txt","w") as f:
            f.write(f"Time: {datetime.now}\n")
            f.write(f"Epoch: {epoch+1}\n")
            f.write(f"best_accuracy: {best_accuracy}\n")
            f.write(f"train_loss: {total_loss/len(train_loader)}\n")
            f.write(f"test_loss: {avg_loss}\n")
        break
