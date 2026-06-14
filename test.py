import torch
import torch.nn as nn
from models.vit import VisionTransformer
from dataset import test_loader
from tqdm import tqdm

def evaluate(model, test_loader, criterion, device):
    model.eval()  # evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # disable gradients
        loop = tqdm(test_loader, leave=True)
        for batch_features, batch_labels in loop:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_features)          # forward pass
            loss = criterion(outputs, batch_labels)  # compute loss
            total_loss += loss.item()

            # Predictions
            _, predicted = torch.max(outputs, 1)     # get class with max score
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


if __name__ == "__main__":
    device = "xpu" if torch.xpu.is_available() else "cpu"
    image_dim = 224
    patch_size = 16
    embedding_dim = 64
    num_heads = 4
    num_block=4

    model = VisionTransformer(image_height=image_dim,image_widht=image_dim,image_channels=3, patch_size=patch_size, embedding_dim=embedding_dim, num_heads=num_heads, num_blocks=num_block, dropout = 0.1)
    model.to(device)

    checkpoint = torch.load(
        "best_model.pth",
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    val_loss, val_acc = evaluate(model, test_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

