from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor()])

train_dataset = ImageFolder(root="/home/abhinavmishra/Desktop/ViT/dataset/Training",transform=transform)
test_dataset = ImageFolder(root="/home/abhinavmishra/Desktop/ViT/dataset/Testing",transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size = 8, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size = 8)

def test():
    for features,label in train_loader:
        image = features[0].permute(1,2,0).numpy()
        plt.imshow(image)
        plt.show()
        print(label)
        break

if __name__ == "__main__":
    test()