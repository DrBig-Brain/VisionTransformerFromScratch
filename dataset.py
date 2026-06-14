from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor()])


train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05,0.05),
        scale=(0.95,1.05)
    ),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])


batch_size=64

train_dataset = ImageFolder(root="/home/abhinavmishra/Desktop/ViT/dataset/Training",transform=train_transform)
test_dataset = ImageFolder(root="/home/abhinavmishra/Desktop/ViT/dataset/Testing",transform=test_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size = batch_size)

def test():
    for features,label in train_loader:
        image = features[0].permute(1,2,0).numpy()
        plt.imshow(image)
        plt.show()
        print(label)
        break

if __name__ == "__main__":
    test()
