import torch.utils.data
import torchvision
from torchvision import transforms
from src.utils import BATCH_SIZE
def download_and_prepare(batch_size):
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_ds = torchvision.datasets.CIFAR10("../data", train=True, transform=t, download=True)
    test_ds = torchvision.datasets.CIFAR10("../data", train=False, transform=t, download=True)

    img_size = train_ds[0][0].shape
    num_classes = len(train_ds.classes)

    # Calculate sizes -> 80% train (90% train and 10% validation) and 20% test
    train_size = int(len(train_ds) * 0.8)
    val_size = len(train_ds) - train_size

    train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])

    # Prepare dataloaders
    train_data = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True)
    val_data = torch.utils.data.DataLoader(val_ds, batch_size, shuffle=True)
    test_data = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=True)

    return img_size, num_classes, train_data, val_data, test_data

class Pipeline:
    """"
    Easy way to access the dataloaders
    """
    IMG_SIZE, NUM_CLASSES, train_data, val_data, test_data = download_and_prepare(BATCH_SIZE)