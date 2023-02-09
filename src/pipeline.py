import torch
from torchvision import datasets, transforms

class Pipeline:
    BATCH_SIZE = 32
    def __init__(self):
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_ds = datasets.CIFAR10("../data", train=True, transform=t, download=True)
        test_ds = datasets.CIFAR10("../data", train=False, transform=t, download=True)

        self.IMG_SIZE = train_ds[0][0].shape
        self.NUM_CLASSES = len(train_ds.classes)

        # Calculate sizes -> 80% train (90% train and 10% validation) and 20% test
        train_size = int(len(train_ds) * 0.8)
        val_size = len(train_ds) - train_size

        train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])

        # Prepare dataloaders
        self.train_data = torch.utils.data.DataLoader(train_ds, self.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        self.val_data = torch.utils.data.DataLoader(val_ds, self.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        self.test_data = torch.utils.data.DataLoader(test_ds, self.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)