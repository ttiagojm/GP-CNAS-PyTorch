from torch import nn, optim
from torchvision import datasets, transforms
import torch
from tqdm import tqdm

# Select Accelerator to train and infer models
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32

t = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_ds = datasets.CIFAR10("../data", train=True, transform=t, download=True)
test_ds = datasets.CIFAR10("../data", train=False, transform=t, download=True)

IMG_SIZE = train_ds[0][0].shape
NUM_CLASSES = len(train_ds.classes)

# Calculate sizes -> 80% train (90% train and 10% validation) and 20% test
train_size = int(len(train_ds) * 0.8)
val_size = len(train_ds) - train_size

train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])

# Prepare dataloaders
train_data = torch.utils.data.DataLoader(train_ds, BATCH_SIZE, shuffle=True)
val_data = torch.utils.data.DataLoader(val_ds, BATCH_SIZE, shuffle=True)
test_data = torch.utils.data.DataLoader(test_ds, BATCH_SIZE, shuffle=True)


def train_step(model: nn.Module, optimizer: optim, loss_fn):
    total_loss = 0.
    total_acc = 0.
    last_loss = 0.
    last_acc = 0.
    num_batches = len(train_data)

    for i, data in tqdm(enumerate(train_data)):
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

        # Reset the derivatives (gradients)
        optimizer.zero_grad()

        logits = model(inputs)
        train_loss = loss_fn(logits, labels)

        # Derivatives
        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            # Calculate loss
            total_loss += train_loss.item()
            # Calculate accuracy
            pred_labels = torch.argmax(logits, dim=-1)
            total_acc += torch.sum(labels == pred_labels).item() / logits.shape[0]

            # Print something after some batches
            if i % num_batches == num_batches - 1:
                total_loss /= num_batches
                total_acc /= num_batches
                print("     Batch {} Loss: {:.4f} Acc: {:.4f}".format(i + 1, total_loss, total_acc))
                last_loss = total_loss
                last_acc = total_acc
                total_loss = 0.
                total_acc = 0.

    return last_loss, last_acc


def train_loop(model: nn.Module, epochs):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), 0.01)

    last_loss = 0.
    last_acc = 0.
    num_val_batches = len(val_data)

    for epoch in range(epochs):
        print("EPOCH %d" % (epoch + 1))

        model.train()
        train_loss, train_acc = train_step(model, optimizer, loss_fn)
        model.eval()

        # Validation losses
        with torch.no_grad():
            val_loss = 0.
            val_acc = 0.
            for data in val_data:
                inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                logits = model(inputs)

                # Calculate loss
                loss = loss_fn(logits, labels).item()
                val_loss += loss

                # Calculate accuracy
                pred_labels = torch.argmax(logits, dim=-1)
                val_acc += torch.sum(labels == pred_labels).item() / logits.shape[0]

            val_loss /= num_val_batches
            val_acc /= num_val_batches
            print(" Train Loss: {:.4f} Val Loss: {:.4f}".format(train_loss, val_loss))
            print(" Train Acc:  {:.4f} Val Acc:  {:.4f}".format(train_acc, val_acc))
            last_loss = val_loss
            last_acc = val_acc

    return last_loss, last_acc


class Tree2Model(nn.Module):
    def __init__(self, layers: list):
        super(Tree2Model, self).__init__()

        self.layers = nn.Sequential(*layers)

        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.LazyLinear(NUM_CLASSES)

    def forward(self, x):
        x = self.layers(x)

        x = self.global_avg(x)
        # Change channel number to the last position because of the next Linear
        x = torch.movedim(x, 1, -1)
        x = self.classifier(x)

        return torch.reshape(x, (x.shape[0], x.shape[-1]))


def is_valid_layers(layers: list):
    with torch.no_grad():
        # Check each resblock
        x = next(iter(train_data))[0].to(DEVICE)
        for i, resblock in enumerate(layers):
            if x.shape[-2] < 3 or x.shape[-1] < 3:
                return False
            x = resblock.layers(x)
        return True


def get_out_channels(layers: nn.Sequential):
    get_out = lambda obj, name: getattr(obj, name) if hasattr(obj, name) else False

    # Go bottom-top until find a layer with the output channels number
    for layer in reversed(layers):
        out = get_out(layer, "out_channels") or get_out(layer, "num_features")
        if out: break
    return out


class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.layers = None

    def forward(self, x):
        net = x.clone()

        net = self.layers(net)

        with torch.no_grad():
            # Calculating the kernel size needed to downsample
            out_channels = get_out_channels(self.layers)
            k_size = (x.shape[-2] - net.shape[-2] + 1, x.shape[-1] - net.shape[-1] + 1)

        # Apply an convolution using its identity function (delta function)
        conv = nn.Conv2d(x.shape[1], out_channels, k_size).to(DEVICE)
        nn.init.dirac_(conv.weight)

        return net + conv(x)


class ResBlock1(ResBlock):
    def __init__(self):
        super(ResBlock1, self).__init__()

        self.layers = nn.Sequential(
            nn.LazyBatchNorm2d(),
            StridedConv2D(16, (3, 3), stride=1),
            nn.ReLU(),
            nn.LazyBatchNorm2d(),
            nn.LazyConv2d(16, (3, 3), stride=1),
            nn.ReLU(),
        ).to(DEVICE)


class ResBlock2(ResBlock):
    def __init__(self):
        super(ResBlock2, self).__init__()

        self.layers = nn.Sequential(
            nn.LazyBatchNorm2d(),
            StridedConv2D(
                16,
                (3, 3),
                stride=1
            ),
            nn.ReLU(),
            nn.LazyBatchNorm2d(),
            nn.LazyConv2d(16, (3, 3), stride=1),
            nn.ReLU(),
            nn.LazyBatchNorm2d(),
            nn.LazyConv2d(16, (3, 3), stride=1),
            nn.ReLU()
        ).to(DEVICE)


class ResBlock3(ResBlock):
    def __init__(self):
        super(ResBlock3, self).__init__()

        self.layers = nn.Sequential(
            nn.LazyBatchNorm2d(),
            StridedConv2D(16, (3, 3), stride=1),
            nn.ReLU(),
            nn.LazyBatchNorm2d(),
            nn.LazyConv2d(16, (1, 1), stride=1),
            nn.ReLU()
        ).to(DEVICE)


class ResBlock4(ResBlock):
    def __init__(self):
        super(ResBlock4, self).__init__()

        self.layers = nn.Sequential(
            nn.LazyBatchNorm2d(),
            StridedConv2D(
                16,
                (3, 3),
                stride=1
            ),
            nn.ReLU(),
            nn.LazyBatchNorm2d(),
            nn.LazyConv2d(16, (1, 1), stride=1),
            nn.ReLU(),
            nn.LazyBatchNorm2d(),
            nn.LazyConv2d(16, (3, 3), stride=1),
            nn.ReLU()
        ).to(DEVICE)


class StridedConv2D(nn.LazyConv2d):
    def __init__(self, *args, **kwargs):
        """
        Class that is a custom type of 2D Convolutions, then could be distiguished to apply operations
        :param args: List of extra parameters
        :param kwargs: Dicitionary of extra parameters
        """
        super(StridedConv2D, self).__init__(*args, **kwargs)
