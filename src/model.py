from torch import nn, optim
import torch
from src.pipeline import Pipeline
from src.utils import DEVICE


def train_step(model: nn.Module, optimizer: optim, loss_fn):
    total_loss = 0.
    total_acc = 0.
    last_loss = 0.
    last_acc = 0.

    for i, data in enumerate(Pipeline.train_data):
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

        # Reset the derivatives (gradients)
        optimizer.zero_grad()

        logits = model(inputs)
        train_loss = loss_fn(logits, labels)

        # Derivatives
        train_loss.backward()
        optimizer.step()

        total_loss += train_loss.item()

        with torch.no_grad():
            # Calculate accuracy
            pred_labels = torch.argmax(logits, dim=-1)
            total_acc += torch.sum(labels == pred_labels).item() / logits.shape[0]

        # Print something after some batches
        if i % 250 == 249:
            total_loss /= 250
            total_acc /= 250
            print(" Batch {} Loss: {:.4f} Acc: {:.4f}".format(i + 1, total_loss, total_acc))
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

    for epoch in range(epochs):
        print("EPOCH %d" % (epoch + 1))

        model.train(True)
        train_loss, train_acc = train_step(model, optimizer, loss_fn)
        model.train(False)

        # Validation losses
        val_loss = 0.
        val_acc = 0.
        for data in Pipeline.val_data:
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            logits = model(inputs)

            # Calculate loss
            loss = loss_fn(logits, labels).item()
            val_loss += loss

            # Calculate accuracy
            pred_labels = torch.argmax(logits, dim=-1)
            val_acc += torch.sum(labels == pred_labels).item() / logits.shape[0]

        val_loss /= len(Pipeline.val_data)
        val_acc /= len(Pipeline.val_data)
        print("Train Loss: {:.4f} Val Loss: {:.4f}".format(train_loss, val_loss))
        print("Train Acc:  {:.4f} Val Acc:  {:.4f}".format(train_acc, val_acc))
        last_loss = val_loss
        last_acc = val_acc

    return last_loss, last_acc


class Tree2Model(nn.Module):
    def __init__(self, layers: list):
        super(Tree2Model, self).__init__()

        self.layers = self.__get_valid_layers(layers)

        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.LazyLinear(Pipeline.NUM_CLASSES)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.global_avg(x)
        # Change channel number to the last position because of the next Linear
        x = torch.movedim(x, 1, -1)
        x = self.classifier(x)

        return torch.reshape(x, (x.shape[0], x.shape[-1]))

    def __get_valid_layers(self, layers):
        # Check each resblock
        x = next(iter(Pipeline.train_data))[0]

        with torch.no_grad():
            for i, resblock in enumerate(layers):
                for layer in resblock.layers:
                    # Since all kernel will be <= 3x3 then inputs less than that
                    # could cause troubles
                    if x.shape[-2] < 3 or x.shape[-1] < 3:
                        return layers[:i]
                    x = layer(x)
        return layers
