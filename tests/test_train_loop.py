from src.model import train_loop, Tree2Model
from torch import nn

model = Tree2Model([
    nn.Conv2d(3, 64, (3,3)),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.Conv2d(64, 64, (3,3)),
    nn.ReLU(),
    nn.BatchNorm2d(64),
])

train_loop(model, 5)