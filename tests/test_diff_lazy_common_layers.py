# Verify the time that a model with the same layers (lazy and common) needs to train it
from time import perf_counter
from torch import nn
from src.model import train_loop

common_model = nn.Sequential(
    nn.Conv2d(3, 6, 5),
    nn.ReLU(),
    nn.MaxPool2d((2, 2)),
    nn.Conv2d(6, 16, 5),
    nn.ReLU(),
    nn.MaxPool2d((2, 2)),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120),
    nn.ReLU(),
    nn.Linear(120, 84),
    nn.ReLU(),
    nn.Linear(84, 10)
)

start = perf_counter()
train_loop(common_model, 2)
elapsed_common = perf_counter() - start
print("\n\n[+] Elapsed time for common: {}s\n\n".format(elapsed_common))

lazy_model = nn.Sequential(
    nn.LazyConv2d(6, 5),
    nn.ReLU(),
    nn.MaxPool2d((2, 2)),
    nn.LazyConv2d(16, 5),
    nn.ReLU(),
    nn.MaxPool2d((2, 2)),
    nn.Flatten(),
    nn.LazyLinear(120),
    nn.ReLU(),
    nn.LazyLinear(84),
    nn.ReLU(),
    nn.LazyLinear(10)
)

start = perf_counter()
train_loop(lazy_model, 2)
elapsed_lazy = perf_counter() - start
print("\n\n[+] Elapsed time for lazy: {}s\n\n".format(elapsed_lazy))

gain = elapsed_common / elapsed_lazy
print("[+] Gain Common over Lazy: {:.3f}x faster!".format(gain))
