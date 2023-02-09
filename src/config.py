import torch
# Select Accelerator to train and infer models
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Activate CuDNN optimizations
torch.backends.cudnn.benchmark = True