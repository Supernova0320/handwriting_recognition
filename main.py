import torch
import torchvision
from torch.utils.data import DataLoader

# 准备参数
n_epochs = 3  # 循环训练集的次数
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)

