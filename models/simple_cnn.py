import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleBelgiumCNN(nn.Module):
    """
    Lightweight CNN for BelgiumTS (32x32 RGB) mirroring the Keras example:
    Conv2D(64, 3x3) -> ReLU -> MaxPool(2) -> Flatten -> Dense(64) -> Dense(num_classes).
    """

    def __init__(self, num_classes: int = 62):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 15 * 15, 64)  # 32->30 after conv valid padding, then 15 after 2x2 pool
        self.fc2 = nn.Linear(64, num_classes)

    def get_feature(self, x):
        out = F.relu(self.conv(x))
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        out = self.get_feature(x)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out
