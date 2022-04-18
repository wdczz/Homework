import torch
import  torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            # in_channels=1,
            # out_channels=16,
            # kernel_size=3,
            # stride=1,
            # padding=1,
            nn.ReLU(),
            nn.AvgPool2d(
                kernel_size=2,
                stride=2,
            ),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(
                in_features=32 * 7 * 7,
                out_features=128,
            ),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), - 1)
        x = self.fc(x)
        output = self.out(x)
        return output