import torch
import torch.nn as nn

class CNNBaseline(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),         # Added BatchNorm
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),         # Added BatchNorm
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()

            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):  # x: (B, 1, 300)
        out = self.conv(x)
        out = out.squeeze(-1)  # now shape: (B, 64)
        return self.classifier(out)
