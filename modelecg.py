import torch
import torch.nn as nn

# --------------------------- Модель ---------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

class ResNet1D(nn.Module):
    def __init__(self, num_classes=5, in_channels=12):
        super().__init__()
        self.layer1 = ResidualBlock(in_channels, 32, stride=2)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

# --------------------------- Загружаем веса ---------------------------
DEVICE = 'cpu'  # или 'cuda'
state_dict_path = "best_ecg_model.pt"

# Создаем модель
model = ResNet1D(num_classes=5, in_channels=12)
model.load_state_dict(torch.load(state_dict_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --------------------------- Сохраняем рабочую модель ---------------------------
torch.save(model.state_dict(), "best_ecg_model_weights.pt")

print("Модель успешно сохранена как 'best_ecg_model_full.pt' и готова к использованию.")
