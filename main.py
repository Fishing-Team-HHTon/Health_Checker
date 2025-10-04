# ---------- ИМПОРТ ----------
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import wfdb
import ast
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import random
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
import time
warnings.filterwarnings('ignore')

# ---------- 1. ПОДГОТОВКА ДАННЫХ ----------
df = pd.read_csv("ptb-xl-1.0.1/ptbxl_database.csv")
df["scp_codes"] = df["scp_codes"].apply(lambda x: ast.literal_eval(x))

scp_df = pd.read_csv("ptb-xl-1.0.1/scp_statements.csv", index_col=0)
scp_df = scp_df[scp_df["diagnostic"] == 1]

def aggregate_diagnostic(y_dict):
    tmp = []
    for key in y_dict.keys():
        if key in scp_df.index:
            tmp.append(scp_df.loc[key].diagnostic_class)
    return list(set(tmp))

df["diagnostic_superclass"] = df["scp_codes"].apply(aggregate_diagnostic)
df = df[df["diagnostic_superclass"].map(len) > 0]
df["diagnostic_superclass"] = df["diagnostic_superclass"].apply(lambda x: x[0])

# Фокус на 5 основных классов
valid_classes = ["NORM", "MI", "STTC", "CD", "HYP"]
df = df[df["diagnostic_superclass"].isin(valid_classes)]
print("Распределение классов:")
print(df["diagnostic_superclass"].value_counts())

# ---------- 2. DATASET С АУГМЕНТАЦИЯМИ ----------
class PTBDataset(Dataset):
    def __init__(self, df, sampling_rate=500, path="ptb-xl-1.0.1/records500/", augment=False):
        self.df = df.reset_index(drop=True)
        self.sampling_rate = sampling_rate
        self.path = path
        self.classes = valid_classes
        self.augment = augment
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def augment_signal(self, signal):
        s = signal.copy()
        # Гауссовский шум
        if random.random() < 0.5:
            s += np.random.normal(0, 0.01, s.shape)
        # Масштабирование
        if random.random() < 0.3:
            s *= np.random.uniform(0.9, 1.1)
        # Временной сдвиг
        if random.random() < 0.3:
            shift = random.randint(-50, 50)
            s = np.roll(s, shift, axis=0)
        return s

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_path = row.filename_hr if self.sampling_rate==500 else row.filename_lr
        signal, _ = wfdb.rdsamp(f"ptb-xl-1.0.1/{record_path}")

        if self.augment:
            signal = self.augment_signal(signal)

        # Нормализация
        signal_norm = np.zeros_like(signal)
        for i in range(signal.shape[1]):
            lead = signal[:, i]
            q1, q3 = np.percentile(lead, [25, 75])
            iqr = q3 - q1
            lead = np.clip(lead, q1 - 1.5*iqr, q3 + 1.5*iqr)
            signal_norm[:, i] = (lead - np.mean(lead)) / (np.std(lead)+1e-8)

        signal_norm = signal_norm.T.astype(np.float32)
        label = self.class_to_idx[row.diagnostic_superclass]
        return torch.tensor(signal_norm), torch.tensor(label)

# ---------- 3. МОДЕЛЬ RESNET1D ----------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 7, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.layer1 = ResidualBlock(12, 32, stride=2)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ---------- 4. РАЗДЕЛЕНИЕ ДАННЫХ ----------
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["diagnostic_superclass"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["diagnostic_superclass"])
print(f"Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

train_dataset = PTBDataset(train_df, augment=True)
val_dataset = PTBDataset(val_df, augment=False)
test_dataset = PTBDataset(test_df, augment=False)

# WeightedRandomSampler для дисбаланса
class_counts = train_df["diagnostic_superclass"].value_counts().sort_index()
class_weights = 1.0 / class_counts
sample_weights = [class_weights[row.diagnostic_superclass] for _, row in train_df.iterrows()]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ---------- 5. ОБУЧЕНИЕ ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используем устройство: {device}")

model = ResNet1D(num_classes=len(valid_classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

best_f1 = 0
patience, patience_counter = 10, 0

start_time = time.time()

for epoch in range(50):
    model.train()
    total_loss = 0
    # tqdm показывает прогресс батчей
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Валидация
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    scheduler.step(1-acc)

    elapsed = time.time() - start_time
    print(f"[{epoch+1:02d}] Loss={avg_loss:.4f} | Acc={acc:.4f} | F1={f1:.4f} | Time={elapsed/60:.1f} мин")

    if f1 > best_f1:
        best_f1 = f1
        patience_counter = 0
        torch.save(model.state_dict(), "best_ecg_model.pt")
        print(">>> Сохранена лучшая модель!")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(">>> Ранняя остановка!")
            break

# ---------- 6. ТЕСТИРОВАНИЕ ----------
model.load_state_dict(torch.load("best_ecg_model.pt"))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        preds = outputs.argmax(dim=1).cpu().numpy()
        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds)

print("\n=== РЕЗУЛЬТАТЫ НА TEST SET ===")
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')
print(f"Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
print(classification_report(y_true, y_pred, target_names=valid_classes))

# ---------- 7. CONFUSION MATRIX ----------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=valid_classes, yticklabels=valid_classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
