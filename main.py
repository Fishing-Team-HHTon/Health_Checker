# -*- coding: utf-8 -*-
import os
import ast
import random
import time
from collections import Counter

import numpy as np
import pandas as pd
import wfdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------- Settings / Repro ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = min(4, os.cpu_count() or 1)  # <= ставим безопасное значение для Windows
PIN_MEMORY = True if torch.cuda.is_available() else False


# --------------------------- Dataset ---------------------------
class PTBDataset(Dataset):
    def __init__(self, df, sampling_rate=500, records_base="ptb-xl-1.0.1/", augment=False):
        self.df = df.reset_index(drop=True)
        self.sampling_rate = sampling_rate
        self.records_base = records_base
        self.augment = augment
        self.classes = ["NORM", "MI", "STTC", "CD", "HYP"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def _read_raw(self, idx):
        row = self.df.iloc[idx]
        record_path = row.filename_hr if self.sampling_rate == 500 else row.filename_lr
        full_path = os.path.join(self.records_base, record_path)
        signal, meta = wfdb.rdsamp(full_path)
        return signal

    def augment_signal(self, signal):
        s = signal.copy().astype(np.float32)
        if random.random() < 0.5:
            s += np.random.normal(0, 0.01, size=s.shape).astype(np.float32)
        if random.random() < 0.3:
            s *= np.float32(np.random.uniform(0.9, 1.1))
        if random.random() < 0.3:
            shift = random.randint(-50, 50)
            s = np.roll(s, shift, axis=0)
        return s

    def _normalize_signal(self, signal):
        lead_q1 = np.percentile(signal, 25, axis=0)
        lead_q3 = np.percentile(signal, 75, axis=0)
        iqr = lead_q3 - lead_q1
        lower = (lead_q1 - 1.5 * iqr)[None, :]
        upper = (lead_q3 + 1.5 * iqr)[None, :]
        clipped = np.minimum(np.maximum(signal, lower), upper)
        mean = clipped.mean(axis=0)[None, :]
        std = clipped.std(axis=0)[None, :] + 1e-8
        normalized = (clipped - mean) / std
        return normalized.T.astype(np.float32)

    def __getitem__(self, idx):
        signal = self._read_raw(idx)
        if self.augment:
            signal = self.augment_signal(signal)
        signal = self._normalize_signal(signal)
        label = self.class_to_idx[self.df.iloc[idx].diagnostic_superclass]
        return torch.from_numpy(signal), torch.tensor(label, dtype=torch.long)


# --------------------------- Model ---------------------------
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


# --------------------------- Main ---------------------------
def main():
    # 1. Загружаем данные
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

    valid_classes = ["NORM", "MI", "STTC", "CD", "HYP"]
    df = df[df["diagnostic_superclass"].isin(valid_classes)].reset_index(drop=True)

    print("Распределение классов:")
    print(df["diagnostic_superclass"].value_counts())

    # 2. Сплит
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=SEED,
                                         stratify=df["diagnostic_superclass"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED,
                                       stratify=temp_df["diagnostic_superclass"])
    print(f"Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 3. Датасеты/даталоадеры
    train_dataset = PTBDataset(train_df, augment=True)
    val_dataset = PTBDataset(val_df, augment=False)
    test_dataset = PTBDataset(test_df, augment=False)

    class_counts = train_df["diagnostic_superclass"].value_counts().to_dict()
    class_weights = {k: 1.0 / v for k, v in class_counts.items()}
    sample_weights = [class_weights[row.diagnostic_superclass] for _, row in train_df.iterrows()]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # 4. Модель/оптимизатор
    model = ResNet1D(num_classes=len(valid_classes), in_channels=12).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    # 5. Тренировка
    best_f1 = 0.0
    patience, patience_counter = 10, 0
    start_time = time.time()

    for epoch in range(50):
        model.train()
        total_loss = 0.0
        it = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for x, y in pbar:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(x)
                loss = criterion(outputs, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            it += 1
            pbar.set_postfix(loss=total_loss / it)

        avg_loss = total_loss / max(1, it)

        # Валидация
        model.eval()
        y_true, y_pred = [], []
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    outputs = model(x)
                    loss = criterion(outputs, y)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1).cpu().numpy()
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds)

        val_loss = val_loss / max(1, len(val_loader))
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        scheduler.step(val_loss)

        elapsed = time.time() - start_time
        print(f"[{epoch+1:02d}] TrainLoss={avg_loss:.4f} | ValLoss={val_loss:.4f} | Acc={acc:.4f} | F1={f1:.4f} | Time={elapsed/60:.1f} мин")

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

    # 6. Тестирование
    model.load_state_dict(torch.load("best_ecg_model.pt"))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(x)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds)

    print("\n=== РЕЗУЛЬТАТЫ НА TEST SET ===")
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=valid_classes))

    # 7. Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(valid_classes))))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=valid_classes, yticklabels=valid_classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()  # важно для Windows
    main()
