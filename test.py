import json
import numpy as np
import torch
import torch.nn.functional as F
from modelecg import ResNet1D  # Импортируем класс модели

# ---------------------- ПАРАМЕТРЫ ----------------------
WEIGHTS_PATH = "best_ecg_model.pt"  # файл с весами
DEVICE = 'cpu'
TARGET_LENGTH = 500
CHANNELS = ('mv',)
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# ---------------------- ФУНКЦИИ ----------------------
def ecg_json_to_tensor(json_lines, channels=CHANNELS, target_length=TARGET_LENGTH, device=DEVICE):
    """Преобразование JSON-строк с ЭКГ в тензор [1, channels, length]"""
    signal_list = []
    for ch in channels:
        vals = [json.loads(line)[ch] for line in json_lines]
        signal_list.append(vals)

    signal_np = np.array(signal_list, dtype=np.float32)

    if signal_np.shape[1] < target_length:
        pad_width = target_length - signal_np.shape[1]
        signal_np = np.pad(signal_np, ((0, 0), (0, pad_width)), mode='edge')
    else:
        signal_np = signal_np[:, :target_length]

    for i in range(signal_np.shape[0]):
        mean = np.mean(signal_np[i])
        std = np.std(signal_np[i])
        signal_np[i] = (signal_np[i] - mean) / (std + 1e-8)

    tensor = torch.tensor(signal_np, dtype=torch.float32).unsqueeze(0).to(device)
    return tensor


def predict_ecg_from_json(model, json_lines, channels=CHANNELS, target_length=TARGET_LENGTH, device=DEVICE):
    """Предсказание класса ЭКГ для JSON-данных"""
    model.eval()
    tensor = ecg_json_to_tensor(json_lines, channels=('mv',), target_length=500, device=DEVICE)
    tensor = tensor.repeat(1, 12, 1)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        predicted_class_idx = torch.argmax(probs, dim=1).item()
        confidence = probs.max().item()

    predicted_class = CLASSES[predicted_class_idx]
    return predicted_class, confidence, probs.cpu().numpy()[0]


# ---------------------- ЗАГРУЗКА МОДЕЛИ ----------------------
print("Создаём модель...")
model = ResNet1D(num_classes=len(CLASSES), in_channels=12)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("Модель загружена.")

# ---------------------- ПРИМЕР ПРЕДСКАЗАНИЯ ----------------------
#       |---------------------- Создаем тестовый сигнал ----------------------
#       | - Генерируем синусоидальный сигнал с шумом, длиной 500
t = np.linspace(0, 1, TARGET_LENGTH)
signal_mv = 0.5 * np.sin(2 * np.pi * 5 * t) + 0.05 * np.random.randn(TARGET_LENGTH)

# Переводим в JSON строки (имитация твоего формата)
example_json_lines = []
for i, val in enumerate(signal_mv):
    example_json_lines.append(json.dumps({
        "t_ms": i*5,  # пример времени
        "ts_unix_ms": 1759583546311 + i*5,
        "adc": int(val*1000),  # просто для примера
        "lead_off": False,
        "hp": val,
        "mv": val
    }))

predicted_class, confidence, probs = predict_ecg_from_json(model, example_json_lines)
print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.4f}")
print(f"Probabilities per class: {probs}")
