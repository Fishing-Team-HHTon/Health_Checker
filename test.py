import json
import numpy as np
import torch
import torch.nn.functional as F
# Не тестил так как модель не обучилась все 30 эпох
# ---------------------- ПАРАМЕТРЫ ----------------------
MODEL_PATH = "best_ecg_model.pt"  # твоя готовая модель
DEVICE = 'cpu'
TARGET_LENGTH = 500
CHANNELS = ('mv',)
# MI - миокард STTC - ишемия CD - нарушение проводимости сердца HYP - гипертрофия
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
    tensor = ecg_json_to_tensor(json_lines, channels, target_length, device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        predicted_class_idx = torch.argmax(probs, dim=1).item()
        confidence = probs.max().item()

    predicted_class = CLASSES[predicted_class_idx]
    return predicted_class, confidence, probs.cpu().numpy()[0]


# ---------------------- ЗАГРУЗКА МОДЕЛИ ----------------------
print("Загружаем готовую модель...")
model = torch.load(MODEL_PATH, map_location=DEVICE)
model.to(DEVICE)
model.eval()
print("Модель загружена.")

# ---------------------- ПРИМЕР ПРЕДСКАЗАНИЯ ----------------------
example_json_lines = [
    '{"t_ms":91779,"ts_unix_ms":1759583546311,"adc":796,"lead_off":false,"hp":-0.44364676,"mv":-2.1683614}',
    '{"t_ms":91794,"ts_unix_ms":1759583546326,"adc":794,"lead_off":false,"hp":0.74286014,"mv":3.6307926}',
    '{"t_ms":91810,"ts_unix_ms":1759583546342,"adc":795,"lead_off":false,"hp":1.574391,"mv":7.694971}',
    '{"t_ms":91825,"ts_unix_ms":1759583546357,"adc":719,"lead_off":false,"hp":-22.712088,"mv":-111.00727}',
    '{"t_ms":91826,"ts_unix_ms":1759583546358,"adc":796,"lead_off":false,"hp":6.742752,"mv":32.95578}'
]

predicted_class, confidence, probs = predict_ecg_from_json(model, example_json_lines)
print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.4f}")
print(f"Probabilities per class: {probs}")
