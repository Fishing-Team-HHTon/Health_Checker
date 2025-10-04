from __future__ import annotations
import os, csv
from typing import Iterable
from ..models import Sample
from ..settings import settings

def _dir_for(rid: str) -> str:
    d = os.path.join(settings.data_dir, rid)
    os.makedirs(d, exist_ok=True)
    return d

def save_ndjson(rid: str, samples: Iterable[Sample]) -> str:
    path = os.path.join(_dir_for(rid), f"{rid}.ndjson")
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(s.model_dump_json())
            f.write("\n")
    return path

def save_csv(rid: str, samples: Iterable[Sample]) -> str:
    path = os.path.join(_dir_for(rid), f"{rid}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_ms","ts_unix_ms","adc","lead_off","hp","mv"])
        for s in samples:
            writer.writerow([s.t_ms, s.ts_unix_ms, s.adc, s.lead_off, s.hp, s.mv])
    return path

def save_txt(rid: str, samples: Iterable[Sample]) -> str:
    path = os.path.join(_dir_for(rid), f"{rid}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("t_ms\tts_unix_ms\tadc\tlead_off\thp\tmv\n")
        for s in samples:
            f.write(f"{s.t_ms}\t{s.ts_unix_ms}\t{s.adc}\t{int(bool(s.lead_off))}\t{'' if s.hp is None else s.hp}\t{'' if s.mv is None else s.mv}\n")
    return path
