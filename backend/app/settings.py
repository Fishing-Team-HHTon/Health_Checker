from __future__ import annotations
import os
from pydantic import BaseModel

class Settings(BaseModel):
    app_name: str = "biosense-backend"
    data_dir: str = os.getenv("DATA_DIR", "./data")
    default_duration_sec: int = int(os.getenv("DEFAULT_DURATION_SEC", "20"))
    nn_url: str | None = os.getenv("NN_URL")  # e.g., http://localhost:9000/infer
    max_batch_len: int = int(os.getenv("NN_BATCH_LEN", "64"))

settings = Settings()
os.makedirs(settings.data_dir, exist_ok=True)
