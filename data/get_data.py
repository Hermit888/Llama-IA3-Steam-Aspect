import json
import os
from datasets import load_dataset

SAVE_DIR = os.path.join(os.path.dirname(__file__), "orig_data")
os.makedirs(SAVE_DIR, exist_ok=True)

LABEL_NAMES = ["recommended", "story", "gameplay", "visual", "audio", "technical", "price", "suggestion"]

dataset = load_dataset("ilos-vigil/steam-review-aspect-dataset")

for split in ("train", "test"):
    out_path = os.path.join(SAVE_DIR, f"{split}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in dataset[split]:
            record = {
                "appid": row["appid"],
                "review": row["review"],
                "cleaned_review": row["cleaned_review"],
                "labels": {name: int(val) for name, val in zip(LABEL_NAMES, row["labels"])},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {split} split ({len(dataset[split])} rows) -> {out_path}")
