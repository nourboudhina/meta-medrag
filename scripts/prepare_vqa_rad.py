import json
from pathlib import Path
from datasets import load_dataset

print("Chargement VQA-RAD depuis le cache...")
ds = load_dataset('flaviagiammarino/vqa-rad')

Path("data/raw/vqa_rad_images").mkdir(parents=True, exist_ok=True)
Path("data/raw/vqa_rad").mkdir(parents=True, exist_ok=True)

all_items = []

for split_name in ["train", "test"]:
    print(f"Conversion {split_name}...")
    for item in ds[split_name]:
        img = item.get('image')
        img_path = None
        if img is not None:
            qid = str(item.get('qid', len(all_items)))
            img_path = f"data/raw/vqa_rad_images/{qid}.png"
            img.save(img_path)
        all_items.append({
            "doc_id":   f"vqa_rad_{item.get('qid', len(all_items))}",
            "question": item.get('question', ''),
            "answer":   item.get('answer', ''),
            "q_type":   item.get('answer_type', ''),
            "domain":   "radiology",
            "image":    img_path,
            "text":     item.get('question', ''),
            "report":   item.get('answer', ''),
            "split":    split_name,
        })

with open("data/raw/vqa_rad/splits.json", "w", encoding="utf-8") as f:
    json.dump(all_items, f, indent=2)

train_n = sum(1 for x in all_items if x['split'] == 'train')
test_n  = sum(1 for x in all_items if x['split'] == 'test')
closed  = sum(1 for x in all_items if 'yes' in str(x.get('q_type','')).lower() or 'closed' in str(x.get('q_type','')).lower())

print(f"\nVQA-RAD pret:")
print(f"  Train: {train_n} | Test: {test_n} | Total: {len(all_items)}")
print(f"  Questions fermees (yes/no): {closed}")
print(f"  Questions ouvertes: {len(all_items) - closed}")
print(f"  Images: data/raw/vqa_rad_images/")
print(f"  Split:  data/raw/vqa_rad/splits.json")