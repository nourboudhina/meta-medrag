import json
import random
from pathlib import Path

random.seed(42)

# ── Charge tous les datasets disponibles ──────────────────────
all_known   = []
all_unknown = []

# 1. IU-Xray — questions sur rapports radiology = KNOWN
print("Chargement IU-Xray...")
iu_data = json.load(open("data/raw/iu_xray/splits.json", encoding="utf-8"))
iu_train = [x for x in iu_data if x["split"] == "train" and x.get("image")]
for item in iu_train[:150]:
    all_known.append({
        "image":    item["image"],
        "question": item["question"],
        "label":    0,
        "domain":   "radiology",
        "source":   "iu_xray"
    })
print(f"  IU-Xray known: {len(all_known)}")

# 2. VQA-RAD — questions fermees yes/no = KNOWN (bien definies)
print("Chargement VQA-RAD...")
vqa_train = json.load(open("data/raw/vqa_rad/splits.json", encoding="utf-8"))
vqa_train = [x for x in vqa_train if x["split"] == "train" and x.get("image")]
for item in vqa_train[:100]:
    all_known.append({
        "image":    item["image"],
        "question": item["question"],
        "label":    0,
        "domain":   "radiology",
        "source":   "vqa_rad"
    })
print(f"  VQA-RAD known ajout: total known = {len(all_known)}")

# 3. SLAKE — questions ouvertes complexes = UNKNOWN
print("Chargement SLAKE...")
slake_train = json.load(open("data/raw/slake/splits.json", encoding="utf-8"))
slake_train = [x for x in slake_train if x["split"] == "train"]
# Questions ouvertes (pas yes/no) = plus difficiles = unknown
open_q = [x for x in slake_train if str(x.get("q_type","")).upper() != "CLOSED"]
for item in open_q[:150]:
    all_unknown.append({
        "image":    item.get("image"),
        "question": item["question"],
        "label":    1,
        "domain":   "radiology",
        "source":   "slake"
    })
print(f"  SLAKE unknown: {len(all_unknown)}")

# 4. Questions difficiles generees (fallback sans GPT-4o)
HARD_QUESTIONS = [
    "Does this image show features of hypersensitivity pneumonitis rather than usual interstitial pneumonia?",
    "Is the pattern consistent with respiratory bronchiolitis-ILD versus desquamative interstitial pneumonia?",
    "Does this radiograph show features of acute fibrinous and organising pneumonia (AFOP)?",
    "Are the findings indicative of pulmonary veno-occlusive disease rather than pulmonary arterial hypertension?",
    "Does this show evidence of combined pulmonary fibrosis and emphysema syndrome?",
    "Is this pattern consistent with non-specific interstitial pneumonia (NSIP)?",
    "Does this chest CT show tree-in-bud opacities suggesting endobronchial spread of tuberculosis?",
    "Are these findings consistent with lymphangioleiomyomatosis (LAM)?",
    "Does this image show signs of Bochdalek hernia in an adult?",
    "Is there evidence of Swyer-James-MacLeod syndrome on this chest X-ray?",
    "Does this show features of yellow nail syndrome with pleural effusion?",
    "Are the findings consistent with pulmonary alveolar proteinosis?",
    "Does this show evidence of POEMS syndrome pulmonary manifestations?",
    "Is there evidence of amyloidosis affecting the lungs in this image?",
    "Does this CT show features of pleuroparenchymal fibroelastosis (PPFE)?",
]

# Utilise les images IU-Xray pour ces questions difficiles
iu_images = [x["image"] for x in iu_train if x.get("image")]
for i, q in enumerate(HARD_QUESTIONS * 5):  # 75 hard questions
    img = iu_images[i % len(iu_images)]
    all_unknown.append({
        "image":    img,
        "question": q,
        "label":    1,
        "domain":   "radiology",
        "source":   "hard_template"
    })
print(f"  Hard questions unknown: total unknown = {len(all_unknown)}")

# 5. MIMIC-CXR si disponible
mimic_split = Path("data/raw/mimic_cxr/splits.json")
if mimic_split.exists():
    mimic_data = json.load(open(mimic_split, encoding="utf-8"))
    mimic_train = [x for x in mimic_data if x.get("split") == "train" and x.get("image")]
    for item in mimic_train[:50]:
        all_known.append({
            "image":    item["image"],
            "question": item.get("question", "Describe the findings in this chest X-ray."),
            "label":    0,
            "domain":   "radiology",
            "source":   "mimic_cxr"
        })
    print(f"  MIMIC-CXR known ajout: total known = {len(all_known)}")
else:
    print("  MIMIC-CXR: non disponible, ignore")

# ── Equilibre et shuffle ──────────────────────────────────────
n = min(len(all_known), len(all_unknown), 250)
known_sample   = random.sample(all_known,   n)
unknown_sample = random.sample(all_unknown, n)

dataset = known_sample + unknown_sample
random.shuffle(dataset)

# ── Sauvegarde ────────────────────────────────────────────────
Path("data/processed").mkdir(parents=True, exist_ok=True)
out_path = "data/processed/contrastive_pairs.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2)

print(f"\nDataset contrastif cree:")
print(f"  Total: {len(dataset)} paires")
print(f"  Known (label=0):   {sum(1 for x in dataset if x['label']==0)}")
print(f"  Unknown (label=1): {sum(1 for x in dataset if x['label']==1)}")
print(f"  Sources: {set(x['source'] for x in dataset)}")
print(f"  Sauvegarde: {out_path}")