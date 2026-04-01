import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import torch
import json
import sys
import numpy as np
import pickle
from pathlib import Path

sys.path.insert(0, r'C:\Users\Nour\Desktop\memoire\LLaVA-Med')
from transformers import AutoTokenizer
from llava.model import LlavaMistralForCausalLM

MODEL_PATH  = 'microsoft/llava-med-v1.5-mistral-7b'
OUTPUT_DIR  = Path('data/activations_real')
LAYERS      = [-2, -5, -8, -11, -15]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("Module 1 — Extraction activations réelles IU-Xray")
print("=" * 55)

# Charge les paires réelles
with open('data/processed/real_contrastive_pairs.json') as f:
    pairs = json.load(f)

known_qs   = pairs['known']
unknown_qs = pairs['unknown']
print(f"Known  : {len(known_qs)} questions")
print(f"Unknown: {len(unknown_qs)} questions")
print(f"Exemple known  : {known_qs[0]}")
print(f"Exemple unknown: {unknown_qs[0]}")

# Charge modèle
print("\n[1/3] Chargement LLaVA-Med CPU...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = LlavaMistralForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    device_map={'': 'cpu'},
    low_cpu_mem_usage=True
)
model.eval()
print(f"✅ Modèle chargé — {model.config.num_hidden_layers} couches")

# Extraction
print("\n[2/3] Extraction activations...")
all_acts   = {l: [] for l in LAYERS}
all_labels = []

for label_name, questions, label_int in [
    ('known',   known_qs,   0),
    ('unknown', unknown_qs, 1)
]:
    print(f"\n  '{label_name}' — {len(questions)} questions:")
    for i, q in enumerate(questions):
        if i % 20 == 0:
            print(f"    {i}/{len(questions)}...")
        inputs = tokenizer(q, return_tensors='pt', truncation=True, max_length=128)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        n = len(out.hidden_states) - 1
        for layer in LAYERS:
            idx = n + layer + 1
            act = out.hidden_states[idx][0].mean(dim=0).float()
            all_acts[layer].append(act)
        all_labels.append(label_int)

# Sauvegarde .pt
print("\n[3/3] Sauvegarde...")
for layer in LAYERS:
    t = torch.stack(all_acts[layer])
    torch.save(t, OUTPUT_DIR / f'layer_{layer}.pt')
    print(f"  layer {layer}: {t.shape}")

torch.save(torch.tensor(all_labels), OUTPUT_DIR / 'labels.pt')

# Sauvegarde pkl pour le vrai pipeline
X = np.concatenate([
    torch.load(OUTPUT_DIR / f'layer_{l}.pt').numpy()
    for l in sorted(LAYERS)
], axis=1)
y = np.array(all_labels)

Path('data/processed').mkdir(exist_ok=True)
with open('data/processed/activations_real.pkl', 'wb') as f:
    pickle.dump({'X': X, 'y': y}, f)

print(f"\n✅ {len(all_labels)} activations sauvegardées")
print(f"   X shape: {X.shape}")