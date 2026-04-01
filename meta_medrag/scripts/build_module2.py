import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import torch
import numpy as np
import faiss
import json
from pathlib import Path
from PIL import Image
from open_clip import create_model_and_transforms, get_tokenizer

MODEL_ID   = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
VECTOR_DIR = Path('data/vector_stores')
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

IU_REPORTS_DIR = Path('data/raw/iu_xray/reports')
IU_IMAGES_DIR  = Path('data/raw/iu_xray/images')

print("=" * 55)
print("Meta-MedRAG — Module 2 : BiomedCLIP + FAISS")
print("=" * 55)

print("\n[1/4] Chargement BiomedCLIP...")
model, _, preprocess = create_model_and_transforms(MODEL_ID)
tokenizer = get_tokenizer(MODEL_ID)
model.eval()
print("  ✅ BiomedCLIP OK")

# ── Étape 2 : Domain classifier ─────────────────────────────
print("\n[2/4] Test domain classifier...")

DOMAIN_PROMPTS = {
    "radiology":   "chest x-ray radiograph lung heart mediastinum",
    "pathology":   "histology biopsy tissue stain microscopy pathology",
    "ophthalmology": "fundus retina optic disc macula eye glaucoma"
}

def classify_domain(image_path: str) -> tuple:
    """Classifie le domaine médical d'une image."""
    img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
    texts = tokenizer(list(DOMAIN_PROMPTS.values()))

    with torch.no_grad():
        img_features  = model.encode_image(img)
        text_features = model.encode_text(texts)
        img_features  = img_features  / img_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        scores = (img_features @ text_features.T).squeeze()

    domain_names = list(DOMAIN_PROMPTS.keys())
    best_idx     = scores.argmax().item()
    return domain_names[best_idx], float(scores[best_idx])

# Test sur 10 images IU-Xray
img_files = list(IU_IMAGES_DIR.rglob("*.png"))[:10]
print(f"  Test sur {len(img_files)} images IU-Xray:")
correct = 0
for img_path in img_files:
    domain, score = classify_domain(str(img_path))
    ok = "✅" if domain == "radiology" else "❌"
    print(f"    {ok} {img_path.name[:30]:30s} → {domain} ({score:.3f})")
    if domain == "radiology":
        correct += 1
accuracy = correct / len(img_files)
print(f"\n  Accuracy domain classifier : {accuracy:.0%} ({correct}/{len(img_files)})")

# ── Étape 3 : Embed les rapports IU-Xray ────────────────────
print("\n[3/4] Construction FAISS index IU-Xray...")

def embed_text(texts: list, batch_size: int = 32) -> np.ndarray:
    """Embed une liste de textes avec BiomedCLIP."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(batch)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_embeddings.append(feats.cpu().numpy())
        if i % 200 == 0:
            print(f"    Embedded {i}/{len(texts)}...")
    return np.concatenate(all_embeddings, axis=0).astype('float32')

# Charge les rapports IU-Xray
report_files = sorted(IU_REPORTS_DIR.glob("*.txt"))[:500]  # 500 rapports pour test
print(f"  Chargement {len(report_files)} rapports...")

reports_data = []
for f in report_files:
    txt = f.read_text(encoding='utf-8', errors='ignore').strip()
    if txt:
        # Tronque à 77 tokens (limite CLIP)
        reports_data.append({
            'id': f.stem,
            'text': txt[:300],
            'path': str(f)
        })

print(f"  {len(reports_data)} rapports valides")
texts = [r['text'] for r in reports_data]

print("  Embedding en cours...")
embeddings = embed_text(texts)
print(f"  Embeddings shape: {embeddings.shape}")

# Crée l'index FAISS
dim   = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)   # Inner product = cosine similarity (après normalisation)
index.add(embeddings)
print(f"  Index FAISS créé : {index.ntotal} vecteurs, dim={dim}")

# Sauvegarde
faiss.write_index(index, str(VECTOR_DIR / 'iu_xray_index.faiss'))
with open(VECTOR_DIR / 'iu_xray_metadata.json', 'w') as f:
    json.dump(reports_data, f, indent=2)
print(f"  ✅ Sauvegardé → {VECTOR_DIR}/iu_xray_index.faiss")

# ── Étape 4 : Test retrieval + adaptive k ───────────────────
print("\n[4/4] Test retrieval adaptatif...")

def retrieve(query_text: str, k: int = 5, ratio_threshold: float = 0.85) -> list:
    """Récupère les k documents les plus similaires avec filtrage par ratio."""
    tokens = tokenizer([query_text])
    with torch.no_grad():
        q_feat = model.encode_text(tokens)
        q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)

    q_vec = q_feat.cpu().numpy().astype('float32')
    scores, indices = index.search(q_vec, k)
    scores  = scores[0]
    indices = indices[0]

    # Similarity ratio filter (MMed-RAG Section 3.2)
    top_score = scores[0]
    filtered  = []
    for score, idx in zip(scores, indices):
        if idx < 0:
            continue
        ratio = score / (top_score + 1e-8)
        if ratio >= ratio_threshold:
            filtered.append({
                'report_id': reports_data[idx]['id'],
                'score':     float(score),
                'ratio':     float(ratio),
                'text':      reports_data[idx]['text'][:150]
            })

    return filtered

# Tests
test_queries = [
    "cardiomegaly pleural effusion chest x-ray",
    "normal chest no acute findings",
    "pneumonia consolidation right lower lobe",
    "atelectasis bilateral lung opacity"
]

for query in test_queries:
    results = retrieve(query, k=5, ratio_threshold=0.85)
    print(f"\n  Query: '{query}'")
    print(f"  → {len(results)} docs après filtrage (ratio ≥ 0.85)")
    if results:
        print(f"     Top score: {results[0]['score']:.4f}")
        print(f"     Top doc  : {results[0]['text'][:80]}...")

print("\n✅ Module 2 terminé !")
print(f"   FAISS index : {VECTOR_DIR}/iu_xray_index.faiss")
print(f"   Metadata    : {VECTOR_DIR}/iu_xray_metadata.json")