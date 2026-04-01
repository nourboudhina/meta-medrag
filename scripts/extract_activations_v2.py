import torch
import json
import os
from pathlib import Path

# Force CPU absolument
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH  = 'microsoft/llava-med-v1.5-mistral-7b'
OUTPUT_DIR  = Path('data/activations')
LAYERS      = [-2, -5, -8, -11, -15]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUESTIONS = {
    "known": [
        "What organ is shown in this chest X-ray?",
        "Is this a frontal or lateral chest X-ray view?",
        "Are the lungs visible in this image?",
        "What is the normal appearance of the cardiac silhouette?",
        "Is the trachea midline in this chest X-ray?",
        "What structures form the mediastinum?",
        "What is the costophrenic angle?",
        "Is the diaphragm visible in this X-ray?",
        "What does a normal lung field look like on X-ray?",
        "Is cardiomegaly present when the heart exceeds 50% of chest width?",
        "What is a pleural effusion?",
        "Where are the hilar structures located?",
        "What does opacification of the lung indicate?",
        "What is the normal cardiothoracic ratio?",
        "What is atelectasis in radiology?",
        "What does consolidation mean in chest radiology?",
        "What is hyperinflation of the lungs?",
        "Are both hemidiaphragms visible?",
        "What is a Kerley B line?",
        "What does pulmonary vascular congestion look like?",
        "What is the significance of air bronchograms?",
        "Describe what pneumonia looks like on chest X-ray.",
        "What is a retrocardiac opacity?",
        "What does a ground-glass opacity indicate?",
        "What is pulmonary edema?",
        "What is the significance of a widened mediastinum?",
        "What is a pulmonary nodule?",
        "What does bilateral infiltrate suggest?",
        "What is the significance of perihilar haziness?",
        "What does a Hampton hump indicate?",
        "What is the normal thickness of the pleura?",
        "Describe the appearance of a normal mediastinum.",
        "What is a tension pneumothorax?",
        "What does increased interstitial markings mean?",
        "What is the significance of blunted costophrenic angles?",
        "What does a normal cardiac silhouette look like?",
        "What is the significance of an elevated hemidiaphragm?",
        "Describe the appearance of COPD on chest X-ray.",
        "What does perihilar infiltrate suggest?",
        "What is a silhouette sign in radiology?",
        "What does linear atelectasis look like?",
        "What is the normal appearance of the aorta on X-ray?",
        "What is pulmonary fibrosis?",
        "What is a lung abscess on chest X-ray?",
        "What does bilateral hilar lymphadenopathy suggest?",
        "What is aspiration pneumonia?",
        "What does a normal lateral chest X-ray show?",
        "What is the retrosternal space?",
        "What does a bat-wing pattern suggest?",
        "What is a pulmonary infarct?",
    ],
    "unknown": [
        "What is the exact Hounsfield unit value of this lesion?",
        "Can you diagnose Erdheim-Chester disease from this X-ray?",
        "What is the mutation status of the EGFR gene in this tumor?",
        "Does this patient have yellow nail syndrome?",
        "What is the prognosis of this specific lymphangioleiomyomatosis?",
        "Can you identify Birt-Hogg-Dube syndrome from this chest X-ray?",
        "What is the 5-year survival rate for this lung adenocarcinoma stage?",
        "Does this X-ray show features of Swyer-James syndrome?",
        "What chemotherapy protocol should be used for this finding?",
        "Can you identify alpha-1-antitrypsin deficiency from this image?",
        "What is the exact size of the pulmonary embolism in this X-ray?",
        "Does this image show Hughes-Stovin syndrome?",
        "What is the radiation dose absorbed by this patient?",
        "What is the specific bacterial strain causing this pneumonia?",
        "Does this show features of Mounier-Kuhn syndrome?",
        "What is the exact FEV1/FVC ratio from this chest X-ray?",
        "What is the tumor microenvironment composition in this mass?",
        "Does this X-ray show features of Williams-Campbell syndrome?",
        "What immunotherapy would be most effective for this patient?",
        "Can you identify the specific pleural mesothelioma subtype?",
        "What is the genetic mutation causing this cystic lung disease?",
        "What is the exact pulmonary artery pressure from this X-ray?",
        "Can you diagnose primary ciliary dyskinesia from this image?",
        "What is the viral load of this COVID-19 patient from the X-ray?",
        "Does this show features of Castleman disease in the chest?",
        "What is the histological subtype of this interstitial lung disease?",
        "What is the exact staging of this mesothelioma from plain X-ray?",
        "What is the carbon monoxide diffusing capacity from this image?",
        "What is the specific occupational exposure causing this pneumoconiosis?",
        "What is the exact right ventricular pressure from this X-ray?",
        "Can you diagnose sclerosing mediastinitis from plain film?",
        "What is the tumor doubling time of this pulmonary nodule?",
        "What is the specific drug causing this drug-induced lung disease?",
        "What is the prognosis of this pulmonary hypertension subtype?",
        "What is the exact neutrophil percentage in this pleural fluid?",
        "What is the specific allergen causing this hypersensitivity?",
        "What is the exact forced vital capacity from this chest X-ray?",
        "What recent 2025 AI model best processes this type of X-ray?",
        "What is the latest treatment guideline published in January 2026?",
        "Does this match the new WHO 2025 pneumonia classification?",
        "What is the current FDA-approved drug for this exact presentation?",
        "Can you apply the 2025 Fleischner Society updated guidelines here?",
        "What does the latest RSNA 2025 consensus say about this finding?",
        "Is this consistent with the new 2026 ATS diagnostic criteria?",
        "What is the most recent meta-analysis conclusion on this pattern?",
        "What is the updated TNM staging from the 2025 IASLC revision?",
        "What rare genetic syndrome affects only 12 patients worldwide?",
        "Can you identify this finding described only in a 2024 case report?",
        "What is the exact protein biomarker level from this plain X-ray?",
        "What is the specific microbiome composition in this patient's lungs?",
    ]
}

def main():
    print("=" * 60)
    print("Meta-MedRAG — Activation Extraction v2 (CPU forcé)")
    print("=" * 60)

    print(f"\nCUDA visible: {torch.cuda.is_available()}")
    print("Doit afficher False — si True, redémarre le terminal\n")

    print("[1/3] Chargement du tokenizer et modèle sur CPU...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map='cpu',
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True
    )
    model.eval()
    print(f"✅ Modèle chargé — couches: {model.config.num_hidden_layers}")

    print("\n[2/3] Extraction des activations...")
    all_activations = {layer: [] for layer in LAYERS}
    all_labels = []

    for label, questions in QUESTIONS.items():
        print(f"\n  Classe '{label}' ({len(questions)} questions):")
        for i, question in enumerate(questions):
            if i % 10 == 0:
                print(f"    {i}/{len(questions)}...")

            inputs = tokenizer(
                question,
                return_tensors='pt',
                truncation=True,
                max_length=128
            )

            with torch.no_grad():
                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    output_hidden_states=True
                )

            hidden_states = outputs.hidden_states
            num_layers = len(hidden_states) - 1

            for layer in LAYERS:
                idx = num_layers + layer + 1
                act = hidden_states[idx][0].mean(dim=0).cpu().float()
                all_activations[layer].append(act)

            all_labels.append(0 if label == 'known' else 1)

    print("\n[3/3] Sauvegarde...")
    for layer in LAYERS:
        combined = torch.stack(all_activations[layer])
        path = OUTPUT_DIR / f'layer_{layer}.pt'
        torch.save(combined, path)
        print(f"  layer {layer}: shape {combined.shape} → {path}")

    torch.save(torch.tensor(all_labels), OUTPUT_DIR / 'labels.pt')

    meta = {
        'num_known': len(QUESTIONS['known']),
        'num_unknown': len(QUESTIONS['unknown']),
        'total': len(all_labels),
        'layers': LAYERS,
        'hidden_dim': all_activations[LAYERS[0]][0].shape[0],
        'label_encoding': {'known': 0, 'unknown': 1}
    }
    with open(OUTPUT_DIR / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Terminé! Fichiers dans {OUTPUT_DIR}/")

if __name__ == '__main__':
    main()