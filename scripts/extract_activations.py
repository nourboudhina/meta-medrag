import torch
import json
import os
from pathlib import Path
from PIL import Image
import sys

sys.path.insert(0, r'C:\Users\Nour\Desktop\memoire\LLaVA-Med')

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

# ── Config ──────────────────────────────────────────────
MODEL_PATH   = 'microsoft/llava-med-v1.5-mistral-7b'
OUTPUT_DIR   = Path('data/activations')
LAYERS       = [-2, -5, -8, -11, -15]
NUM_SAMPLES  = 200
DEVICE       = 'cpu'   # change to 'cuda' on Toulouse GPUs
# ────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 200 medical questions — 100 "known" (easy) + 100 "unknown" (hard)
QUESTIONS = {
    "known": [
        "What organ is shown in this chest X-ray?",
        "Is this a frontal or lateral chest X-ray view?",
        "Are the lungs visible in this image?",
        "What is the normal appearance of the cardiac silhouette?",
        "Is the trachea midline in this chest X-ray?",
        "What structures form the mediastinum?",
        "Are there any obvious fractures in this image?",
        "What is the costophrenic angle?",
        "Is the diaphragm visible in this X-ray?",
        "What does a normal lung field look like on X-ray?",
        "Is cardiomegaly present when the heart exceeds 50% of chest width?",
        "What is a pleural effusion?",
        "Where are the hilar structures located?",
        "What does opacification of the lung indicate?",
        "Is this chest X-ray of acceptable diagnostic quality?",
        "What is the normal cardiothoracic ratio?",
        "Are the clavicles symmetrical in this image?",
        "What is atelectasis in radiology?",
        "Is there pneumothorax visible in this image?",
        "What does consolidation mean in chest radiology?",
        "Describe the normal appearance of ribs on chest X-ray.",
        "Is the lung parenchyma clear in this image?",
        "What is hyperinflation of the lungs?",
        "Are both hemidiaphragms visible?",
        "What is a Kerley B line?",
        "Is there any tracheal deviation visible?",
        "What does pulmonary vascular congestion look like?",
        "Is the aortic knob visible in this X-ray?",
        "What is the significance of air bronchograms?",
        "Describe what pneumonia looks like on chest X-ray.",
        "Is there any pleural thickening visible?",
        "What is a retrocardiac opacity?",
        "Are the lung apices clear?",
        "What does a ground-glass opacity indicate?",
        "Is there cardiomegaly in this image?",
        "What is pulmonary edema?",
        "Are the soft tissues normal in appearance?",
        "What is the significance of a widened mediastinum?",
        "Is there any foreign body visible in this X-ray?",
        "What does a normal chest X-ray report conclude?",
        "Is there evidence of previous surgery in this image?",
        "What is a pulmonary nodule?",
        "Are the bones intact in this chest X-ray?",
        "What does bilateral infiltrate suggest?",
        "Is the cardiac apex pointing left as expected?",
        "What is the significance of perihilar haziness?",
        "Are there any calcifications visible?",
        "What does a Hampton hump indicate?",
        "Is there any subcutaneous emphysema?",
        "What is the normal thickness of the pleura?",
        "Describe the appearance of a normal mediastinum.",
        "What is a tension pneumothorax?",
        "Is the trachea displaced in this image?",
        "What does increased interstitial markings mean?",
        "Is there any evidence of heart failure in this X-ray?",
        "What is the significance of blunted costophrenic angles?",
        "Are there any pacemaker leads visible?",
        "What does a normal cardiac silhouette look like?",
        "Is there any lung mass or nodule visible?",
        "What is the significance of an elevated hemidiaphragm?",
        "Describe the appearance of COPD on chest X-ray.",
        "Is there consolidation in the right lower lobe?",
        "What does perihilar infiltrate suggest?",
        "Are the pulmonary vessels normal in caliber?",
        "What is a silhouette sign in radiology?",
        "Is there any rib fracture visible?",
        "What does linear atelectasis look like?",
        "Is there opacity in the left costophrenic angle?",
        "What is the normal appearance of the aorta on X-ray?",
        "Are there any pleural plaques visible?",
        "What does a small pleural effusion look like?",
        "Is there any mediastinal shift?",
        "What is pulmonary fibrosis?",
        "Are there any signs of infection in this X-ray?",
        "What is a lung abscess on chest X-ray?",
        "Is there interstitial lung disease visible?",
        "What does bilateral hilar lymphadenopathy suggest?",
        "Are the hila of normal size?",
        "What is aspiration pneumonia?",
        "Is there any cavitation visible in this X-ray?",
        "What does a normal lateral chest X-ray show?",
        "Are there any signs of sarcoidosis in this image?",
        "What is the retrosternal space?",
        "Is there any upper lobe diversion of blood flow?",
        "What does a bat-wing pattern suggest?",
        "Are the lungs hyperinflated?",
        "What is a pulmonary infarct?",
        "Is the heart size normal in this image?",
        "What does increased lung markings indicate?",
        "Are the costophrenic angles sharp?",
        "What is a hydropneumothorax?",
        "Is there any aortic calcification?",
        "What does a normal pediatric chest X-ray show?",
        "Are the vertebrae visible behind the heart?",
        "What is the posterior costophrenic angle?",
        "Is there any evidence of tuberculosis in this X-ray?",
        "What does a miliary pattern indicate?",
        "Are there any chest tubes visible?",
        "What is a subclavian catheter on chest X-ray?",
        "Is there any evidence of malignancy in this image?",
        "What does lobar collapse look like on chest X-ray?",
    ],
    "unknown": [
        "What is the exact Hounsfield unit value of this lesion?",
        "Can you diagnose Erdheim-Chester disease from this X-ray?",
        "What is the mutation status of the EGFR gene in this tumor?",
        "Does this patient have yellow nail syndrome?",
        "What is the prognosis of this specific lymphangioleiomyomatosis presentation?",
        "Can you identify Birt-Hogg-Dube syndrome from this chest X-ray?",
        "What is the 5-year survival rate for this specific lung adenocarcinoma stage?",
        "Does this X-ray show features of Swyer-James syndrome?",
        "What chemotherapy protocol should be used for this finding?",
        "Can you identify alpha-1-antitrypsin deficiency from this image alone?",
        "What is the exact size of the pulmonary embolism in this X-ray?",
        "Does this image show Hughes-Stovin syndrome?",
        "What is the radiation dose absorbed by this patient during the scan?",
        "Can you diagnose pulmonary alveolar proteinosis severity from X-ray?",
        "What is the specific bacterial strain causing this pneumonia?",
        "Does this show features of Mounier-Kuhn syndrome?",
        "What is the exact FEV1/FVC ratio from this chest X-ray?",
        "Can you identify Hermansky-Pudlak syndrome pulmonary features?",
        "What is the tumor microenvironment composition in this mass?",
        "Does this X-ray show features of Williams-Campbell syndrome?",
        "What immunotherapy would be most effective for this patient?",
        "Can you identify the specific pleural mesothelioma subtype?",
        "What is the genetic mutation causing this cystic lung disease?",
        "Does this image show features of Bochdalek hernia in an adult?",
        "What is the exact pulmonary artery pressure from this X-ray?",
        "Can you diagnose primary ciliary dyskinesia from this image?",
        "What is the viral load of this COVID-19 patient from the X-ray?",
        "Does this show features of Castleman disease in the chest?",
        "What is the histological subtype of this interstitial lung disease?",
        "Can you identify the specific vasculitis type from this chest X-ray?",
        "What is the exact staging of this mesothelioma from plain X-ray?",
        "Does this X-ray show features of alveolar hemorrhage syndrome?",
        "What is the carbon monoxide diffusing capacity from this image?",
        "Can you identify plasma cell granuloma from this chest X-ray?",
        "What is the specific occupational exposure causing this pneumoconiosis?",
        "Does this image show features of lymphomatoid granulomatosis?",
        "What is the exact right ventricular pressure from this X-ray?",
        "Can you diagnose sclerosing mediastinitis from plain film alone?",
        "What is the tumor doubling time of this pulmonary nodule?",
        "Does this X-ray show features of Rosai-Dorfman disease?",
        "What is the specific drug causing this drug-induced lung disease?",
        "Can you identify the exact pleural fluid protein level from this image?",
        "What is the prognosis of this pulmonary hypertension subtype?",
        "Does this show features of congenital lobar emphysema in an adult?",
        "What is the exact neutrophil percentage in this pleural fluid?",
        "Can you diagnose pulmonary Langerhans cell histiocytosis stage?",
        "What is the specific allergen causing this hypersensitivity pneumonitis?",
        "Does this image show features of desquamative interstitial pneumonia?",
        "What is the exact forced vital capacity from this chest X-ray?",
        "Can you identify amyloidosis subtype from this chest X-ray?",
        "What recent 2025 AI model best processes this type of X-ray?",
        "What is the latest treatment guideline published in January 2026?",
        "Does this match the new WHO 2025 pneumonia classification?",
        "What is the current FDA-approved drug for this exact presentation?",
        "Can you apply the 2025 Fleischner Society updated guidelines here?",
        "What does the latest RSNA 2025 consensus say about this finding?",
        "Is this consistent with the new 2026 ATS diagnostic criteria?",
        "What is the most recent meta-analysis conclusion on this pattern?",
        "Does this match findings from the 2025 COVID long-haul studies?",
        "What is the updated TNM staging from the 2025 IASLC revision?",
        "What rare genetic syndrome affects only 12 patients worldwide?",
        "Can you identify this finding described only in a 2024 case report?",
        "What is the exact protein biomarker level from this plain X-ray?",
        "Does this match the presentation in the 2025 NEJM case series?",
        "What is the specific microbiome composition in this patient's lungs?",
        "Can you identify this finding from a 2023 single-center study?",
        "What is the exact T-cell subtype infiltrating this tumor?",
        "Does this match findings from an unpublished 2026 clinical trial?",
        "What is the specific epigenetic modification causing this pattern?",
        "Can you apply the 2025 revised Scadding staging for sarcoidosis?",
        "What is the exact complement pathway activation in this case?",
        "Does this show features described only in pediatric literature?",
        "What is the specific cytokine storm pattern in this COVID case?",
        "Can you identify this finding described only in Japanese literature?",
        "What is the exact immunoglobulin subclass causing this pattern?",
        "Does this match a finding from the 2025 ERS annual meeting?",
        "What is the specific mast cell subtype in this interstitial disease?",
        "Can you identify this ultra-rare condition affecting 3 families?",
        "What is the exact HLA type predisposing to this condition?",
        "Does this match the 2026 updated GOLD COPD classification?",
        "What is the specific exosome profile from this tumor?",
        "Can you identify findings from an embargoed 2026 study?",
        "What is the exact neuroendocrine marker level from this X-ray?",
        "Does this show features of a condition first described in 2024?",
        "What is the specific proteome signature of this lung cancer?",
        "Can you apply criteria from a retracted 2023 paper to this case?",
        "What is the exact VEGF level associated with this effusion?",
        "Does this match the presentation in a 2025 single-patient report?",
        "What is the specific mitochondrial dysfunction pattern here?",
        "Can you identify this finding using 2026 AI-specific guidelines?",
        "What is the exact surfactant composition deficiency in this case?",
        "Does this match findings from a 2026 preprint not yet peer-reviewed?",
        "What is the specific CRISPR target for treating this condition?",
        "Can you identify this pattern from quantum imaging research 2025?",
        "What is the exact piezoelectric signal from this bone lesion?",
        "Does this show features described in a 2025 conference abstract only?",
        "What is the specific RNA methylation pattern in this tumor?",
        "Can you identify this finding from a classified military study?",
        "What is the exact photon count in this digital X-ray image?",
        "Does this match findings from an alien medical textbook?",
        "What is the consciousness level of this patient from the X-ray?",
    ]
}

def extract_activations(model, tokenizer, image_processor, questions, label, layers, device):
    all_activations = {layer: [] for layer in layers}
    labels = []
    
    print(f"\nExtracting activations for {len(questions)} '{label}' questions...")
    
    for i, question in enumerate(questions):
        if i % 10 == 0:
            print(f"  Processing {i}/{len(questions)}...")
        
        # Tokenize question
        inputs = tokenizer(
            question,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        # Extract hidden states
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                output_hidden_states=True
            )
        
        hidden_states = outputs.hidden_states  # tuple of (num_layers+1) tensors
        num_layers = len(hidden_states) - 1
        
        # Extract from specified layers (negative indexing)
        for layer in layers:
            layer_idx = num_layers + layer + 1  # convert negative to positive index
            # Mean pool over sequence length
            activation = hidden_states[layer_idx][0].mean(dim=0).cpu().float()
            all_activations[layer].append(activation)
        
        labels.append(0 if label == 'known' else 1)
    
    # Stack into tensors
    for layer in layers:
        all_activations[layer] = torch.stack(all_activations[layer])
    
    return all_activations, labels

def main():
    print("=" * 60)
    print("Meta-MedRAG — Activation Extraction Script")
    print("=" * 60)
    
    # Load model
    print("\n[1/3] Loading LLaVA-Med model...")
    model_name = get_model_name_from_path(MODEL_PATH)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        MODEL_PATH, None, model_name,
        load_8bit=False, load_4bit=False,
        device_map=DEVICE
    )
    model.eval()
    print(f"Model loaded. Hidden layers: {model.config.num_hidden_layers}")
    
    # Extract activations
    print("\n[2/3] Extracting activations...")
    known_acts, known_labels = extract_activations(
        model, tokenizer, image_processor,
        QUESTIONS['known'], 'known', LAYERS, DEVICE
    )
    unknown_acts, unknown_labels = extract_activations(
        model, tokenizer, image_processor,
        QUESTIONS['unknown'], 'unknown', LAYERS, DEVICE
    )
    
    # Save
    print("\n[3/3] Saving activations...")
    all_labels = known_labels + unknown_labels
    
    for layer in LAYERS:
        combined = torch.cat([known_acts[layer], unknown_acts[layer]], dim=0)
        save_path = OUTPUT_DIR / f'layer_{layer}.pt'
        torch.save(combined, save_path)
        print(f"  Saved layer {layer}: shape {combined.shape} → {save_path}")
    
    # Save labels
    torch.save(torch.tensor(all_labels), OUTPUT_DIR / 'labels.pt')
    
    # Save metadata
    metadata = {
        'num_known': len(QUESTIONS['known']),
        'num_unknown': len(QUESTIONS['unknown']),
        'total': len(all_labels),
        'layers': LAYERS,
        'hidden_dim': known_acts[LAYERS[0]].shape[1],
        'label_encoding': {'known': 0, 'unknown': 1}
    }
    with open(OUTPUT_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Done! Files saved in {OUTPUT_DIR}/")
    print(f"   - layer_-2.pt through layer_-15.pt")
    print(f"   - labels.pt ({len(all_labels)} labels)")
    print(f"   - metadata.json")

if __name__ == '__main__':
    main()