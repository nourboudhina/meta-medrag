import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import torch
import json
import sys
from pathlib import Path

sys.path.insert(0, r'C:\Users\Nour\Desktop\memoire\LLaVA-Med')

from transformers import AutoTokenizer
from llava.model import LlavaMistralForCausalLM

MODEL_PATH = 'microsoft/llava-med-v1.5-mistral-7b'
OUTPUT_DIR = Path('data/activations')
LAYERS     = [-2, -5, -8, -11, -15]
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
        "Is cardiomegaly present when heart exceeds 50 percent of chest width?",
        "What is a pleural effusion?",
        "Where are the hilar structures located?",
        "What does opacification of the lung indicate?",
        "What is the normal cardiothoracic ratio?",
        "What is atelectasis in radiology?",
        "What does consolidation mean in chest radiology?",
        "What is hyperinflation of the lungs?",
        "Are both hemidiaphragms visible on a normal X-ray?",
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
        "What is the normal cardiothoracic ratio in adults?",
        "What is a Fleischner sign in radiology?",
        "What does a large pleural effusion look like?",
        "What is consolidation in the right lower lobe?",
        "What are the causes of bilateral pulmonary infiltrates?",
        "What is an air bronchogram sign?",
        "What does cardiomegaly indicate on chest X-ray?",
        "What is a pulmonary cavity?",
        "What does right heart enlargement look like on X-ray?",
        "What is the significance of upper lobe blood diversion?",
        "What is a cephalization of pulmonary vessels?",
        "What does a normal chest X-ray show?",
        "What is lobar pneumonia?",
        "What is a miliary pattern on chest X-ray?",
        "What does bilateral lower lobe consolidation indicate?",
        "What is a pneumothorax?",
        "What does subcutaneous emphysema look like?",
        "What is a pleural plaque?",
        "What does a pulmonary mass suggest?",
        "What is tracheal deviation?",
        "What does an elevated right hemidiaphragm indicate?",
        "What is a hydropneumothorax?",
        "What does aortic calcification suggest?",
        "What is a Westermark sign?",
        "What does a normal diaphragm look like on X-ray?",
        "What is a perihilar bat-wing opacity?",
        "What does bilateral hilar prominence suggest?",
        "What is the significance of absent lung markings?",
        "What is a pulmonary hamartoma?",
        "What does interstitial lung disease look like?",
        "What is a pulmonary sequestration?",
        "What does a round pneumonia look like?",
        "What is a lung contusion on chest X-ray?",
        "What is a pneumomediastinum?",
        "What does a normal trachea look like on X-ray?",
        "What is a lung infarction?",
        "What does bilateral pleural effusion indicate?",
        "What is a transudate versus exudate effusion?",
        "What does a normal rib cage look like on X-ray?",
        "What is a pulmonary arteriovenous malformation?",
        "What does a rib fracture look like on chest X-ray?",
        "What is an empyema?",
        "What does a normal thymus look like in a child?",
        "What is a pulmonary bleb?",
        "What does a right paratracheal mass suggest?",
        "What is sarcoidosis on chest X-ray?",
        "What does a normal mediastinal width measure?",
        "What is a pulmonary laceration?",
        "What does bilateral opacification of the lungs indicate?",
        "What is a bronchocele on chest X-ray?",
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
        "What is the specific microbiome composition in this patient lungs?",
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
        "What is the exact neuroendocrine marker level from this X-ray?",
        "Does this show features of a condition first described in 2024?",
        "What is the specific proteome signature of this lung cancer?",
        "What is the exact VEGF level associated with this effusion?",
        "Does this match the presentation in a 2025 single-patient report?",
        "What is the specific mitochondrial dysfunction pattern here?",
        "What is the exact surfactant composition deficiency in this case?",
        "Does this match findings from a 2026 preprint not yet peer-reviewed?",
        "What is the specific CRISPR target for treating this condition?",
        "What is the exact piezoelectric signal from this bone lesion?",
        "Does this show features described in a 2025 conference abstract only?",
        "What is the specific RNA methylation pattern in this tumor?",
        "What is the exact photon count in this digital X-ray image?",
        "What is the consciousness level of this patient from the X-ray?",
        "What is the specific quantum state of the tumor cells here?",
        "Can you identify this finding from a classified military study?",
        "What is the exact DNA methylation profile of this pleural fluid?",
        "Does this match findings from an unpublished 2026 clinical trial?",
        "What is the specific T-cell receptor clonotype in this patient?",
        "Can you identify the exact single-nucleotide polymorphism here?",
        "What is the telomere length of the tumor cells in this image?",
        "What is the exact lipidomic signature of this pleural effusion?",
        "Can you identify the epigenetic clock age from this chest X-ray?",
        "What is the specific retrotransposon activity in this tumor?",
        "Does this match the presentation from a 2025 Nature Medicine paper?",
        "What is the exact ferroptosis pathway activation in this lesion?",
        "Can you identify the specific senescence biomarker from this image?",
        "What is the exact metabolomic fingerprint of this lung nodule?",
        "Does this show features from a 2026 preprint on machine learning?",
        "What is the specific exosome miRNA profile from this effusion?",
        "Can you identify the circulating tumor DNA level from this X-ray?",
        "What is the exact chromatin remodeling pattern in this tumor?",
        "Does this match findings from an embargoed 2026 clinical study?",
        "What is the specific autophagy flux in the cells of this lesion?",
        "Can you identify the exact ferroptosis resistance mechanism here?",
        "What is the spatial transcriptomics profile of this lung mass?",
        "Does this match the presentation in an alien medical textbook?",
    ]
}

def main():
    print("=" * 60)
    print("Meta-MedRAG — Extraction activations (CPU final)")
    print("=" * 60)
    print(f"CUDA disponible: {torch.cuda.is_available()} (doit être False)")

    print("\n[1/3] Chargement modèle sur CPU...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    model = LlavaMistralForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map={'': 'cpu'},
        low_cpu_mem_usage=True
    )
    model.eval()
    print(f"✅ Modèle chargé — {model.config.num_hidden_layers} couches")

    print("\n[2/3] Extraction des activations...")
    all_activations = {layer: [] for layer in LAYERS}
    all_labels = []

    for label, questions in QUESTIONS.items():
        lbl_int = 0 if label == 'known' else 1
        print(f"\n  '{label}' — {len(questions)} questions:")
        for i, q in enumerate(questions):
            if i % 20 == 0:
                print(f"    {i}/{len(questions)}...")
            inputs = tokenizer(
                q, return_tensors='pt',
                truncation=True, max_length=128
            )
            with torch.no_grad():
                out = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    output_hidden_states=True
                )
            n = len(out.hidden_states) - 1
            for layer in LAYERS:
                idx = n + layer + 1
                act = out.hidden_states[idx][0].mean(dim=0).float()
                all_activations[layer].append(act)
            all_labels.append(lbl_int)

    print("\n[3/3] Sauvegarde...")
    for layer in LAYERS:
        t = torch.stack(all_activations[layer])
        p = OUTPUT_DIR / f'layer_{layer}.pt'
        torch.save(t, p)
        print(f"  layer {layer}: {t.shape} → {p}")

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

    print(f"\n✅ Terminé! {len(all_labels)} activations sauvegardées dans {OUTPUT_DIR}/")

if __name__ == '__main__':
    main()