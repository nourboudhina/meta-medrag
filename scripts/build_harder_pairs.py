import json
import random
from pathlib import Path

random.seed(42)

# Charge les paires existantes
existing = json.load(open('data/processed/contrastive_pairs.json'))
print(f'Paires existantes: {len(existing)}')

# Ajoute des paires ambigues (ni clairement known ni unknown)
# Questions moderement difficiles — le modele hesite
AMBIGUOUS = [
    # Pathologies moderement rares
    "What is the significance of mild cardiomegaly with pulmonary vascular congestion?",
    "Can you identify signs of early interstitial lung disease in this image?",
    "What does bilateral hilar lymphadenopathy suggest in this context?",
    "Is there evidence of mild pulmonary arterial hypertension?",
    "What is the differential diagnosis for this parenchymal opacity?",
    "Can you describe the pattern of consolidation visible in this image?",
    "What is the clinical significance of an enlarged cardiac silhouette in elderly patients?",
    "Is there evidence of early fibrosis in the lung bases?",
    "Can you identify any mediastinal abnormalities in this chest radiograph?",
    "What does the presence of Kerley B lines indicate?",
    "Is there evidence of a small pleural effusion on the left?",
    "What is the radiological appearance of mild pulmonary edema?",
    "Can you identify subtle signs of pneumonia in this radiograph?",
    "What does an air bronchogram sign indicate in this context?",
    "Is there evidence of subtle pneumothorax at the apex?",
    # Questions ouvertes moderement difficiles
    "Describe the vascular markings in this chest radiograph.",
    "What is the significance of tracheal deviation in this image?",
    "Can you assess the bone density in this radiograph?",
    "What does the presence of calcified granulomas suggest?",
    "Describe the diaphragmatic contour in this image.",
]

# Label 0.5 n'existe pas en classification binaire
# On divise en 10 known (0) et 10 unknown (1) mais avec des activations ambigues attendues
ambiguous_known   = [{"question": q, "label": 0, "source": "ambiguous", "domain": "radiology"} 
                     for q in AMBIGUOUS[:10]]
ambiguous_unknown = [{"question": q, "label": 1, "source": "ambiguous", "domain": "radiology"} 
                     for q in AMBIGUOUS[10:]]

# Paires harder existantes — questions plus specifiques
HARDER_KNOWN = [
    "Is there cardiomegaly based on the cardiothoracic ratio?",
    "Are the costophrenic angles blunted?",
    "Is the trachea midline?",
    "Are there any rib fractures visible?",
    "Is the aortic knob prominent?",
]

HARDER_UNKNOWN = [
    "What is the typical radiological appearance of pulmonary alveolar proteinosis?",
    "Can you identify signs of Swyer-James syndrome in this radiograph?",
    "What does unilateral hyperlucency suggest in pediatric chest radiographs?",
    "Describe the radiological features of pulmonary sequestration.",
    "What are the imaging characteristics of bronchogenic cysts?",
]

harder_known   = [{"question": q, "label": 0, "source": "harder_known", "domain": "radiology"} 
                  for q in HARDER_KNOWN]
harder_unknown = [{"question": q, "label": 1, "source": "harder_unknown", "domain": "radiology"} 
                  for q in HARDER_UNKNOWN]

# Combine tout
new_pairs = existing + ambiguous_known + ambiguous_unknown + harder_known + harder_unknown
random.shuffle(new_pairs)

Path('data/processed').mkdir(parents=True, exist_ok=True)
json.dump(new_pairs, open('data/processed/contrastive_pairs_v2.json','w'), indent=2)

known   = sum(1 for x in new_pairs if x['label']==0)
unknown = sum(1 for x in new_pairs if x['label']==1)
print(f'Nouvelles paires: {len(new_pairs)} (Known: {known} | Unknown: {unknown})')