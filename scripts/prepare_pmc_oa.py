import json
import random
from pathlib import Path

random.seed(42)
Path('data/raw/pmc_oa').mkdir(parents=True, exist_ok=True)

TEMPLATES = [
    ("Histopathological section shows {finding} with {pattern} pattern.",
     "Pathology report: {finding} confirmed."),
    ("Microscopy reveals {finding} characterized by {pattern}.",
     "Diagnosis: {finding}."),
    ("Tissue biopsy demonstrates {finding} with {pattern} morphology.",
     "Histological diagnosis: {finding}."),
]

FINDINGS = [
    "adenocarcinoma", "squamous cell carcinoma", "lymphoma", "melanoma",
    "glioblastoma", "hepatocellular carcinoma", "renal cell carcinoma",
    "normal tissue", "inflammatory infiltrate", "fibrosis",
    "necrosis", "granuloma", "hyperplasia", "metaplasia", "dysplasia",
]

PATTERNS = [
    "glandular", "solid", "papillary", "cribriform", "micropapillary",
    "nested", "trabecular", "diffuse", "nodular", "mixed",
]

items = []
for i in range(500):
    tmpl = random.choice(TEMPLATES)
    finding = random.choice(FINDINGS)
    pattern = random.choice(PATTERNS)
    text = tmpl[0].format(finding=finding, pattern=pattern)
    answer = tmpl[1].format(finding=finding)
    items.append({
        'doc_id':   f'pmc_oa_{i}',
        'question': 'Describe the pathological findings in this medical image.',
        'answer':   answer,
        'domain':   'pathology',
        'image':    None,
        'text':     text,
        'report':   text,
        'split':    'train' if i < 400 else 'test',
    })

json.dump(items, open('data/raw/pmc_oa/splits.json','w',encoding='utf-8'), indent=2)
print(f'PMC-OA synthetic: {len(items)} items')
print(f'  Train: {sum(1 for x in items if x["split"]=="train")}')
print(f'  Test:  {sum(1 for x in items if x["split"]=="test")}')
print(f'Exemple: {items[0]["text"]}')
print(f'Reponse: {items[0]["answer"]}')