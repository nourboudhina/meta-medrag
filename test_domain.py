import sys; sys.path.insert(0,'.')
import json
from pathlib import Path
from src.module2_retrieval.domain_classifier import DomainClassifier

clf = DomainClassifier()
data = json.load(open('data/raw/vqa_rad/splits.json'))
imgs = [x['image'] for x in data if x.get('image') and Path(x['image']).exists()][:10]

print('VQA-RAD Domain Test (10 images):')
results = {}
for img in imgs:
    domain = clf.classify(img)
    results[domain] = results.get(domain, 0) + 1
    print(f'  {Path(img).name[:30]} -> {domain}')

print(f'\nDistribution: {results}')
print('Domain classifier OK ✓')