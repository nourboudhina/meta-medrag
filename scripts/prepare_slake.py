import json
from pathlib import Path

Path('data/raw/slake').mkdir(parents=True, exist_ok=True)
raw = json.load(open('data/raw/slake_train.json', encoding='utf-8'))

# Filtre anglais uniquement
items = []
for i, x in enumerate(raw):
    if x.get('q_lang', 'en') != 'en':
        continue
    items.append({
        'doc_id':   f'slake_{i}',
        'question': x.get('question', ''),
        'answer':   x.get('answer', ''),
        'q_type':   x.get('q_type', ''),
        'domain':   'radiology',
        'image':    None,
        'text':     x.get('question', ''),
        'report':   x.get('answer', ''),
        'split':    'train',
    })

json.dump(items, open('data/raw/slake/splits.json', 'w', encoding='utf-8'), indent=2)
print(f'SLAKE EN: {len(items)} questions')
print(f'Exemple: {items[0]["question"]} → {items[0]["answer"]}')