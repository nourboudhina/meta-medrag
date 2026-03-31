import json
import xml.etree.ElementTree as ET
from pathlib import Path

rep_dir  = Path('data/raw/iu_xray/reports')
img_dir  = Path('data/raw/iu_xray/images')
xml_dir  = Path('data/raw/iu_xray/reports_raw')
out_path = Path('data/raw/iu_xray/splits.json')

# Index images par numero CXR  ->  CXR1000 -> [path1, path2]
img_index = {}
for img in img_dir.glob('**/*.png'):
    cxr_id = img.name.split('_')[0]  # CXR1000
    if cxr_id not in img_index:
        img_index[cxr_id] = []
    img_index[cxr_id].append(str(img))

print(f'Images indexees: {len(img_index)} IDs uniques')

# Parse chaque XML
items = []
xml_files = sorted(xml_dir.glob('**/*.xml'))

for xml_file in xml_files:
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Recupere l ID CXR
        uid_elem = root.find('uId')
        if uid_elem is None:
            continue
        cxr_id = uid_elem.get('id', '')  # ex: CXR1
        if not cxr_id:
            continue

        # Recupere le texte du rapport
        sections = {}
        for abstract in root.findall('.//AbstractText'):
            label = abstract.get('Label', '').upper()
            text  = (abstract.text or '').strip()
            if text:
                sections[label] = text

        if not sections:
            continue

        lines = []
        for section in ['FINDINGS', 'IMPRESSION', 'INDICATION', 'COMPARISON']:
            if section in sections:
                lines.append(f'{section}: {sections[section]}')

        report_text = '\n'.join(lines)
        if not report_text:
            continue

        # Cherche les images correspondantes
        img_paths = img_index.get(cxr_id, [])
        img_path  = img_paths[0] if img_paths else None

        # Question et reponse depuis IMPRESSION
        impression = sections.get('IMPRESSION', '')
        answer = impression[:200] if impression else 'Normal'

        items.append({
            'doc_id':   cxr_id,
            'image':    img_path,
            'text':     report_text,
            'question': 'Describe the key findings in this chest X-ray.',
            'answer':   answer,
            'report':   report_text,
            'domain':   'radiology',
            'split':    '',
        })

    except Exception as e:
        continue

print(f'Items construits: {len(items)}')

# Split 70/15/15
n       = len(items)
n_train = int(n * 0.70)
n_val   = int(n * 0.15)

for i, item in enumerate(items):
    if i < n_train:
        item['split'] = 'train'
    elif i < n_train + n_val:
        item['split'] = 'val'
    else:
        item['split'] = 'test'

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(items, f, indent=2, ensure_ascii=False)

counts = {s: sum(1 for x in items if x['split'] == s)
          for s in ['train', 'val', 'test']}
print(f'Split: {counts}')
print(f'Avec image: {sum(1 for x in items if x["image"])}')
print(f'Sans image: {sum(1 for x in items if not x["image"])}')
print(f'Sauvegarde: {out_path}')